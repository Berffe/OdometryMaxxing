"""
Closed-loop visual controller: scheduled LQR baseline + manual experimental trim.

Constraint: once the visual controller is active, commands use ONLY visual data
(target offset, normalized optical flow, divergence, area_fraction). No PX4
position/velocity enters the control law.

Per lateral axis (roll <- x, pitch <- y), the identified open-loop model is
    x[k+1] = A(af) x[k] + B(af) u[k],     x = [offset, flow_norm]^T
with A, B scheduled on area_fraction (af). solve_discrete_lqr gives the optimal
baseline gain K_lqr = [k_p, k_d]; the command is
    u = sign * ( -(K x) ),   K = [s_p*k_p, s_d*k_d]
where (s_p, s_d) are MANUAL per-component scales. The LQR fixes the gain shape;
the scales are turned by hand to tune behavior, since the identified models are
local and optimistic about loop/sensor lag at the 2 Hz control rate.

Damping note: in the open-loop fit the optimal k_d is ~30x smaller than k_p, so
the flow (velocity) term barely acts. *_damp_scale exists to deliberately raise
it. If the loop still oscillates after raising damping, relax *_slew_rate /
command_filter_alpha: a tight slew rate rate-limits the damping command itself
and reintroduces the lag the damping was meant to remove.

Thrust uses the scalar divergence model d[k+1] = a d[k] + b (thrust - hover),
with LQR feedback on the divergence error plus a visual-only integral term.

Commands are passed through a purely internal shaper (soft saturation ->
first-order filter -> slew limit -> clamp). This uses only previous commands,
never PX4 state, and removes bang-bang excitation of the slow image dynamics.
"""

import math
import numpy as np

try:
    from .lqr import ScheduledLQR
    from .state import AttitudeSetpoint, FlowResult, TargetEstimate
except ImportError:
    from lqr import ScheduledLQR
    from state import AttitudeSetpoint, FlowResult, TargetEstimate


# Open-loop scheduled models from the reconstructed calibration.
# Entry: (area_fraction, A, B) for [offset[k+1], flow[k+1]]^T = A x + B u.
ROLL_STATE_MODELS = (
    (0.066, [[0.7508, 0.3322], [-0.5140, 0.0995]], [[-0.28352], [-0.91928]]),
    (0.133, [[0.6785, 0.2522], [-0.7866, 0.0246]], [[-0.40704], [-1.06228]]),
    (0.215, [[0.7334, 0.3234], [-0.4976, 0.0619]], [[-0.36399], [-0.88281]]),
    (0.511, [[0.7437, 0.4134], [-0.4053, 0.0191]], [[-0.48681], [-0.84894]]),
)

PITCH_STATE_MODELS = (
    (0.066, [[0.6834, 0.2032], [-0.8344, 0.0830]], [[-0.56504], [-1.22694]]),
    (0.133, [[0.7885, 0.0946], [-0.4753, -0.2675]], [[-0.88419], [-1.93529]]),
    (0.215, [[0.8739, 0.1809], [-0.3154, -0.1988]], [[-0.77769], [-1.65307]]),
    (0.511, [[0.8375, 0.2004], [-0.3098, -0.1282]], [[-0.62260], [-1.56707]]),
)

# Scalar divergence model: d[k+1] = a d[k] + b (thrust[k] - hover).
THRUST_DIVERGENCE_MODELS = (
    (0.066, [[0.9856]], [[-0.0818]]),
    (0.133, [[0.9302]], [[-0.1294]]),
    (0.215, [[0.9481]], [[-0.1192]]),
    (0.511, [[1.0315]], [[-0.1068]]),
)


class ControlLaw:
    def __init__(
        self,
        hover_thrust: float = 0.73,
        yaw_setpoint: float = 0.0,
        divergence_setpoint: float = 0.0,  # 0 = visual hover; raise slowly to descend.

        # --- Baseline LQR cost (gain SHAPE). Larger R -> smaller gains; the
        #     second Q entry weights the flow/velocity state -> damping. ---
        roll_q=((1.0, 0.0), (0.0, 0.25)),
        roll_r=((2.0,),),
        pitch_q=((1.0, 0.0), (0.0, 0.25)),
        pitch_r=((2.0,),),
        thrust_q=((1.0,),),
        thrust_r=((0.9,),),

        # --- Manual gain trim (the experimental surface). ---
        # prop_scale multiplies the LQR proportional gain k_p.
        # Damping is set ONE of two ways, per axis:
        #   damp_ratio is not None -> k_d = damp_ratio * k_p  (RECOMMENDED).
        #     Guarantees damping is non-zero and same-signed as k_p at every
        #     area_fraction. Needed because the open-loop flow row is noisy and
        #     the LQR k_d it produces passes through zero / flips sign across the
        #     schedule (notably ~0 at af=0.133, the descent dead-zone).
        #   damp_ratio is None -> k_d = damp_scale * (LQR k_d)  (legacy).
        roll_prop_scale: float = 0.5,
        roll_damp_scale: float = 15.0,
        roll_damp_ratio=None,           # roll already stable on LQR damping; leave as-is.
        pitch_prop_scale: float = 0.2,
        pitch_damp_scale: float = 10.0,
        pitch_damp_ratio: float = 10.0,  # fills the af=0.133 damping dead-zone.
        thrust_gain_scale: float = 1.0,

        # --- Command limits [rad] / normalized thrust. ---
        roll_limit: float = 0.035,
        pitch_limit: float = 0.030,
        thrust_min: float = 0.64,
        thrust_max: float = 0.82,

        # --- Command shaping. Slew rates relaxed vs the first run so the
        #     damping command is not itself rate-limited away. ---
        roll_slew_rate_rad_s: float = 0.050,
        pitch_slew_rate_rad_s: float = 0.040,
        thrust_slew_rate_per_s: float = 0.050,
        command_filter_alpha: float = 0.60,

        # --- Visual thrust loop. Positive divergence = target expanding =
        #     approach -> increase thrust. Stays purely visual. ---
        enable_divergence_control: bool = True,
        require_target_for_descent: bool = True,
        max_visual_thrust_delta_from_hover: float = 0.09,
        divergence_integral_gain: float = 0.035,
        divergence_integral_limit: float = 1.2,
        raw_divergence_weight: float = 0.10,

        # Sign convention confirmed by closed-loop tests.
        roll_output_sign: float = -1.0,
        pitch_output_sign: float = -1.0,
    ):
        self._hover_thrust = float(hover_thrust)
        self._yaw_setpoint = float(yaw_setpoint)
        self._divergence_setpoint = float(divergence_setpoint)

        self._roll_limit = abs(float(roll_limit))
        self._pitch_limit = abs(float(pitch_limit))
        self._thrust_min = float(thrust_min)
        self._thrust_max = float(thrust_max)

        self._roll_output_sign = 1.0 if roll_output_sign >= 0.0 else -1.0
        self._pitch_output_sign = 1.0 if pitch_output_sign >= 0.0 else -1.0

        # Damp-ratio mode: don't pre-scale the (unreliable) LQR k_d; k_d is
        # synthesized from k_p in compute(). Otherwise pre-scale k_d by damp_scale.
        self._roll_damp_ratio = None if roll_damp_ratio is None else float(roll_damp_ratio)
        self._pitch_damp_ratio = None if pitch_damp_ratio is None else float(pitch_damp_ratio)
        roll_kd_scale = 1.0 if self._roll_damp_ratio is not None else roll_damp_scale
        pitch_kd_scale = 1.0 if self._pitch_damp_ratio is not None else pitch_damp_scale

        # Scheduled gains: LQR baseline times the manual [prop, damp] scale.
        self._roll_lqr = ScheduledLQR(
            self._schedule(ROLL_STATE_MODELS, roll_q, roll_r),
            gain_scale=[[roll_prop_scale, roll_kd_scale]],
        )
        self._pitch_lqr = ScheduledLQR(
            self._schedule(PITCH_STATE_MODELS, pitch_q, pitch_r),
            gain_scale=[[pitch_prop_scale, pitch_kd_scale]],
        )
        self._thrust_lqr = ScheduledLQR(
            self._schedule(THRUST_DIVERGENCE_MODELS, thrust_q, thrust_r),
            gain_scale=[[thrust_gain_scale]],
        )

        self._roll_slew_rate_rad_s = abs(float(roll_slew_rate_rad_s))
        self._pitch_slew_rate_rad_s = abs(float(pitch_slew_rate_rad_s))
        self._thrust_slew_rate_per_s = abs(float(thrust_slew_rate_per_s))
        self._command_filter_alpha = self._clamp(command_filter_alpha, 0.0, 1.0)

        self._enable_divergence_control = bool(enable_divergence_control)
        self._require_target_for_descent = bool(require_target_for_descent)
        self._max_visual_thrust_delta = abs(float(max_visual_thrust_delta_from_hover))
        self._divergence_integral_gain = float(divergence_integral_gain)
        self._divergence_integral_limit = abs(float(divergence_integral_limit))
        self._raw_divergence_weight = self._clamp(raw_divergence_weight, 0.0, 1.0)
        self._divergence_integral = 0.0

        self._previous_roll_cmd = 0.0
        self._previous_pitch_cmd = 0.0
        self._previous_thrust_cmd = self._hover_thrust
        self._has_previous_command = False

    @property
    def hover_thrust(self) -> float:
        return self._hover_thrust

    @property
    def divergence_integral(self) -> float:
        return self._divergence_integral

    def reset_visual_integrators(self):
        self._divergence_integral = 0.0
        self._previous_roll_cmd = 0.0
        self._previous_pitch_cmd = 0.0
        self._previous_thrust_cmd = self._hover_thrust
        self._has_previous_command = False

    def compute(self, target: TargetEstimate, flow: FlowResult, dt: float) -> AttitudeSetpoint:
        """Desired roll/pitch/yaw/thrust from visual data only."""
        dt = max(1e-3, float(dt))

        roll_cmd = 0.0
        pitch_cmd = 0.0
        visual_thrust_delta = 0.0

        area_fraction = self._safe_area_fraction(target)
        flow_valid = flow is not None and bool(getattr(flow, "valid", False))
        target_found = target is not None and bool(getattr(target, "found", False))

        # --- Lateral axes: u = sign * ( -(k_p*offset + k_d*flow) ). ---
        if target_found:
            flow_x = float(getattr(flow, "mean_flow_x_norm", 0.0)) if flow_valid else 0.0
            flow_y = float(getattr(flow, "mean_flow_y_norm", 0.0)) if flow_valid else 0.0

            roll_u = self._axis_command(
                self._roll_lqr.gain_at(area_fraction), float(target.offset_x), flow_x,
                self._roll_damp_ratio,
            )
            pitch_u = self._axis_command(
                self._pitch_lqr.gain_at(area_fraction), float(target.offset_y), flow_y,
                self._pitch_damp_ratio,
            )

            # Smooth saturation toward the limit (not a hard clip).
            roll_cmd = self._soft_limit(self._roll_output_sign * roll_u, self._roll_limit)
            pitch_cmd = self._soft_limit(self._pitch_output_sign * pitch_u, self._pitch_limit)

        # --- Thrust axis: feedback on divergence error + visual integral. ---
        can_use_divergence = self._enable_divergence_control and flow_valid
        if self._require_target_for_descent:
            can_use_divergence = can_use_divergence and target_found

        if can_use_divergence:
            error = self._divergence_for_control(flow) - self._divergence_setpoint

            self._divergence_integral = self._clamp(
                self._divergence_integral + error * dt,
                -self._divergence_integral_limit,
                self._divergence_integral_limit,
            )

            thrust_state = np.array([[error]])
            lqr_delta = float(-(self._thrust_lqr.gain_at(area_fraction) @ thrust_state)[0, 0])
            integral_delta = self._divergence_integral_gain * self._divergence_integral
            visual_thrust_delta = self._soft_limit(
                lqr_delta + integral_delta, self._max_visual_thrust_delta
            )
        else:
            # No visual measurement: decay (don't hard-reset) so one dropped
            # frame is not a discontinuity, while stale info is forgotten.
            self._divergence_integral *= 0.90

        thrust_cmd = self._clamp(
            self._hover_thrust + visual_thrust_delta, self._thrust_min, self._thrust_max
        )

        roll_cmd, pitch_cmd, thrust_cmd = self._shape_commands(roll_cmd, pitch_cmd, thrust_cmd, dt)

        return AttitudeSetpoint(
            timestamp=getattr(target, "timestamp", 0.0),
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=self._yaw_setpoint,
            thrust=thrust_cmd,
        )

    def _shape_commands(self, roll: float, pitch: float, thrust: float, dt: float):
        """
        filter:  c_f = (1-a) c_prev + a c       (first-order low-pass)
        slew:    c_s = c_prev + clip(c_f - c_prev, +-rate*dt)
        clamp:   to the axis limits.
        """
        if not self._has_previous_command:
            self._previous_roll_cmd = 0.0
            self._previous_pitch_cmd = 0.0
            self._previous_thrust_cmd = self._hover_thrust
            self._has_previous_command = True

        a = self._command_filter_alpha
        roll_f = (1.0 - a) * self._previous_roll_cmd + a * roll
        pitch_f = (1.0 - a) * self._previous_pitch_cmd + a * pitch
        thrust_f = (1.0 - a) * self._previous_thrust_cmd + a * thrust

        roll_s = self._slew_limit(self._previous_roll_cmd, roll_f, self._roll_slew_rate_rad_s * dt)
        pitch_s = self._slew_limit(self._previous_pitch_cmd, pitch_f, self._pitch_slew_rate_rad_s * dt)
        thrust_s = self._slew_limit(self._previous_thrust_cmd, thrust_f, self._thrust_slew_rate_per_s * dt)

        roll_s = self._clamp(roll_s, -self._roll_limit, self._roll_limit)
        pitch_s = self._clamp(pitch_s, -self._pitch_limit, self._pitch_limit)
        thrust_s = self._clamp(thrust_s, self._thrust_min, self._thrust_max)

        self._previous_roll_cmd = roll_s
        self._previous_pitch_cmd = pitch_s
        self._previous_thrust_cmd = thrust_s
        return roll_s, pitch_s, thrust_s

    def _divergence_for_control(self, flow: FlowResult) -> float:
        """Blend filtered and raw divergence: (1-w) d_filt + w d_raw."""
        filtered = self._safe_float(getattr(flow, "divergence", 0.0))
        raw = self._safe_float(getattr(flow, "raw_divergence", filtered), default=filtered)
        w = self._raw_divergence_weight
        return (1.0 - w) * filtered + w * raw

    @staticmethod
    def _schedule(models, q, r):
        """Expand (af, A, B) models into ScheduledLQR (af, A, B, Q, R) tuples."""
        return ((af, A, B, q, r) for af, A, B in models)

    @staticmethod
    def _axis_command(gain: np.ndarray, offset: float, flow: float, damp_ratio) -> float:
        """
        Lateral feedback u = -(k_p*offset + k_d*flow).
        k_p = gain[0,0]. k_d = damp_ratio*k_p if damp_ratio is set (consistent
        sign across the schedule), else the scheduled gain[0,1].
        """
        k_p = float(gain[0, 0])
        k_d = damp_ratio * k_p if damp_ratio is not None else float(gain[0, 1])
        return -(k_p * offset + k_d * flow)

    @staticmethod
    def _safe_area_fraction(target: TargetEstimate) -> float:
        if target is None:
            return 0.066
        try:
            return max(1e-4, float(getattr(target, "area_fraction", 0.066)))
        except (TypeError, ValueError):
            return 0.066

    @staticmethod
    def _soft_limit(value: float, limit: float) -> float:
        """L * tanh(v / L): smooth, bounded by +-L, ~linear near 0."""
        limit = abs(float(limit))
        return 0.0 if limit <= 1e-12 else limit * math.tanh(float(value) / limit)

    @staticmethod
    def _slew_limit(previous: float, desired: float, max_step: float) -> float:
        max_step = abs(float(max_step))
        return previous + max(-max_step, min(max_step, desired - previous))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))