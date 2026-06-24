"""
Closed-loop visual controller built from the latest open-loop calibration.

Project constraint for this version:
    PX4 local position / velocity is NOT used by the control law.
    Vehicle state may be logged by bee_node.py for diagnostics, and MAVSDK/PX4
    may be used for the initial automatic takeoff/handoff, but once the visual
    controller is active the commands below are computed only from:

        target offset_x / offset_y
        normalized optical flow mean_flow_x_norm / mean_flow_y_norm
        optical-flow divergence
        target area_fraction for gain scheduling

Inputs used by this controller:
    roll axis:   x_roll  = [target_offset_x, flow_mean_x_norm_s]
    pitch axis:  x_pitch = [target_offset_y, flow_mean_y_norm_s]
    thrust axis: divergence - divergence_setpoint, plus a visual-only integral
                 of the same divergence error.

This v15 tuning keeps the successful sign convention from v14 but backs away
from the first unstable oscillatory run:

    - roll/pitch still use the calibrated 2-state scheduled LQR models;
    - the pitch loop has substantially higher control cost and a lower command
      limit, because the 20260624_183401 run spent most of its time saturated;
    - commands are passed through a purely internal visual setpoint shaper
      (soft saturation + slew-rate limiting). This uses only previous commanded
      setpoints, not PX4 state, and prevents bang-bang excitation of the slow
      image dynamics.

All schedules are raw area_fraction schedules, not sqrt(area_fraction) scaling,
because the reconstructed data showed the sqrt law was a poor fit.
"""

import math
import numpy as np

try:
    from .lqr import ScheduledLQR
    from .state import AttitudeSetpoint, FlowResult, TargetEstimate
except ImportError:
    from lqr import ScheduledLQR
    from state import AttitudeSetpoint, FlowResult, TargetEstimate


# Raw scheduled models from the reconstructed calibration results.
# Each roll/pitch entry is (area_fraction, A, B) for
# [offset[k+1], flow[k+1]]^T = A [offset[k], flow[k]]^T + B u[k].
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

# Scalar divergence model around hover thrust:
# d[k+1] = a d[k] + b (thrust[k] - HOVER_THRUST).
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
        # 0.0 means visual hover. Increase slowly later, for example
        # 0.02 -> 0.04 -> 0.06, only after the centering loop is stable.
        divergence_setpoint: float = 0.0,
        # v14 got the signs right, but the 20260624_183401 run was clearly
        # underdamped. Keep roll authority slightly higher because x stayed
        # near zero; reduce pitch authority because y spent ~60% of the run at
        # saturation and developed a growing low-frequency oscillation.
        roll_limit: float = 0.035,
        pitch_limit: float = 0.025,
        # Purely visual thrust controller. Leave authority available, but slow
        # the command changes below so divergence noise does not inject a fast
        # thrust chatter.
        thrust_min: float = 0.64,
        thrust_max: float = 0.82,
        require_target_for_descent: bool = True,
        # Sign convention confirmed by closed-loop tests after v14.
        roll_output_sign: float = -1.0,
        pitch_output_sign: float = -1.0,
        # Visual thrust loop. Positive divergence means the target expands in
        # the image, so thrust must increase. This remains purely visual: no
        # vehicle_z or vehicle_vz enters the control law.
        enable_divergence_control: bool = True,
        max_visual_thrust_delta_from_hover: float = 0.09,
        divergence_integral_gain: float = 0.035,
        divergence_integral_limit: float = 1.2,
        raw_divergence_weight: float = 0.10,
        # Conservative roll/pitch LQR costs. Larger R -> smaller attitude.
        # Pitch gets a much larger R than v14: the latest run showed pitch
        # saturation, not lack of sign/authority, was the dominant instability.
        roll_q=((1.0, 0.0), (0.0, 0.20)),
        roll_r=((1.8,),),
        pitch_q=((1.0, 0.0), (0.0, 0.20)),
        pitch_r=((8.0,),),
        # Less aggressive than v14, compensated by the visual integral. The
        # goal is a stable hover test before attempting descent.
        thrust_q=((1.0,),),
        thrust_r=((0.9,),),
        # Purely internal command shaping. This is not feedback from PX4 state;
        # it only limits how abruptly the visual controller may change its own
        # setpoint. It removes bang-bang behavior when the LQR output hits a
        # saturation limit.
        roll_slew_rate_rad_s: float = 0.020,
        pitch_slew_rate_rad_s: float = 0.012,
        thrust_slew_rate_per_s: float = 0.050,
        command_filter_alpha: float = 0.55,
    ):
        self._hover_thrust = float(hover_thrust)
        self._yaw_setpoint = float(yaw_setpoint)
        self._divergence_setpoint = float(divergence_setpoint)

        self._roll_limit = abs(float(roll_limit))
        self._pitch_limit = abs(float(pitch_limit))
        self._thrust_min = float(thrust_min)
        self._thrust_max = float(thrust_max)
        self._require_target_for_descent = bool(require_target_for_descent)

        self._roll_output_sign = 1.0 if roll_output_sign >= 0.0 else -1.0
        self._pitch_output_sign = 1.0 if pitch_output_sign >= 0.0 else -1.0

        self._enable_divergence_control = bool(enable_divergence_control)
        self._max_visual_thrust_delta_from_hover = abs(float(max_visual_thrust_delta_from_hover))
        self._divergence_integral_gain = float(divergence_integral_gain)
        self._divergence_integral_limit = abs(float(divergence_integral_limit))
        self._raw_divergence_weight = self._clamp(raw_divergence_weight, 0.0, 1.0)
        self._divergence_integral = 0.0

        self._roll_lqr = self._build_schedule(ROLL_STATE_MODELS, roll_q, roll_r)
        self._pitch_lqr = self._build_schedule(PITCH_STATE_MODELS, pitch_q, pitch_r)
        self._thrust_lqr = self._build_schedule(THRUST_DIVERGENCE_MODELS, thrust_q, thrust_r)

        self._roll_slew_rate_rad_s = abs(float(roll_slew_rate_rad_s))
        self._pitch_slew_rate_rad_s = abs(float(pitch_slew_rate_rad_s))
        self._thrust_slew_rate_per_s = abs(float(thrust_slew_rate_per_s))
        self._command_filter_alpha = self._clamp(command_filter_alpha, 0.0, 1.0)

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
        """Return desired roll/pitch/yaw/thrust using visual data only."""
        dt = max(1e-3, float(dt))

        roll_cmd = 0.0
        pitch_cmd = 0.0
        visual_thrust_delta = 0.0
        yaw_cmd = self._yaw_setpoint

        area_fraction = self._safe_area_fraction(target)
        flow_valid = flow is not None and bool(getattr(flow, "valid", False))
        target_found = target is not None and bool(getattr(target, "found", False))

        if target_found:
            flow_x = float(getattr(flow, "mean_flow_x_norm", 0.0)) if flow_valid else 0.0
            flow_y = float(getattr(flow, "mean_flow_y_norm", 0.0)) if flow_valid else 0.0

            roll_state = np.array([[float(target.offset_x)], [flow_x]])
            pitch_state = np.array([[float(target.offset_y)], [flow_y]])

            roll_raw = float(-(self._roll_lqr.gain_at(area_fraction) @ roll_state)[0, 0])
            pitch_raw = float(-(self._pitch_lqr.gain_at(area_fraction) @ pitch_state)[0, 0])

            roll_cmd = self._roll_output_sign * roll_raw
            pitch_cmd = self._pitch_output_sign * pitch_raw

            # Soft saturation rather than a hard bang-bang clamp. The command is
            # still bounded by roll_limit/pitch_limit, but approaches the bound
            # smoothly when the target is far from center.
            roll_cmd = self._soft_limit(roll_cmd, self._roll_limit)
            pitch_cmd = self._soft_limit(pitch_cmd, self._pitch_limit)

        can_use_divergence = self._enable_divergence_control and flow_valid
        if self._require_target_for_descent:
            can_use_divergence = can_use_divergence and target_found

        if can_use_divergence:
            divergence = self._divergence_for_control(flow)
            divergence_error = divergence - self._divergence_setpoint

            # Visual-only integral action: positive divergence means the target
            # expands in the image, i.e. approach/descent. It should therefore
            # increase thrust. This closes steady visual bias without using PX4
            # z/vz.
            self._divergence_integral += divergence_error * dt
            self._divergence_integral = self._clamp(
                self._divergence_integral,
                -self._divergence_integral_limit,
                self._divergence_integral_limit,
            )

            thrust_state = np.array([[divergence_error]])
            lqr_delta = float(-(self._thrust_lqr.gain_at(area_fraction) @ thrust_state)[0, 0])
            integral_delta = self._divergence_integral_gain * self._divergence_integral
            visual_thrust_delta = lqr_delta + integral_delta
            visual_thrust_delta = self._soft_limit(
                visual_thrust_delta,
                self._max_visual_thrust_delta_from_hover,
            )
        else:
            # No visual measurement -> no visual integral accumulation. Decay
            # rather than hard-reset to avoid one dropped frame causing a
            # discontinuity, while still forgetting stale visual information.
            self._divergence_integral *= 0.90

        thrust_cmd = self._hover_thrust + visual_thrust_delta
        thrust_cmd = self._clamp(thrust_cmd, self._thrust_min, self._thrust_max)

        roll_cmd, pitch_cmd, thrust_cmd = self._shape_commands(
            roll_cmd, pitch_cmd, thrust_cmd, dt
        )

        return AttitudeSetpoint(
            timestamp=getattr(target, "timestamp", 0.0),
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            thrust=thrust_cmd,
        )

    def _shape_commands(self, roll: float, pitch: float, thrust: float, dt: float):
        if not self._has_previous_command:
            self._previous_roll_cmd = 0.0
            self._previous_pitch_cmd = 0.0
            self._previous_thrust_cmd = self._hover_thrust
            self._has_previous_command = True

        alpha = self._command_filter_alpha

        roll_filtered = (1.0 - alpha) * self._previous_roll_cmd + alpha * roll
        pitch_filtered = (1.0 - alpha) * self._previous_pitch_cmd + alpha * pitch
        thrust_filtered = (1.0 - alpha) * self._previous_thrust_cmd + alpha * thrust

        roll_limited = self._slew_limit(
            self._previous_roll_cmd,
            roll_filtered,
            self._roll_slew_rate_rad_s * dt,
        )
        pitch_limited = self._slew_limit(
            self._previous_pitch_cmd,
            pitch_filtered,
            self._pitch_slew_rate_rad_s * dt,
        )
        thrust_limited = self._slew_limit(
            self._previous_thrust_cmd,
            thrust_filtered,
            self._thrust_slew_rate_per_s * dt,
        )

        roll_limited = self._clamp(roll_limited, -self._roll_limit, self._roll_limit)
        pitch_limited = self._clamp(pitch_limited, -self._pitch_limit, self._pitch_limit)
        thrust_limited = self._clamp(thrust_limited, self._thrust_min, self._thrust_max)

        self._previous_roll_cmd = roll_limited
        self._previous_pitch_cmd = pitch_limited
        self._previous_thrust_cmd = thrust_limited

        return roll_limited, pitch_limited, thrust_limited

    def _divergence_for_control(self, flow: FlowResult) -> float:
        filtered = self._safe_float(getattr(flow, "divergence", 0.0), default=0.0)
        raw = self._safe_float(getattr(flow, "raw_divergence", filtered), default=filtered)
        w = self._raw_divergence_weight
        return (1.0 - w) * filtered + w * raw

    @staticmethod
    def _build_schedule(models, q, r) -> ScheduledLQR:
        return ScheduledLQR((area, A, B, q, r) for area, A, B in models)

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
        limit = abs(float(limit))
        if limit <= 1e-12:
            return 0.0
        return limit * math.tanh(float(value) / limit)

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
