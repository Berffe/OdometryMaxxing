"""The contract between ``MissionRoutine`` and its caller.

Everything here is ROS-free and import-light on purpose: ``mission_routine``
can be exercised in a plain unit test, and ``bee_node`` never has to know what
a mission phase means.

Three things used to leak into ``bee_node`` and now live here:

``MissionInputs``
    Was 11 keyword arguments unpacked from ``target`` / ``flow`` at the call
    site.  A phase that wanted one more visual signal forced an edit in
    ``bee_node``.  Now the mission receives the measurement objects and reads
    whatever it needs.

``ControlEffect``
    Was ``if mc.info.get("event") in ("center_done", "final_probe_start",
    "descent_start"): control_law.reset_divergence_integral()`` -- mission
    semantics encoded as string matching in the node.  A phase now *declares*
    the effect it wants; the node just applies whatever it is handed.

``PhaseSpec``
    Was a ``display`` dict inside ``bee_node._announce_mission_substate``.  A
    phase now carries its own human-readable name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from bee_control.core.state import FlowResult, TargetEstimate
# --------------------------------------------------------------------------
# Phase identifiers
#
# Deliberately plain strings, not an Enum: these values are written straight
# into the ``mission_substate`` CSV column and compared by ``analyse_log.py``.
# A str-Enum formats differently across Python 3.10/3.11+, which would silently
# change the logs. The registry below supplies everything an Enum would have.
# --------------------------------------------------------------------------
CENTER = "center"
APPROACH_PROBE = "approach_probe"
FINAL_PROBE = "final_probe"
PROBE_HOLD = "probe_hold"
DESCEND = "descend"
INFEASIBLE = "infeasible"
LANDED = "landed"
#: Terminal, latched by the node via ``MissionRoutine.mark_aborted``. Exists so
#: that ``mission_substate`` never has to be overwritten at write time.
ABORTED = "aborted"


@dataclass(frozen=True)
class TerminalRequest:
    """A phase's request to end the run.

    Phases own the decision and the reason; ``bee_node`` owns what ending a run
    means (sequencer latch, motor stop, process shutdown). The field is declared
    on :class:`MissionControl` for the same reason ``effects`` is: the node
    applies whatever it is handed and never has to recognise a phase name.

    ``outcome`` is the terminal mission substate that caused it -- LANDED,
    ABORTED or INFEASIBLE -- which is also what the campaign outcome record
    reports as the run's status.
    """

    outcome: str
    reason: str = ""


class ControlEffect(Enum):
    """A side effect on ``ControlLaw`` that a phase transition requests.

    The mission decides WHEN; ``bee_node`` only owns the mapping from effect to
    method call. Adding an effect for a new phase means adding a member here
    and one line in the node's effect table -- never an edit to the control
    tick's branching logic.
    """

    RESET_DIVERGENCE_INTEGRAL = "reset_divergence_integral"
    RESET_VISUAL_INTEGRATORS = "reset_visual_integrators"


@dataclass(frozen=True)
class ActuationFeedback:
    """What the controller actually commanded last tick.

    The probes are driven from this, not from any physical measurement: the
    whole feasibility argument rests on the THRUST-COMMAND RESIDUAL, so the
    provenance of these numbers matters.
    """

    last_thrust_cmd: float = 0.0
    last_roll_cmd_rad: float = 0.0
    last_pitch_cmd_rad: float = 0.0
    last_vertical_accel_cmd: Optional[float] = None
    last_roll_accel_cmd: float = 0.0
    last_pitch_accel_cmd: float = 0.0

    @classmethod
    def from_control_law(
        cls,
        control_law,
        last_thrust_cmd: float,
        *,
        last_roll_cmd_rad: float = 0.0,
        last_pitch_cmd_rad: float = 0.0,
    ) -> "ActuationFeedback":
        return cls(
            last_thrust_cmd=float(last_thrust_cmd),
            last_roll_cmd_rad=float(last_roll_cmd_rad),
            last_pitch_cmd_rad=float(last_pitch_cmd_rad),
            last_vertical_accel_cmd=control_law.last_vertical_accel_cmd,
            last_roll_accel_cmd=control_law.last_roll_accel_cmd,
            last_pitch_accel_cmd=control_law.last_pitch_accel_cmd,
        )


@dataclass(frozen=True)
class MissionInputs:
    """One control tick's worth of mission input.

    ``t`` is the source camera timestamp in Gazebo SIM seconds -- the same time
    base as ``FlowResult.timestamp``. ``dt`` is the SIM-time interval between
    consecutive controlled frames, never a wall-clock duration.
    """

    t: float
    dt: float
    target: TargetEstimate
    flow: FlowResult
    actuation: ActuationFeedback = field(default_factory=ActuationFeedback)

    # -- Visual accessors. Phases read these instead of touching the raw
    # -- dataclasses, so a change of measurement plumbing lands in one place.
    @property
    def offset_x(self) -> float:
        return float(getattr(self.target, "offset_x", 0.0))

    @property
    def offset_y(self) -> float:
        return float(getattr(self.target, "offset_y", 0.0))

    @property
    def target_found(self) -> bool:
        return bool(getattr(self.target, "found", False))

    @property
    def area_fraction(self) -> float:
        return float(getattr(self.target, "area_fraction", 0.0))

    @property
    def fov_saturated(self) -> bool:
        return bool(getattr(self.target, "fov_saturated", False))

    @property
    def flow_x_norm_s(self) -> float:
        return float(getattr(self.flow, "mean_flow_x_norm", 0.0))

    @property
    def flow_y_norm_s(self) -> float:
        return float(getattr(self.flow, "mean_flow_y_norm", 0.0))

    @property
    def flow_valid(self) -> bool:
        return bool(getattr(self.flow, "valid", False))

    @property
    def divergence_1_s(self) -> float:
        return float(getattr(self.flow, "divergence", 0.0))


@dataclass
class MissionControl:
    """What the mission asks of the control law for one tick.

    The gain fields are unchanged from the previous revision, so every existing
    construction site and every ``analyse_log`` column still line up. What is
    new is ``effects``: the phase states its control-law side effects instead of
    encoding them as an event string the node has to recognise.
    """

    divergence_setpoint: float = 0.0
    thrust_gain_override: Optional[float] = None
    lateral_p_scale: float = 1.0
    lateral_d_scale: float = 1.0
    roll_p_scale: Optional[float] = None
    roll_d_scale: Optional[float] = None
    pitch_p_scale: Optional[float] = None
    pitch_d_scale: Optional[float] = None
    # Optional lateral trim contract.
    #
    # CENTER / APPROACH   setpoint = geometric tilt offset MINUS the learned
    #                     adaptive-centre bias; no acceleration feedforward.
    #                     The bias is what makes the steady P error generate
    #                     the counter-wind force.
    # FINAL_PROBE         setpoint = geometric tilt offset ONLY, and the static
    #                     acceleration trim takes over the wind force.  Keeping
    #                     the bias here would double-count it: the feedforward
    #                     already supplies that force, so P would additionally
    #                     hold an error to supply it a second time.  A small
    #                     residual P remains for centring -- see config
    #                     ``final_probe_lateral_p_scale``.
    # DESCENT             setpoint = 0 with P off; the committed feedforward
    #                     carries everything, because image offset stops being
    #                     a position measurement about a second before contact.
    roll_offset_setpoint: float = 0.0
    pitch_offset_setpoint: float = 0.0
    roll_accel_feedforward_m_s2: float = 0.0
    pitch_accel_feedforward_m_s2: float = 0.0
    # Reference the large-offset gain blend measures "off centre" FROM.
    #
    # Defaults to the offset setpoint above, which is the previous behaviour.
    # CENTER and APPROACH override it with the geometric tilt offset ALONE,
    # deliberately excluding the learned wind bias: that bias is the steady
    # image error the P loop must hold to reject wind, so counting it as
    # off-centredness attenuates the lateral gain hardest exactly when the wind
    # is strongest. The blend should read how far the vehicle is from being
    # PHYSICALLY over the platform, which is what the geometric offset marks.
    roll_gain_blend_setpoint: Optional[float] = None
    pitch_gain_blend_setpoint: Optional[float] = None
    # False disables the large-offset gain blend ENTIRELY for this tick: both
    # the lateral P and the lateral D branch keep their commanded scales, and
    # the blend reference fields above are not consulted.
    #
    # The blend exists to keep the FAR-FIELD compound P+D request out of the
    # collapsing-gain part of the angle soft limit during a large-offset
    # capture transient. That transient is a CENTER phenomenon. Every later
    # phase either has a small centring error by construction (it passed the
    # CENTER gate to get there) or has no lateral P at all, so from
    # APPROACH_PROBE onward the blend can only subtract lateral authority --
    # and it subtracts most as the residual centring error grows, which is
    # exactly when the wind is strongest and the authority is most needed.
    #
    # Like ``scale_lateral_d_with_offset``, this is the CALLER's decision and
    # not an inference inside ControlLaw. A substate test down there would be
    # the same trap that the ``p_scale > 0`` proxy already sprang once.
    #
    # NOTE the transition ticks: the first APPROACH_PROBE control is built in
    # ``phases/center.py`` and the first FINAL_PROBE control in
    # ``phases/approach_probe.py``, so "CENTER only" is four construction
    # sites, not one file. Phases that leave this at the default and command
    # ``lateral_p_scale=0.0`` (DESCEND, PROBE_HOLD, INFEASIBLE, LANDED,
    # ABORTED) are unaffected: the blend multiplies a zero P and the D branch
    # is already gated on P being active.
    apply_offset_gain_blend: bool = True
    # False keeps the optical-flow D branch at exactly ``lateral_d_scale``,
    # unattenuated by the large-offset blend. FINAL_PROBE needs this because
    # its D branch is the evidence the lateral feasibility gates rest on.
    #
    # Now largely vestigial: it is only consulted when the blend is ON, and
    # after the change above the only phase with the blend on is CENTER, which
    # wants D attenuated. Kept so the log schema and the FINAL_PROBE contract
    # it documents stay stable, and so a future phase that re-enables the
    # blend still gets to exempt its D branch.
    scale_lateral_d_with_offset: bool = True
    enable_integral: bool = True
    substate: str = CENTER
    effects: tuple[ControlEffect, ...] = ()
    # Set by a terminal phase to end the run. None means "keep flying".
    terminal_request: Optional[TerminalRequest] = None
    info: dict = field(default_factory=dict)

    def control_kwargs(self) -> dict:
        """Exactly the keyword arguments ``ControlLaw.compute`` expects.

        Built here rather than in ``bee_node`` so that adding a scheduled gain
        is a change in two files that already know about gains, not three.
        """
        return {
            "divergence_setpoint": self.divergence_setpoint,
            "thrust_gain_override": self.thrust_gain_override,
            "lateral_p_scale": self.lateral_p_scale,
            "lateral_d_scale": self.lateral_d_scale,
            "roll_p_scale": self.roll_p_scale,
            "roll_d_scale": self.roll_d_scale,
            "pitch_p_scale": self.pitch_p_scale,
            "pitch_d_scale": self.pitch_d_scale,
            "roll_offset_setpoint": self.roll_offset_setpoint,
            "pitch_offset_setpoint": self.pitch_offset_setpoint,
            "roll_accel_feedforward_m_s2": self.roll_accel_feedforward_m_s2,
            "pitch_accel_feedforward_m_s2": self.pitch_accel_feedforward_m_s2,
            "roll_gain_blend_setpoint": self.roll_gain_blend_setpoint,
            "pitch_gain_blend_setpoint": self.pitch_gain_blend_setpoint,
            "apply_offset_gain_blend": self.apply_offset_gain_blend,
            "scale_lateral_d_with_offset": self.scale_lateral_d_with_offset,
            "enable_integral": self.enable_integral,
        }

    @property
    def event(self) -> str:
        return str(self.info.get("event", "") or "")

    def summary(self) -> str:
        """One-line human description of the commanded gains.

        Used for the phase-transition log line. Lives with the data rather than
        as an f-string in the node, so a new scheduled gain shows up in the
        console automatically.
        """
        roll_p = self.roll_p_scale if self.roll_p_scale is not None else self.lateral_p_scale
        roll_d = self.roll_d_scale if self.roll_d_scale is not None else self.lateral_d_scale
        pitch_p = self.pitch_p_scale if self.pitch_p_scale is not None else self.lateral_p_scale
        pitch_d = self.pitch_d_scale if self.pitch_d_scale is not None else self.lateral_d_scale
        return (
            f"D*={float(self.divergence_setpoint):+.4f} 1/s, "
            f"K={float(self.thrust_gain_override or 0.0):.4f}, "
            f"RP/RD={float(roll_p):.4f}/{float(roll_d):.4f}, "
            f"PP/PD={float(pitch_p):.4f}/{float(pitch_d):.4f}, "
            f"trim=({float(self.roll_offset_setpoint):+.3f},"
            f"{float(self.pitch_offset_setpoint):+.3f}), "
            f"ff=({float(self.roll_accel_feedforward_m_s2):+.3f},"
            f"{float(self.pitch_accel_feedforward_m_s2):+.3f}) m/s^2, "
            f"integral={int(bool(self.enable_integral))}"
        )


@dataclass(frozen=True)
class PhaseSpec:
    """Registry entry for one mission phase.

    ``handler`` receives ``(routine, inputs)`` and returns a ``MissionControl``.
    Keeping it a small adapter means a phase's numerics stay in one method with
    its own explicit argument list, while dispatch, naming and ordering are
    table-driven.

    Adding a phase is: write ``_do_<name>``, add a ``PhaseSpec`` to
    ``MissionRoutine.PHASES``. Nothing outside ``mission_routine`` changes.
    """

    name: str
    display_name: str
    handler: Callable[[Any, MissionInputs], MissionControl]
    terminal: bool = False
    description: str = ""