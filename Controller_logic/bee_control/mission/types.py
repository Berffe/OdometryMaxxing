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
from typing import Any, Callable, Mapping, Optional

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

# Retained for backwards compatibility with older imports.
PROBE = "probe"


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
    last_vertical_accel_cmd: Optional[float] = None
    last_roll_accel_cmd: float = 0.0
    last_pitch_accel_cmd: float = 0.0

    @classmethod
    def from_control_law(cls, control_law, last_thrust_cmd: float) -> "ActuationFeedback":
        return cls(
            last_thrust_cmd=float(last_thrust_cmd),
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
    enable_integral: bool = True
    substate: str = CENTER
    effects: tuple[ControlEffect, ...] = ()
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


def phase_display_name(phases: Mapping[str, PhaseSpec], substate: str) -> str:
    spec = phases.get(substate)
    return spec.display_name if spec is not None else str(substate).upper()
