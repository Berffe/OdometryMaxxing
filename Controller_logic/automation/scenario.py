"""The campaign's scenario space: what a seed and a cell actually determine.

One module so that the runner and the world generator cannot disagree about
what a run *is*. Every random draw happens here, from one seeded generator, in
a fixed order.

The rule the ablation rests on
------------------------------
**The seed excludes the gate flag.** A gate-on and a gate-off run with the same
seed are the same scenario -- same wind realisation, same platform phase, same
spawn -- differing only in whether the commit decision is allowed to stop the
descent. Without that, the two arms are not paired and the comparison measures
scenario variation instead of the gate.

So the seed is derived from (condition, platform size, repetition) and never
from the gate. `run_id` carries the gate so the directories stay distinct.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field

# Platform radii in metres. The nominal world ships 0.50 m; the campaign spans
# a factor of four around it, which is roughly the range over which the visual
# scale estimate stays usable at the approach height.
PLATFORM_RADII_M = (0.30, 0.40, 0.50, 0.65, 0.80)

CONDITIONS = ("calm", "severe")
GATES = ("on", "off")


@dataclass(frozen=True)
class WindSpec:
    """What the WindController plugin is told.

    The nominal world hard-codes two explicit `<component>` blocks, which means
    its `<seed>` element controls nothing: the gust is identical on every run.
    The campaign uses the plugin's `<synthesis>` path instead, where the seed
    draws amplitudes, frequencies and phases -- so fifteen repetitions of a
    condition are fifteen different gust realisations of the same *statistics*,
    which is what a repetition is supposed to be.
    """

    enabled: bool
    seed: int
    mean_velocity: tuple[float, float, float]
    max_wind_speed: float
    # Per-axis uniform synthesis: (count, amplitude_min, amplitude_max,
    # frequency_min, frequency_max). An empty tuple means a quiet axis.
    axis_x: tuple = ()
    axis_y: tuple = ()
    axis_z: tuple = ()
    # Explicit components written beside the synthesis: (amplitude, frequency,
    # phase). The plugin sums both, so a seeded dominant gust can ride on top
    # of the synthesised background turbulence.
    gust_x: tuple = ()
    gust_y: tuple = ()
    gust_z: tuple = ()


@dataclass(frozen=True)
class PlatformSpec:
    """What the OscillatingPlatformController is told.

    Amplitudes, frequencies and the relative phasing between components are
    FIXED by condition: they are the sea state, and randomising them would make
    every repetition a different sea rather than a different moment in one.

    What the seed draws is one phase OFFSET per axis, added to every component
    on that axis. That rotates the whole axis through its cycle while preserving
    the relative phasing that gives the deck its spectrum. Without it, every run
    starts the platform at the same point and the probe samples a far narrower
    slice of the motion than the fifteen repetitions suggest.
    """

    radius_m: float
    phase_offsets: tuple[float, float, float, float, float]  # x y z roll pitch
    scale: float = 1.0
    surge: tuple = ()
    sway: tuple = ()
    heave: tuple = ()
    roll: tuple = ()
    pitch: tuple = ()


@dataclass(frozen=True)
class Scenario:
    run_id: str
    condition: str
    gate: str
    radius_m: float
    repetition: int
    seed: int
    spawn_pose: tuple[float, float, float, float, float, float]
    wind: WindSpec
    platform: PlatformSpec

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sea states. Amplitude in metres (radians for roll/pitch), frequency in Hz.
# Components are (amplitude, frequency, base_phase).
#
# The base phases are the nominal world's, and they are NOT decoration: they
# are the relative phasing between the three wave components, which is what
# gives the deck its characteristic motion rather than three independent sines.
# The seeded per-axis offset is ADDED to them, rotating the axis through its
# cycle while leaving that relative structure intact.
#
# Both conditions share the same three wave frequencies, as the nominal world
# does -- 0.10 / 0.16 / 0.22 Hz -- because a sea state is an amplitude change,
# not a frequency change. Keeping them equal means the gate-relevant quantity
# that differs between calm and severe is the disturbance magnitude alone.
# ---------------------------------------------------------------------------
_SEA = {
    "calm": {
        "surge": ((0.010, 0.10, 1.40), (0.008, 0.16, 3.00), (0.004, 0.22, 5.40)),
        "sway": ((0.003, 0.10, 1.90), (0.003, 0.16, 3.50), (0.002, 0.22, 5.90)),
        "heave": ((0.040, 0.10, 0.50), (0.100, 0.20, 2.10), (0.020, 0.25, 4.50)),
        "roll": ((0.0017, 0.10, 1.60), (0.0028, 0.16, 3.20), (0.0010, 0.22, 5.60)),
        "pitch": ((0.0035, 0.10, 0.90), (0.0070, 0.16, 2.50), (0.0024, 0.22, 4.90)),
    },
    # The nominal bee_platform.sdf values: this is the sea the world was tuned
    # against, so "severe" is the shipped configuration rather than an invention.
    "severe": {
        "surge": ((0.050, 0.10, 1.40), (0.040, 0.16, 3.00), (0.020, 0.22, 5.40)),
        "sway": ((0.015, 0.10, 1.90), (0.015, 0.16, 3.50), (0.010, 0.22, 5.90)),
        "heave": ((0.200, 0.10, 0.50), (0.500, 0.20, 2.10), (0.100, 0.25, 4.50)),
        "roll": ((0.0087, 0.10, 1.60), (0.0140, 0.16, 3.20), (0.0052, 0.22, 5.60)),
        "pitch": ((0.0175, 0.10, 0.90), (0.0349, 0.16, 2.50), (0.0122, 0.22, 4.90)),
    },
}

_WIND = {
    "calm": {
        "mean": (1.0, 0.0, 0.0),
        "max": 3.0,
        # (count, amp_min, amp_max, freq_min, freq_max)
        "axis_x": (4, 0.05, 0.20, 0.010, 0.060),
        "axis_y": (3, 0.02, 0.10, 0.010, 0.050),
        "axis_z": (),
    },
    "severe": {
        "mean": (6.0, 0.0, 0.0),
        "max": 8.0,
        "axis_x": (5, 0.20, 0.80, 0.015, 0.080),
        "axis_y": (4, 0.10, 0.40, 0.010, 0.060),
        "axis_z": (),
    },
}


def derive_seed(base_seed: int, condition: str, radius_m: float,
                repetition: int) -> int:
    """Seed from the scenario only -- never from the gate flag.

    A hash rather than an arithmetic mix so that adding a platform size or a
    repetition later does not renumber the runs already flown: each cell's seed
    depends only on its own coordinates.
    """
    key = f"{base_seed}|{condition}|{radius_m:.3f}|{repetition}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & 0x7FFFFFFF


def make_run_id(condition: str, radius_m: float, gate: str,
                repetition: int) -> str:
    """Directory name. Carries the gate so paired runs do not collide."""
    return (f"{condition}_r{int(round(radius_m * 100)):03d}"
            f"_gate{gate}_rep{repetition:02d}")


def build(base_seed: int, condition: str, radius_m: float, gate: str,
          repetition: int) -> Scenario:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if gate not in GATES:
        raise ValueError(f"unknown gate {gate!r}")

    seed = derive_seed(base_seed, condition, radius_m, repetition)
    rng = _SplitMix64(seed)

    # Fixed draw order. Adding a draw at the END keeps earlier runs identical;
    # inserting one in the middle silently renumbers every scenario after it.
    phase_offsets = tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(5))

    # The spawn is jittered because a fixed spawn makes the CENTER phase solve
    # the same problem every time, and CENTER convergence is one of the things
    # the campaign reports. The jitter is small enough that the platform stays
    # in frame at the spawn height.
    spawn = (
        1.20 + rng.uniform(-0.25, 0.25),
        1.20 + rng.uniform(-0.25, 0.25),
        0.40,
        0.0, 0.0,
        rng.uniform(-0.35, 0.35),
    )

    sea = _SEA[condition]
    wind = _WIND[condition]

    return Scenario(
        run_id=make_run_id(condition, radius_m, gate, repetition),
        condition=condition,
        gate=gate,
        radius_m=radius_m,
        repetition=repetition,
        seed=seed,
        spawn_pose=spawn,
        wind=WindSpec(
            enabled=True,
            # The plugin does its own seeded synthesis; give it a value derived
            # from -- but not equal to -- the run seed, so a future change to
            # the plugin's draw order cannot alias with the draws made above.
            seed=(seed * 2654435761) & 0x7FFFFFFF,
            mean_velocity=wind["mean"],
            max_wind_speed=wind["max"],
            axis_x=wind["axis_x"],
            axis_y=wind["axis_y"],
            axis_z=wind["axis_z"],
        ),
        platform=PlatformSpec(
            radius_m=radius_m,
            phase_offsets=phase_offsets,
            surge=sea["surge"],
            sway=sea["sway"],
            heave=sea["heave"],
            roll=sea["roll"],
            pitch=sea["pitch"],
        ),
    )


def plan(base_seed: int, *, conditions=CONDITIONS, radii=PLATFORM_RADII_M,
         repetitions: int = 15, gates=("on",),
         severe_ablation: bool = True) -> list[Scenario]:
    """The campaign matrix, in flight order.

    Gate-off runs are generated for the SEVERE condition only, reusing the
    severe seeds exactly -- that pairing is the ablation. Calm gate-off runs
    exist in the paper's plan too, but only as a re-run of whichever calm runs
    actually refused, which cannot be known before the calm runs have flown.
    """
    runs: list[Scenario] = []
    for condition in conditions:
        for radius in radii:
            for rep in range(repetitions):
                for gate in gates:
                    runs.append(build(base_seed, condition, radius, gate, rep))
    if severe_ablation and "severe" in conditions:
        for radius in radii:
            for rep in range(repetitions):
                runs.append(build(base_seed, "severe", radius, "off", rep))
    return runs


class _SplitMix64:
    """A tiny, fully specified PRNG.

    Deliberately not `random.Random`: the campaign has to be reproducible from
    the seed alone across Python versions, and the standard library makes no
    promise that its stream is stable. Twenty lines is cheaper than that risk.
    """

    def __init__(self, seed: int):
        self._state = int(seed) & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * ((self._next() >> 11) * (1.0 / 9007199254740992.0))
