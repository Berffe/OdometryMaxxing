"""Run the campaign: plan the matrix, fly it, resume where it died.

    python3 campaign.py --campaign campaign_02 --base-seed 20260921 --dry-run
    python3 campaign.py --campaign campaign_02 --base-seed 20260921 --limit 2
    python3 campaign.py --campaign campaign_02 --base-seed 20260921

The matrix
----------
6 wind cases x 6 platform cases x 5 deck radii x 2 gate settings = 360 runs,
one repetition per cell. Wind and platform each have three weak and three
strong cases. The platform moves in translation only (surge, sway, heave).
Case values and their justification live in README_automation.md,
section "Campaign matrix".

Resumability
------------
A run is "done" when its directory holds an outcome record. Re-running the same
command skips those and continues, so a campaign that dies at 3 am costs the
runs it had not reached and nothing more.

The controller writes only the first half of that record -- `postprocessed` is
false, and the truth-derived fields are filled in by a separate offline pass.
A record with `postprocessed: false` still counts as done. Waiting for the
offline pass instead would mean a campaign interrupted between the two passes
re-flies everything it had already flown.

Two failure statuses belong to the harness rather than the controller, because
in both cases the node never reaches its own `close()`:

    launch_failed   a launch stage never became ready
    timeout         the run exceeded its wall-clock budget

The harness writes those records itself, so a failed run still leaves evidence
and is not silently re-attempted forever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario as scenario_mod  # noqa: E402
from make_world import generate  # noqa: E402

AUTOMATION_DIR = Path(__file__).resolve().parent
DEFAULT_PX4_DIR = Path(os.environ.get("PX4_DIR", Path.home() / "PX4-Autopilot"))
DEFAULT_BEE_DIR = DEFAULT_PX4_DIR / "BEE_LAND"
DEFAULT_LOGS_DIR = DEFAULT_BEE_DIR / "controller" / "logs"

EXIT_LAUNCH_FAILED = 2
EXIT_TIMEOUT = 3


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindCase:
    """One wind condition.

    turbulence_*  plugin synthesis: (count, amp_min, amp_max, f_min, f_max),
                  amplitudes in m/s, frequencies in Hz.
    gust_*        one dominant gust per run: (amp_min, amp_max, f_min, f_max).
                  Its amplitude, frequency and phase are drawn from the run
                  seed and written as an explicit plugin component.
    """

    name: str
    level: str
    mean_velocity: tuple[float, float, float]
    turbulence_x: tuple = ()
    turbulence_y: tuple = ()
    gust_x: tuple = ()
    gust_y: tuple = ()


@dataclass(frozen=True)
class PlatformCase:
    """One sea state, translation only.

    Components are (amplitude_m, frequency_hz, base_phase_rad). The base
    phases are the nominal world's; the run seed adds one offset per axis.
    """

    name: str
    level: str
    surge: tuple
    sway: tuple
    heave: tuple


_SURGE_PHASES = (1.40, 3.00, 5.40)
_SWAY_PHASES = (1.90, 3.50, 5.90)
_HEAVE_PHASES = (0.50, 2.10, 4.50)


def _axis(amplitudes, frequencies, phases) -> tuple:
    return tuple(zip(amplitudes, frequencies, phases))


def _sea(name, level, *, heave_f, heave_a, lateral_f, surge_a, sway_a) -> PlatformCase:
    return PlatformCase(
        name=name,
        level=level,
        surge=_axis(surge_a, lateral_f, _SURGE_PHASES),
        sway=_axis(sway_a, lateral_f, _SWAY_PHASES),
        heave=_axis(heave_a, heave_f, _HEAVE_PHASES),
    )


_NOMINAL_F = (0.10, 0.16, 0.22)
_NOMINAL_SURGE = (0.050, 0.040, 0.020)
_NOMINAL_SWAY = (0.015, 0.015, 0.010)

PLATFORM_CASES = (
    # Weak: platform jerk kept below 40 % of the chi_z budget (README).
    _sea("pw1", "weak", heave_f=(0.06, 0.10, 0.14), heave_a=(0.08, 0.20, 0.03),
         lateral_f=(0.06, 0.10, 0.14),
         surge_a=(0.030, 0.025, 0.010), sway_a=(0.010, 0.010, 0.005)),
    _sea("pw2", "weak", heave_f=(0.075, 0.12, 0.165), heave_a=(0.10, 0.28, 0.05),
         lateral_f=(0.075, 0.12, 0.165),
         surge_a=(0.040, 0.030, 0.015), sway_a=(0.012, 0.012, 0.008)),
    _sea("pw3", "weak", heave_f=_NOMINAL_F, heave_a=(0.10, 0.18, 0.06),
         lateral_f=_NOMINAL_F,
         surge_a=_NOMINAL_SURGE, sway_a=_NOMINAL_SWAY),
    # Strong: heavy swell at the authority boundary (rho_z ~ 1, chi clean).
    _sea("ps1", "strong", heave_f=(0.08, 0.13, 0.18), heave_a=(0.25, 0.85, 0.10),
         lateral_f=(0.08, 0.13, 0.18),
         surge_a=_NOMINAL_SURGE, sway_a=_NOMINAL_SWAY),
    # Strong: short chop, bandwidth-limited (rho_z < 1, chi_z ~ 1.5 x limit).
    _sea("ps2", "strong", heave_f=(0.23, 0.36, 0.50), heave_a=(0.02, 0.07, 0.02),
         lateral_f=(0.23, 0.36, 0.50),
         surge_a=(0.004, 0.004, 0.002), sway_a=(0.002, 0.002, 0.001)),
    # Strong: the nominal bee_platform.sdf sea, translation only (both limits).
    _sea("ps3", "strong", heave_f=(0.10, 0.20, 0.25), heave_a=(0.20, 0.50, 0.10),
         lateral_f=_NOMINAL_F,
         surge_a=_NOMINAL_SURGE, sway_a=_NOMINAL_SWAY),
)

_GUST_BAND_HZ = (0.03, 0.12)

WIND_CASES = (
    WindCase("ww1", "weak", (2.0, 0.0, 0.0),
             turbulence_x=(3, 0.10, 0.30, *_GUST_BAND_HZ),
             turbulence_y=(2, 0.05, 0.15, *_GUST_BAND_HZ)),
    WindCase("ww2", "weak", (4.0, 0.0, 0.0),
             turbulence_x=(3, 0.20, 0.50, *_GUST_BAND_HZ),
             turbulence_y=(2, 0.10, 0.25, *_GUST_BAND_HZ)),
    WindCase("ww3", "weak", (6.0, 0.0, 0.0),
             turbulence_x=(3, 0.40, 0.80, *_GUST_BAND_HZ),
             turbulence_y=(2, 0.15, 0.35, *_GUST_BAND_HZ)),
    # Same mean and background as ww3: only the dominant gust differs.
    WindCase("ws1", "strong", (6.0, 0.0, 0.0),
             turbulence_x=(3, 0.40, 0.80, *_GUST_BAND_HZ),
             turbulence_y=(2, 0.15, 0.35, *_GUST_BAND_HZ),
             gust_x=(4.2, 5.0, 0.06, 0.10)),
    # Low mean, strongest gust: refusal must follow the gust, not the mean.
    WindCase("ws2", "strong", (3.0, 0.0, 0.0),
             turbulence_x=(3, 0.60, 1.00, *_GUST_BAND_HZ),
             turbulence_y=(2, 0.20, 0.40, *_GUST_BAND_HZ),
             gust_x=(5.0, 5.8, 0.06, 0.10)),
    # Oblique: gusts on both lateral axes, so both lateral gates are exercised.
    WindCase("ws3", "strong", (3.5, 3.5, 0.0),
             turbulence_x=(3, 0.40, 0.80, *_GUST_BAND_HZ),
             turbulence_y=(3, 0.40, 0.80, *_GUST_BAND_HZ),
             gust_x=(3.6, 4.4, 0.06, 0.10),
             gust_y=(3.6, 4.4, 0.06, 0.10)),
)

# Nominal deck first: the first 72 runs are already a complete 6 x 6 x 2 block.
PLATFORM_RADII_M = (0.50, 1.50)
GATES = ("on", "off")

# Spawn geometry. The drone climbs vertically from its spawn point, so it must
# start outside the deck footprint: radius + lateral motion + drone half-span.
SPAWN_MIN_DISTANCE_M = 1.70
SPAWN_DECK_CLEARANCE_M = 0.85
SPAWN_JITTER_M = 0.15
SPAWN_YAW_JITTER_RAD = 0.35
SPAWN_HEIGHT_M = 0.40


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def derive_seed(base_seed: int, wind: WindCase, platform: PlatformCase,
                radius_m: float, repetition: int) -> int:
    """Seed from the scenario coordinates only -- never from the gate flag."""
    key = (f"{base_seed}|{wind.name}|{platform.name}|{radius_m:.3f}|"
           f"{repetition}").encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & 0x7FFFFFFF


def _spawn_pose(radius_m: float, rng) -> tuple:
    distance = max(SPAWN_MIN_DISTANCE_M, radius_m + SPAWN_DECK_CLEARANCE_M)
    diagonal = distance / math.sqrt(2.0)
    return (
        diagonal + rng.uniform(-SPAWN_JITTER_M, SPAWN_JITTER_M),
        diagonal + rng.uniform(-SPAWN_JITTER_M, SPAWN_JITTER_M),
        SPAWN_HEIGHT_M,
        0.0, 0.0,
        rng.uniform(-SPAWN_YAW_JITTER_RAD, SPAWN_YAW_JITTER_RAD),
    )


def _draw_gust(spec: tuple, rng) -> tuple:
    """Always consumes three draws, so every case keeps the same draw order."""
    amplitude_u = rng.uniform(0.0, 1.0)
    frequency_u = rng.uniform(0.0, 1.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    if not spec:
        return ()
    amp_lo, amp_hi, f_lo, f_hi = spec
    return ((amp_lo + (amp_hi - amp_lo) * amplitude_u,
             f_lo + (f_hi - f_lo) * frequency_u,
             phase),)


def _speed_bound(mean: float, turbulence: tuple, gust: tuple) -> float:
    bound = abs(mean)
    if turbulence:
        count, _, amp_hi, _, _ = turbulence
        bound += count * amp_hi
    bound += sum(abs(amplitude) for amplitude, _, _ in gust)
    return bound


def build(base_seed: int, wind: WindCase, platform: PlatformCase,
          radius_m: float, gate: str, repetition: int) -> scenario_mod.Scenario:
    seed = derive_seed(base_seed, wind, platform, radius_m, repetition)
    rng = scenario_mod._SplitMix64(seed)

    # Fixed draw order: phase offsets, spawn, gust x, gust y.
    phase_offsets = tuple(rng.uniform(0.0, 2.0 * math.pi) for _ in range(5))
    spawn = _spawn_pose(radius_m, rng)
    gust_x = _draw_gust(wind.gust_x, rng)
    gust_y = _draw_gust(wind.gust_y, rng)

    bound_x = _speed_bound(wind.mean_velocity[0], wind.turbulence_x, gust_x)
    bound_y = _speed_bound(wind.mean_velocity[1], wind.turbulence_y, gust_y)
    max_wind_speed = math.ceil(math.hypot(bound_x, bound_y) * 10.0) / 10.0

    condition = f"{wind.name}-{platform.name}"
    return scenario_mod.Scenario(
        run_id=scenario_mod.make_run_id(condition, radius_m, gate, repetition),
        condition=condition,
        gate=gate,
        radius_m=radius_m,
        repetition=repetition,
        seed=seed,
        spawn_pose=spawn,
        wind=scenario_mod.WindSpec(
            enabled=True,
            seed=(seed * 2654435761) & 0x7FFFFFFF,
            mean_velocity=wind.mean_velocity,
            max_wind_speed=max_wind_speed,
            axis_x=wind.turbulence_x,
            axis_y=wind.turbulence_y,
            axis_z=(),
            gust_x=gust_x,
            gust_y=gust_y,
        ),
        platform=scenario_mod.PlatformSpec(
            radius_m=radius_m,
            phase_offsets=phase_offsets,
            surge=platform.surge,
            sway=platform.sway,
            heave=platform.heave,
            roll=(),
            pitch=(),
        ),
    )


def plan(base_seed: int, *, repetitions: int = 1) -> list[scenario_mod.Scenario]:
    """The campaign matrix, in flight order.

    Gate-on and gate-off twins are flown back to back, so slow drift of the
    machine (measured_T under load) affects both arms of a pair equally.
    """
    return [
        build(base_seed, wind, platform, radius, gate, rep)
        for rep in range(repetitions)
        for radius in PLATFORM_RADII_M
        for platform in PLATFORM_CASES
        for wind in WIND_CASES
        for gate in GATES
    ]


def matrix_description() -> dict:
    return {
        "wind_cases": [asdict(case) for case in WIND_CASES],
        "platform_cases": [asdict(case) for case in PLATFORM_CASES],
        "radii_m": list(PLATFORM_RADII_M),
        "gates": list(GATES),
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def outcome_path(run_dir: Path) -> Path | None:
    """The controller names its files by timestamp, so this is a glob."""
    matches = sorted(run_dir.glob("bee_outcome_*.json"))
    return matches[0] if matches else None


def is_done(run_dir: Path) -> bool:
    path = outcome_path(run_dir)
    if path is None:
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A record that will not parse is worse than none: it would be skipped
        # forever while carrying nothing. Treat the run as not done.
        return False
    return True


def write_harness_outcome(run_dir: Path, spec, status: str, reason: str) -> None:
    """Record a run the node never got to finish.

    Without this a launch failure leaves an empty directory, which is
    indistinguishable from a run that was never attempted -- so the campaign
    would retry it on every resume and never make progress.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "outcome_schema_version": "1.0-outcome",
        "run_id": spec.run_id,
        "postprocessed": False,
        "written_by": "campaign.py",
        "seed": spec.seed,
        "status": status,
        "reason": reason,
        "abort_phase": None,
        "verdict_reached": False,
        "refusal_criteria": None,
        "refusal_axes": None,
        "commit_gate_enabled": spec.gate == "on",
    }
    target = run_dir / f"bee_outcome_{stamp}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _prepare_run_dir(spec, campaign_dir: Path, source_world: Path) -> tuple[Path, Path]:
    run_dir = campaign_dir / "runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    world = generate(source_world, spec, run_dir / "world.sdf")
    (run_dir / "run.json").write_text(
        json.dumps(spec.to_json(), indent=2) + "\n", encoding="utf-8")
    return run_dir, world


def run_one(spec, campaign_dir: Path, source_world: Path, *,
            timeout_sec: int, bee_dir: Path, px4_dir: Path) -> str:
    run_dir, world = _prepare_run_dir(spec, campaign_dir, source_world)

    env = dict(os.environ)
    env.update({
        "PX4_DIR": str(px4_dir),
        "BEE_DIR": str(bee_dir),
        "SPAWN_POSE": ",".join(f"{v:.4f}" for v in spec.spawn_pose),
        "RUN_TIMEOUT": str(timeout_sec),
    })

    completed = subprocess.run(
        [str(AUTOMATION_DIR / "run_once.sh"),
         "--run-dir", str(run_dir),
         "--world", str(world),
         "--gate", spec.gate,
         "--seed", str(spec.seed),
         "--timeout", str(timeout_sec)],
        env=env, check=False)

    if completed.returncode == EXIT_LAUNCH_FAILED:
        write_harness_outcome(run_dir, spec, "launch_failed",
                              "a launch stage never became ready")
        return "launch_failed"
    if completed.returncode == EXIT_TIMEOUT:
        write_harness_outcome(run_dir, spec, "timeout",
                              f"exceeded {timeout_sec}s of wall clock")
        return "timeout"

    path = outcome_path(run_dir)
    if path is None:
        # The node exited without writing, which close()'s finally block should
        # make impossible. Record it rather than leaving a hole in the matrix.
        write_harness_outcome(run_dir, spec, "crashed",
                              "node exited without writing an outcome record")
        return "crashed"
    return json.loads(path.read_text(encoding="utf-8")).get("status", "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True,
                        help="Campaign directory name, e.g. campaign_02.")
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--bee-dir", type=Path, default=DEFAULT_BEE_DIR)
    parser.add_argument("--px4-dir", type=Path, default=DEFAULT_PX4_DIR)
    parser.add_argument("--source-world", type=Path, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Wall-clock budget per run, seconds.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Fly at most N runs, then stop. For validation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and generate worlds; fly nothing.")
    args = parser.parse_args()

    source_world = args.source_world or (args.bee_dir / "worlds" / "bee_platform.sdf")
    if not source_world.is_file():
        print(f"campaign: source world not found: {source_world}", file=sys.stderr)
        return 1

    campaign_dir = args.logs_dir / args.campaign
    campaign_dir.mkdir(parents=True, exist_ok=True)

    runs = plan(args.base_seed, repetitions=args.repetitions)
    pending = [r for r in runs
               if not is_done(campaign_dir / "runs" / r.run_id)]

    (campaign_dir / "campaign.json").write_text(json.dumps({
        "campaign": args.campaign,
        "base_seed": args.base_seed,
        "repetitions": args.repetitions,
        "planned_runs": len(runs),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_world": str(source_world),
        "timeout_sec": args.timeout,
        "matrix": matrix_description(),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"campaign {args.campaign}: {len(runs)} planned, "
          f"{len(runs) - len(pending)} already done, {len(pending)} to fly")

    if args.limit is not None:
        pending = pending[:args.limit]

    if args.dry_run:
        for spec in pending:
            _prepare_run_dir(spec, campaign_dir, source_world)
        print(f"dry run: generated {len(pending)} worlds, flew none")
        return 0

    tally: dict[str, int] = {}
    started = time.time()

    for index, spec in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] {spec.run_id} (seed {spec.seed})",
              flush=True)
        status = run_one(spec, campaign_dir, source_world,
                         timeout_sec=args.timeout,
                         bee_dir=args.bee_dir, px4_dir=args.px4_dir)
        tally[status] = tally.get(status, 0) + 1
        print(f"    -> {status}", flush=True)

        # Between runs, not just at the end: an orphan contaminates every run
        # after it, so it is worth stopping the campaign to fix rather than
        # collecting 80 more runs that share a stale Gazebo.
        sweep = subprocess.run([str(AUTOMATION_DIR / "teardown.sh")],
                               check=False, capture_output=True, text=True)
        if sweep.returncode != 0:
            print("\ncampaign: STOPPING -- the machine did not come back clean.",
                  file=sys.stderr)
            print(sweep.stderr, file=sys.stderr)
            return 1

    elapsed = time.time() - started
    print(f"\nflew {len(pending)} runs in {elapsed / 60:.1f} min")
    for status, count in sorted(tally.items()):
        print(f"  {status:>14}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
