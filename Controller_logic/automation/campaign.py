"""Run the campaign: plan the matrix, fly it, resume where it died.

    python3 campaign.py --campaign campaign_01 --base-seed 20260920
    python3 campaign.py --campaign campaign_01 --base-seed 20260920 --dry-run
    python3 campaign.py --campaign campaign_01 --base-seed 20260920 --limit 2

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
import json
import os
import subprocess
import sys
import time
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


def run_one(spec, campaign_dir: Path, source_world: Path, *,
            timeout_sec: int, bee_dir: Path, px4_dir: Path) -> str:
    run_dir = campaign_dir / "runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    world = generate(source_world, spec, run_dir / "world.sdf")
    (run_dir / "run.json").write_text(
        json.dumps(spec.to_json(), indent=2) + "\n", encoding="utf-8")

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True,
                        help="Campaign directory name, e.g. campaign_01.")
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--bee-dir", type=Path, default=DEFAULT_BEE_DIR)
    parser.add_argument("--px4-dir", type=Path, default=DEFAULT_PX4_DIR)
    parser.add_argument("--source-world", type=Path, default=None)
    parser.add_argument("--repetitions", type=int, default=15)
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

    runs = scenario_mod.plan(args.base_seed, repetitions=args.repetitions)
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
    }, indent=2) + "\n", encoding="utf-8")

    print(f"campaign {args.campaign}: {len(runs)} planned, "
          f"{len(runs) - len(pending)} already done, {len(pending)} to fly")

    if args.limit is not None:
        pending = pending[:args.limit]

    if args.dry_run:
        for spec in pending:
            run_dir = campaign_dir / "runs" / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            generate(source_world, spec, run_dir / "world.sdf")
            (run_dir / "run.json").write_text(
                json.dumps(spec.to_json(), indent=2) + "\n", encoding="utf-8")
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
