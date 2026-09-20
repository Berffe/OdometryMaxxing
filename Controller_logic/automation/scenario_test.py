"""Four runs that answer "is the system behaving?" before an overnight commits.

    python3 scenario_test.py --dry-run     # generate and check worlds, fly nothing
    python3 scenario_test.py               # fly all four (~8 min)

The matrix is the campaign's in miniature: two conditions x the gate flag.

    calm   gate on      calm   gate off
    severe gate on      severe gate off

It deliberately reuses `campaign.run_one` rather than reimplementing the launch.
A smoke test that exercises a different code path than the thing it is testing
is worth very little, so the only difference between this and the overnight is
which scenarios are in the list.

Harness checks versus observations
----------------------------------
The two are reported separately and only the first can fail the run.

A **harness check** is something that must hold no matter what the physics did:
the paired worlds are identical, every run left a parseable record, `--gate off`
actually reached the controller, the machine came back clean. A failure here
means the automation is broken and the overnight would produce garbage.

An **observation** is a property of the flight: whether the severe run refused,
whether its gate-off twin descended, what rho was. These are single samples of
a stochastic process -- the Gazebo README is explicit that peak acceleration
varies several-fold between identical configurations -- so a surprising value
is information, not a failure. They are printed for a human to read.

The distinction matters because a test that fails on legitimate physics
variation gets ignored within two days.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign  # noqa: E402
import scenario as scenario_mod  # noqa: E402
from make_world import generate  # noqa: E402

AUTOMATION_DIR = Path(__file__).resolve().parent

# The nominal deck radius, so the test flies the geometry you have been flying
# by hand and a surprise is attributable to the automation rather than to a
# platform size never tried before.
TEST_RADIUS_M = 0.50
TEST_REPETITION = 0

# Smoke-test motion switch. True keeps all translational excitation (surge,
# sway, heave) but removes platform roll and pitch from every generated world.
# Set False to recover the normal campaign motion without touching scenario.py.
TRANSLATION_ONLY = True


def build_matrix(base_seed: int) -> list:
    specs = [
        scenario_mod.build(base_seed, condition, TEST_RADIUS_M, gate,
                           TEST_REPETITION)
        for condition in ("calm", "severe")
        for gate in ("on", "off")
    ]

    if TRANSLATION_ONLY:
        specs = [
            replace(
                spec,
                platform=replace(
                    spec.platform,
                    roll=(),
                    pitch=(),
                ),
            )
            for spec in specs
        ]

    return specs


class Report:
    """Accumulates results so the summary is written once, at the end.

    A smoke test whose verdict is scattered through 200 lines of launch output
    will be read as "it printed a lot" and nothing more.
    """

    def __init__(self) -> None:
        self.checks: list[tuple[bool, str, str]] = []
        self.observations: list[str] = []

    def check(self, passed: bool, name: str, detail: str = "") -> bool:
        self.checks.append((bool(passed), name, detail))
        return bool(passed)

    def observe(self, line: str) -> None:
        self.observations.append(line)

    @property
    def failed(self) -> list[tuple[bool, str, str]]:
        return [c for c in self.checks if not c[0]]

    def render(self) -> str:
        out = ["", "=" * 68, "HARNESS CHECKS", "=" * 68]
        for passed, name, detail in self.checks:
            mark = "PASS" if passed else "FAIL"
            out.append(f"  [{mark}] {name}")
            if detail:
                out.append(f"         {detail}")
        out += ["", "=" * 68, "OBSERVATIONS (physics, not pass/fail)", "=" * 68]
        out += [f"  {line}" for line in self.observations] or ["  (none)"]
        out += ["", "=" * 68]
        if self.failed:
            out.append(f"RESULT: {len(self.failed)} harness check(s) FAILED. "
                       "Do not start the overnight.")
        else:
            out.append("RESULT: harness checks passed. Read the observations "
                       "before committing to the overnight.")
        out.append("=" * 68)
        return "\n".join(out)


def check_worlds(specs, test_dir: Path, source_world: Path,
                 report: Report) -> dict:
    """Generate every world and verify the pairing invariant.

    The gate flag must not reach the world file. If it does, a gate-on and a
    gate-off run are different scenarios and the ablation compares two
    different seas instead of two decisions -- which would invalidate the most
    important result in the paper while looking completely normal in the logs.
    """
    worlds = {}
    for spec in specs:
        run_dir = test_dir / "runs" / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        worlds[spec.run_id] = generate(source_world, spec, run_dir / "world.sdf")
        (run_dir / "run.json").write_text(
            json.dumps(spec.to_json(), indent=2) + "\n", encoding="utf-8")

    by_condition = {}
    for spec in specs:
        by_condition.setdefault(spec.condition, {})[spec.gate] = spec

    for condition, pair in by_condition.items():
        on_spec, off_spec = pair["on"], pair["off"]
        identical = filecmp.cmp(worlds[on_spec.run_id], worlds[off_spec.run_id],
                                shallow=False)
        report.check(
            identical,
            f"{condition}: gate-on and gate-off worlds are identical",
            "" if identical else
            "the gate flag leaked into the world; the ablation would be unpaired")
        report.check(
            on_spec.seed == off_spec.seed,
            f"{condition}: paired runs share a seed",
            f"on={on_spec.seed} off={off_spec.seed}")

    calm = by_condition["calm"]["on"]
    severe = by_condition["severe"]["on"]
    report.check(
        calm.seed != severe.seed,
        "calm and severe are different scenarios",
        f"both drew seed {calm.seed}" if calm.seed == severe.seed else "")

    # The two conditions must actually differ in the world, or "severe" is a
    # label rather than a sea state.
    differ = not filecmp.cmp(worlds[calm.run_id], worlds[severe.run_id],
                             shallow=False)
    report.check(differ, "calm and severe worlds differ")
    return worlds


def _reject_non_finite(token: str):
    """Make json.loads as strict as a non-Python consumer would be.

    Python accepts bare `Infinity` and `NaN` by default, so a plain json.loads
    would happily read a record that every strict JSON parser rejects -- which
    is the exact defect the writer's _json_safe/allow_nan handling exists to
    prevent. Checking with the permissive default would mean this test passes
    whether or not that handling works.
    """
    raise ValueError(f"non-finite literal {token!r} is not valid JSON")


def inspect_outcome(run_dir: Path, spec, report: Report) -> dict | None:
    path = campaign.outcome_path(run_dir)
    if path is None:
        report.check(False, f"{spec.run_id}: wrote an outcome record",
                     "no bee_outcome_*.json; close()'s finally block did not run")
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"),
                            parse_constant=_reject_non_finite)
    except (json.JSONDecodeError, ValueError) as exc:
        report.check(False, f"{spec.run_id}: outcome record is strict JSON",
                     f"{exc}; rho is legitimately inf and must be written null")
        return None

    report.check(True, f"{spec.run_id}: wrote a parseable outcome record")

    # --gate has to survive the whole chain: campaign -> run_once.sh -> argparse
    # -> BeeConfig. If it silently does not, every "gate off" run is a gate-on
    # run and the ablation is 75 duplicates.
    expected = (spec.gate == "on")
    actual = record.get("commit_gate_enabled")
    report.check(
        actual == expected,
        f"{spec.run_id}: --gate {spec.gate} reached the controller",
        f"record says commit_gate_enabled={actual!r}, expected {expected!r}")

    report.check(
        record.get("seed") == spec.seed,
        f"{spec.run_id}: record carries the scenario seed",
        f"record={record.get('seed')!r} expected={spec.seed}")

    known = {"landed", "infeasible", "aborted", "crashed",
             "launch_failed", "timeout"}
    status = record.get("status")
    report.check(status in known, f"{spec.run_id}: status is in the vocabulary",
                 f"got {status!r}")

    # The three CSVs must exist beside the record, or the run produced a verdict
    # with no evidence behind it.
    for stem in ("bee_controller_", "bee_truth_", "bee_wind_"):
        found = list(run_dir.glob(f"{stem}*.csv"))
        report.check(bool(found), f"{spec.run_id}: {stem}*.csv written")

    return record


def observe(record: dict, spec, report: Report) -> None:
    status = record.get("status")
    rho = record.get("vertical_rho")
    rho_text = "n/a" if rho is None else f"{rho:.2f}"
    report.observe(
        f"{spec.run_id:<34} status={status:<12} "
        f"verdict_reached={record.get('verdict_reached')!s:<5} "
        f"rho_v={rho_text}")

    if record.get("verdict_reached"):
        criteria = record.get("refusal_criteria") or "none"
        report.observe(f"{'':<34} criteria={criteria} "
                       f"axes={record.get('refusal_axes') or 'none'} "
                       f"h_cross={record.get('predicted_crossing_height_m')}")
    if status == "aborted":
        report.observe(f"{'':<34} abort_phase={record.get('abort_phase')} "
                       f"reason={record.get('reason')}")


def cross_check(records: dict, specs, report: Report) -> None:
    """The things only visible across runs. Observations, not checks."""
    by_key = {(s.condition, s.gate): records.get(s.run_id) for s in specs}

    for condition in ("calm", "severe"):
        on = by_key.get((condition, "on"))
        off = by_key.get((condition, "off"))
        if not on or not off:
            continue
        if on.get("status") == "infeasible" and off.get("status") != "infeasible":
            report.observe(
                f"{condition}: gate-on refused, gate-off did not -- the ablation "
                f"is doing what it exists to do (off ended '{off.get('status')}')")
        elif on.get("status") == "infeasible" and off.get("status") == "infeasible":
            report.observe(
                f"{condition}: BOTH arms refused. With --gate off the commit "
                "decision should not stop a descent; if this repeats, check "
                "that enable_commit_gate is reaching FINAL_PROBE.")
        else:
            report.observe(
                f"{condition}: gate-on ended '{on.get('status')}', "
                f"gate-off ended '{off.get('status')}'")

    calm_on = by_key.get(("calm", "on"))
    severe_on = by_key.get(("severe", "on"))
    if calm_on and severe_on:
        c = calm_on.get("probed_peak_accel_m_s2")
        s = severe_on.get("probed_peak_accel_m_s2")
        if c is not None and s is not None:
            report.observe(
                f"probed peak accel: calm={c:.3f} severe={s:.3f} "
                + ("(severe > calm, as intended)" if s > c else
                   "(NOT ordered -- one sample each, but worth a second look)"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=20260920)
    parser.add_argument("--name", default=None,
                        help="Output directory name. Defaults to a timestamp.")
    parser.add_argument("--logs-dir", type=Path, default=campaign.DEFAULT_LOGS_DIR)
    parser.add_argument("--bee-dir", type=Path, default=campaign.DEFAULT_BEE_DIR)
    parser.add_argument("--px4-dir", type=Path, default=campaign.DEFAULT_PX4_DIR)
    parser.add_argument("--source-world", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and check worlds; fly nothing.")
    args = parser.parse_args()

    source_world = args.source_world or (args.bee_dir / "worlds" / "bee_platform.sdf")
    if not source_world.is_file():
        print(f"scenario_test: source world not found: {source_world}",
              file=sys.stderr)
        return 1

    name = args.name or datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
    test_dir = args.logs_dir / name
    test_dir.mkdir(parents=True, exist_ok=True)

    specs = build_matrix(args.base_seed)
    report = Report()

    print(f"scenario_test: {len(specs)} runs -> {test_dir}")
    print(
        "scenario_test: platform motion = "
        + ("TRANSLATION ONLY (roll/pitch disabled)" if TRANSLATION_ONLY
           else "FULL 5-DOF excitation")
    )
    for spec in specs:
        print(f"  {spec.run_id}  (seed {spec.seed})")

    check_worlds(specs, test_dir, source_world, report)

    if args.dry_run:
        print(report.render())
        return 1 if report.failed else 0

    # A dirty machine before the first run would be blamed on the first run.
    pre = subprocess.run([str(AUTOMATION_DIR / "teardown.sh")],
                         check=False, capture_output=True, text=True)
    report.check(pre.returncode == 0, "machine was clean before the first run",
                 pre.stderr.strip()[:300])

    records: dict[str, dict] = {}
    for index, spec in enumerate(specs, start=1):
        print(f"\n[{index}/{len(specs)}] {spec.run_id}", flush=True)
        status = campaign.run_one(spec, test_dir, source_world,
                                  timeout_sec=args.timeout,
                                  bee_dir=args.bee_dir, px4_dir=args.px4_dir)
        print(f"    -> {status}", flush=True)

        run_dir = test_dir / "runs" / spec.run_id
        record = inspect_outcome(run_dir, spec, report)
        if record:
            records[spec.run_id] = record
            observe(record, spec, report)

        post = subprocess.run([str(AUTOMATION_DIR / "teardown.sh")],
                              check=False, capture_output=True, text=True)
        report.check(post.returncode == 0,
                     f"{spec.run_id}: machine clean afterwards",
                     post.stderr.strip()[:300])

    cross_check(records, specs, report)

    summary = report.render()
    print(summary)
    (test_dir / "smoke_report.txt").write_text(summary + "\n", encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
