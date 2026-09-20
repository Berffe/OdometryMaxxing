"""Generate a per-run world from the nominal bee_platform.sdf.

Why this edits the SDF structurally instead of substituting @TOKEN@
------------------------------------------------------------------
Plain string substitution cannot express these edits. The wind changes from a
pair of fixed `<component>` blocks to a `<synthesis>` block, and the platform's
component counts differ by condition -- both change the NUMBER of child
elements, not just their values. A token template would have to hard-code one
shape and could never carry the other.

Editing the real file also means the nominal world stays the single source of
truth. A change to the lighting, the deck texture or the truth plugin's
geometry is picked up by every subsequent run with no second file to update --
which is exactly the desynchronisation that a `.in` copy invites.

The generated world is a build artifact: comments survive, indentation does
not. It is written into the run directory and kept. Local visual assets are NOT
copied into every run; instead, the run gets a ``materials`` symlink pointing
back to the canonical materials directory beside the ORIGINAL source world.
That preserves the nominal SDF's relative ``materials/...`` references while
keeping one editable texture / mesh tree for the whole project.

What is deliberately NOT touched
-------------------------------
- `<world name="bee_platform">`. The model name, the truth plugin's
  `contact_target_substring`, and both gear contact topics in `bridge.sh` are
  built from it. Renaming the world per run would break touchdown detection
  silently -- the bridge would keep running, bridging a topic nobody publishes.
- `<real_time_factor>`. The gate rests on a self-measured dead time and
  K_max is proportional to h/T; changing the RTF changes the relation between
  wall-clock vision latency and simulated time, so the campaign would validate
  a different controller than the paper describes. Asserted, not edited.
- The `bee_x500_0` model name in the truth plugin. That suffix is PX4's
  instance number, and it is only 0 when Gazebo starts clean -- see
  `verify_world_is_fresh` in run_once.sh.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario as scenario_mod  # noqa: E402

# The nominal deck: cylinder radius 0.5 m, and the flower mesh scaled by the
# same 0.5 so the texture disc exactly covers the physical deck. Scaling one
# without the other is the failure this constant exists to prevent -- the
# vehicle would then see a target whose apparent size does not match what it
# can land on.
NOMINAL_RADIUS_M = 0.5


def _parser_keeping_comments() -> ET.XMLParser:
    return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def _repair_comments(text: str) -> tuple[str, int]:
    """Make the SDF well-formed enough for a strict XML parser.

    XML forbids `--` anywhere inside a comment, and the nominal world contains
    at least one (`f2 = 0.16 Hz  <-- dominant`). Gazebo never notices because
    TinyXML2 is lenient, so the file has always worked; ElementTree rejects it
    outright. Repairing here rather than demanding the source be fixed means a
    future comment written the same way costs a warning instead of a failed
    campaign. Returns the repaired text and how many comments were touched.
    """
    out: list[str] = []
    repaired = 0
    cursor = 0
    while True:
        start = text.find("<!--", cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        end = text.find("-->", start + 4)
        if end < 0:
            out.append(text[cursor:])
            break
        body = text[start + 4:end]
        fixed = body
        while "--" in fixed:
            fixed = fixed.replace("--", "-")
        if fixed != body:
            repaired += 1
        out.append(text[cursor:start + 4])
        out.append(fixed)
        out.append("-->")
        cursor = end + 3
    return "".join(out), repaired


def _load_world(path: Path) -> ET.ElementTree:
    raw = path.read_text(encoding="utf-8")
    repaired_text, repaired = _repair_comments(raw)
    if repaired:
        print(f"make_world: repaired {repaired} XML comment(s) containing '--' "
              f"in {path.name}; Gazebo tolerates them, strict parsers do not.",
              file=sys.stderr)
    return ET.ElementTree(ET.fromstring(repaired_text,
                                        parser=_parser_keeping_comments()))


def _require(node, path: str, what: str):
    found = node.find(path)
    if found is None:
        raise SystemExit(
            f"make_world: could not find {what} ({path}) in the source world. "
            "The nominal SDF has changed shape; update make_world.py rather "
            "than letting the campaign fly an unmodified world.")
    return found


def _set_components(axis_element, components, phase_offset: float) -> None:
    """Replace an axis's `<component>` list.

    The phase offset is added to every component on the axis, which rotates the
    whole axis through its cycle while preserving the relative phasing between
    components. That relative phasing IS the sea state's shape; randomising it
    per component would make each repetition a different sea rather than a
    different moment in the same one.
    """
    for child in list(axis_element):
        if child.tag == "component":
            axis_element.remove(child)
    for amplitude, frequency, base_phase in components:
        component = ET.SubElement(axis_element, "component")
        ET.SubElement(component, "amplitude").text = f"{amplitude:.6f}"
        ET.SubElement(component, "frequency").text = f"{frequency:.6f}"
        phase = (base_phase + phase_offset) % (2.0 * math.pi)
        ET.SubElement(component, "phase").text = f"{phase:.6f}"


def _set_wind(plugin, wind) -> None:
    """Rewrite WindController to use its seeded synthesis path.

    The nominal world lists explicit `<component>` blocks, which means its
    `<seed>` controls nothing and every run gets an identical gust. Switching
    to `<synthesis>` is what makes the seed real: amplitudes, frequencies and
    phases are drawn from it, so repetitions of a condition are different
    realisations of the same statistics rather than the same wind fifteen times.
    """
    for tag in ("enabled", "seed", "mean_velocity", "max_wind_speed",
                "axis_x", "axis_y", "axis_z"):
        for child in plugin.findall(tag):
            plugin.remove(child)

    ET.SubElement(plugin, "enabled").text = "true" if wind.enabled else "false"
    ET.SubElement(plugin, "seed").text = str(wind.seed)
    ET.SubElement(plugin, "mean_velocity").text = (
        f"{wind.mean_velocity[0]:.4f} {wind.mean_velocity[1]:.4f} "
        f"{wind.mean_velocity[2]:.4f}")
    ET.SubElement(plugin, "max_wind_speed").text = f"{wind.max_wind_speed:.4f}"

    for tag, spec in (("axis_x", wind.axis_x), ("axis_y", wind.axis_y),
                      ("axis_z", wind.axis_z)):
        axis = ET.SubElement(plugin, tag)
        if not spec:
            continue
        count, amp_lo, amp_hi, freq_lo, freq_hi = spec
        synthesis = ET.SubElement(axis, "synthesis")
        ET.SubElement(synthesis, "count").text = str(int(count))
        ET.SubElement(synthesis, "amplitude_min").text = f"{amp_lo:.6f}"
        ET.SubElement(synthesis, "amplitude_max").text = f"{amp_hi:.6f}"
        ET.SubElement(synthesis, "frequency_min").text = f"{freq_lo:.6f}"
        ET.SubElement(synthesis, "frequency_max").text = f"{freq_hi:.6f}"


def _set_platform_radius(world, radius_m: float) -> None:
    """Resize the deck in all three places that describe it.

    The collision cylinder is what the skids touch, the visual cylinder is the
    deck's rim, and the flower mesh is the texture the controller actually
    tracks. They must move together: a mismatch gives a target whose apparent
    size does not correspond to the surface it can land on, which would show up
    in the data as a systematic scale bias rather than as an obvious failure.

    Height is left alone. Only the radius varies across the five sizes, so the
    truth plugin's `platform_top_offset_m` of 0.1 stays correct.
    """
    model = _require(world, "./model[@name='bee_platform']", "the platform model")
    link = _require(model, "./link[@name='platform_link']", "platform_link")

    visual_body = _require(link, "./visual[@name='visual_body']", "the deck visual")
    _require(visual_body, "./geometry/cylinder/radius",
             "the deck visual radius").text = f"{radius_m:.4f}"

    collision = _require(link, "./collision[@name='collision']", "the deck collision")
    _require(collision, "./geometry/cylinder/radius",
             "the deck collision radius").text = f"{radius_m:.4f}"

    visual_top = _require(link, "./visual[@name='visual_top']", "the flower visual")
    mesh_scale = _require(visual_top, "./geometry/mesh/scale", "the flower mesh scale")
    # The mesh's nominal scale equals the nominal radius, so the ratio carries
    # straight across. Z is left at 1.0: the disc is flat.
    factor = radius_m / NOMINAL_RADIUS_M
    nominal = [float(v) for v in (mesh_scale.text or "0.5 0.5 1.0").split()]
    mesh_scale.text = (f"{nominal[0] * factor:.4f} "
                       f"{nominal[1] * factor:.4f} {nominal[2]:.4f}")


def _link_original_materials(source_sdf: Path, destination: Path) -> None:
    """Expose the canonical world assets beside a generated per-run world.

    The nominal SDF deliberately uses relative paths such as
    ``materials/textures/flower_top.png``. Those are correct when the world is
    launched from ``BEE_LAND/worlds`` but stop resolving when automation writes
    ``world.sdf`` deep inside ``logs/.../runs/<run_id>/``.

    Do not copy the textures / meshes into every run. Instead create one tiny
    directory symlink:

        <run_dir>/materials -> <original BEE_LAND/worlds>/materials

    ``source_sdf.resolve()`` is important: if a caller passes PX4's "used"
    symlink under ``Tools/simulation/gz/worlds``, we still walk through it and
    anchor the run to the editable ORIGINAL tree under ``BEE_LAND/worlds``.

    Existing correct links are accepted because scenario_test generates worlds
    once during its preflight checks and campaign.run_one may generate the same
    world again. A conflicting real directory / wrong link is treated as an
    error rather than silently deleting user data.
    """
    try:
        original_world = source_sdf.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"make_world: source world does not exist: {source_sdf}") from exc

    original_materials = original_world.parent / "materials"
    if not original_materials.is_dir():
        raise SystemExit(
            "make_world: canonical materials directory is missing: "
            f"{original_materials}. Generated worlds keep relative "
            "materials/... paths, so this directory must exist beside the "
            "original bee_platform.sdf."
        )
    original_materials = original_materials.resolve(strict=True)

    run_materials = destination.parent / "materials"

    # Path.exists() is False for a broken symlink, so test is_symlink() first.
    if run_materials.is_symlink():
        current_target = run_materials.resolve(strict=False)
        if current_target == original_materials:
            return
        raise SystemExit(
            "make_world: run materials link points to the wrong source:\n"
            f"  link:     {run_materials}\n"
            f"  current:  {current_target}\n"
            f"  expected: {original_materials}"
        )

    if run_materials.exists():
        # This can happen only if destination itself lives in the originals
        # directory. In that special case the existing canonical directory is
        # already exactly what the relative SDF references need.
        try:
            if run_materials.resolve(strict=True) == original_materials:
                return
        except FileNotFoundError:
            pass
        raise SystemExit(
            "make_world: refusing to replace existing run asset path: "
            f"{run_materials}. Expected a symlink to {original_materials}."
        )

    run_materials.symlink_to(original_materials, target_is_directory=True)


def _assert_real_time_factor(world) -> None:
    rtf = world.find("./physics/real_time_factor")
    if rtf is None:
        return
    value = float(rtf.text or "1.0")
    if abs(value - 1.0) > 1e-9:
        raise SystemExit(
            f"make_world: real_time_factor is {value}, not 1.0. The feasibility "
            "gate rests on a self-measured dead time and K_max is proportional "
            "to h/T, so a different RTF validates a different controller. Fix "
            "the source world before running the campaign.")


def generate(source_sdf: Path, spec: "scenario_mod.Scenario",
             destination: Path) -> Path:
    # Resolve through the PX4 "used" symlink, if one was supplied, so both the
    # parsed world and the per-run materials reference come from the editable
    # BEE_LAND originals.
    source_sdf = source_sdf.expanduser().resolve(strict=True)
    destination = destination.expanduser()

    tree = _load_world(source_sdf)
    root = tree.getroot()
    world = _require(root, "./world", "the <world> element")

    _assert_real_time_factor(world)
    _set_platform_radius(world, spec.radius_m)

    wind_plugin = _require(
        world, "./model[@name='bee_platform']/plugin[@name='custom::WindController']",
        "the WindController plugin")
    _set_wind(wind_plugin, spec.wind)

    platform_plugin = _require(
        world,
        "./model[@name='bee_platform']"
        "/plugin[@name='custom::OscillatingPlatformController']",
        "the OscillatingPlatformController plugin")

    offsets = spec.platform.phase_offsets
    for tag, components, offset in (
            ("axis_x", spec.platform.surge, offsets[0]),
            ("axis_y", spec.platform.sway, offsets[1]),
            ("axis_z", spec.platform.heave, offsets[2]),
            ("axis_roll", spec.platform.roll, offsets[3]),
            ("axis_pitch", spec.platform.pitch, offsets[4])):
        axis = platform_plugin.find(tag)
        if axis is None:
            axis = ET.SubElement(platform_plugin, tag)
        _set_components(axis, components, offset)

    # Indent the whole tree: the generated world is the only evidence of what
    # a failed run actually flew, so it has to be readable by a human at 3 am.
    ET.indent(root, space="\t")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Keep the nominal world's relative materials/... references valid even
    # though the generated world lives under logs/.../runs/<run_id>/. This is
    # a reference to the canonical originals, not an asset copy.
    _link_original_materials(source_sdf, destination)

    tree.write(destination, encoding="unicode", xml_declaration=False)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="The nominal bee_platform.sdf.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Where to write the generated world.")
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--condition", required=True,
                        choices=scenario_mod.CONDITIONS)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--gate", required=True, choices=scenario_mod.GATES)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--write-spec", type=Path, default=None,
                        help="Also write the resolved scenario as JSON.")
    args = parser.parse_args()

    spec = scenario_mod.build(args.base_seed, args.condition, args.radius,
                              args.gate, args.repetition)
    generate(args.source, spec, args.out)
    if args.write_spec:
        args.write_spec.parent.mkdir(parents=True, exist_ok=True)
        args.write_spec.write_text(json.dumps(spec.to_json(), indent=2) + "\n")

    # The launcher reads these; keep the format trivially parseable.
    print(f"run_id={spec.run_id}")
    print(f"seed={spec.seed}")
    print("spawn_pose=" + ",".join(f"{v:.4f}" for v in spec.spawn_pose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
