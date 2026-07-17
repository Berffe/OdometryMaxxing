# BeeLandingTruth

Gazebo-native diagnostics truth for BEE_LAND. The plugin reads drone, camera,
platform and contact state in Gazebo, computes geometry / velocities on one SIM
clock, and publishes one fixed-layout `gz.msgs.Float_V` packet on
`/bee_land/truth`.

## Install in PX4's plugin tree

Copy this folder to:

```text
PX4-Autopilot/src/modules/simulation/gz_plugins/bee_landing_truth
```

Add to `src/modules/simulation/gz_plugins/CMakeLists.txt`:

```cmake
add_subdirectory(bee_landing_truth)
```

Build once:

```bash
cd ~/PX4-Autopilot
make px4_sitl
find build/px4_sitl_default -name 'libBeeLandingTruth.so'
```

Add `world_plugin_snippet.sdf` under the `<world>` element of
`bee_platform.sdf`, restart Gazebo, and verify:

```bash
gz topic -l | grep /bee_land/truth
gz topic -e -t /bee_land/truth
```

## ROS bridge

The packet uses a stock bridge mapping:

```text
gz.msgs.Float_V -> ros_gz_interfaces/msg/Float32Array
```

Use `bridge_with_truth.sh`, then verify:

```bash
ros2 topic info /bee_land/truth -v
ros2 topic echo /bee_land/truth --once
```

The legacy platform pose and contact bridges are deliberately retained during
bring-up as independent cross-checks. They can be removed after the truth path
has been validated.

## Geometry conventions

- Gazebo world coordinates, metres / seconds.
- Deck plane: `platform_link` local `z = +platform_top_offset_m`.
- Each skid distance is the minimum over its four physical bottom corners, so roll / pitch are handled.
- Positive pad signed distance: lowest skid point above deck.
- Negative pad signed distance: penetration below mathematical deck plane.
- Positive closing rate: moving toward the deck.
- Normal expansion truth: camera normal closing rate / camera normal distance (`c/h`).
- Fronto-parallel flow-divergence expectation: `2*c/h`, matching the mathematical trace `du/dx + dv/dy` when rotation / tilt effects are negligible.
- Camera optical axis is Gazebo camera `+X`; `camera_view_alignment` is 1 when
  the camera looks exactly opposite the deck normal.

## Why Float_V first

A custom Protobuf + custom ROS message would be prettier, but stock
`parameter_bridge` cannot bridge arbitrary custom message pairs without adding a
conversion package. `Float_V` preserves atomicity and works with the existing
bridge immediately. The schema is explicit and versioned in both the C++ enum
and `truth_layout.py`. All physics calculations remain double precision inside
Gazebo; only the transport envelope is float32.
