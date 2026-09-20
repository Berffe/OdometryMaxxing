#!/usr/bin/env bash
# Is the camera failing on the Gazebo side or the bridge side?
#
# The smoke test can only report "no camera images in ROS", which covers two
# very different faults. This separates them: it starts Gazebo, spawns PX4,
# and checks for frames on the GAZEBO topic before any bridge exists. Then it
# starts the bridge and checks the ROS topic.
#
#   ./diagnose_camera.sh [path/to/world.sdf]
set -uo pipefail

# Own session: PX4 and Gazebo must not share this terminal, and teardown needs
# a group it can clear.
if [[ "${BEE_DIAG_SESSION:-0}" != "1" ]]; then
	export BEE_DIAG_SESSION=1
	exec setsid --wait "$0" "$@"
fi

AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BEE_DIR="${BEE_DIR:-${PX4_DIR}/BEE_LAND}"
WORLD="${1:-${BEE_DIR}/worlds/bee_platform.sdf}"
WORLD_NAME="bee_platform"
CAMERA_TOPIC="/bee_x500/camera/image"

# shellcheck disable=SC1091
source "${AUTOMATION_DIR}/lib_env.sh"

say() { printf '\n=== %s ===\n' "$*"; }
cleanup() { "${AUTOMATION_DIR}/teardown.sh" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

bee_source_ros_and_venv
bee_setup_gz_paths "${PX4_DIR}" "${BEE_DIR}" "${WORLD}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
bee_resolve_gz >/dev/null || { bee_gz_not_found_message; exit 1; }
echo "gz: ${BEE_GZ_BIN} (GZ_CONFIG_PATH=${BEE_GZ_CONFIG})"

"${AUTOMATION_DIR}/teardown.sh" >/dev/null 2>&1

say "1. gazebo"
tmp="$(mktemp -d)"
# Headless rendering is required for unattended camera sensors.
# GZ_SIM_EXTRA_ARGS can still be used for additional Gazebo arguments.
# shellcheck disable=SC2086
bee_gz sim -s -r -v 4 --headless-rendering ${GZ_SIM_EXTRA_ARGS:-} "${WORLD}" \
	>"${tmp}/gz.log" 2>&1 </dev/null &

gazebo_up=0
for _ in $(seq 1 60); do
	if bee_gz topic -l 2>/dev/null | grep -qE \
			"^(/clock|/world/${WORLD_NAME}/clock|/world/${WORLD_NAME}/stats)$"; then
		gazebo_up=1
		break
	fi
	sleep 1
done

if [[ "${gazebo_up}" != "1" ]]; then
	echo "  FAILED -- gazebo never advertised clock/stats"
	tail -40 "${tmp}/gz.log" | sed 's/^/    /'
	exit 1
fi

echo "  up"
echo "  render-engine lines from gz.log:"
grep -iE "render|ogre|engine|GL|EGL|display" "${tmp}/gz.log" | head -15 | sed 's/^/    /' \
	|| echo "    (none)"

say "2. px4"
MicroXRCEAgent udp4 -p 8888 >"${tmp}/agent.log" 2>&1 </dev/null &
sleep 2
(
	cd "${PX4_DIR}" || exit 1
	HEADLESS=1 PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_SIMULATOR=gz \
	PX4_SIM_MODEL=bee_x500 PX4_GZ_MODEL=bee_x500 \
	PX4_GZ_MODEL_POSE="1.2,1.2,0.4,0,0,0" PX4_GZ_WORLD="${WORLD_NAME}" \
	./build/px4_sitl_default/bin/px4 -d
) >"${tmp}/px4.log" 2>&1 </dev/null &

for _ in $(seq 1 90); do
	bee_gz topic -l 2>/dev/null | grep -q "^${CAMERA_TOPIC}$" && break
	sleep 1
done

if bee_gz topic -l 2>/dev/null | grep -q "^${CAMERA_TOPIC}$"; then
	echo "  camera topic ADVERTISED in gazebo"
else
	echo "  camera topic NEVER ADVERTISED -- the model or its sensor did not load"
	grep -iE "error|warn|sensor|camera" "${tmp}/gz.log" | tail -20 | sed 's/^/    /'
	exit 1
fi

say "3. THE DECISIVE TEST: is gazebo publishing frames?"
echo "  (no bridge running yet, so this is purely the gazebo side)"

camera_sample="${tmp}/camera_sample.txt"
camera_err="${tmp}/camera_echo.err"

# `bee_gz` is a shell function, so it cannot be passed directly to `timeout`.
# Invoke the Gazebo executable resolved by bee_resolve_gz(), preserving the
# same GZ_CONFIG_PATH semantics used by bee_gz().
if [[ "${BEE_GZ_CONFIG}" == "${BEE_GZ_CONFIG_UNSET}" ]]; then
	timeout 15 env -u GZ_CONFIG_PATH "${BEE_GZ_BIN}" topic -e -n 1 -t "${CAMERA_TOPIC}" \
		>"${camera_sample}" 2>"${camera_err}"
	subscriber_rc=$?
else
	timeout 15 env GZ_CONFIG_PATH="${BEE_GZ_CONFIG}" \
		"${BEE_GZ_BIN}" topic -e -n 1 -t "${CAMERA_TOPIC}" \
		>"${camera_sample}" 2>"${camera_err}"
	subscriber_rc=$?
fi

if [[ "${subscriber_rc}" == "0" ]]; then
	if [[ -s "${camera_sample}" ]]; then
		echo
		echo "  YES -- gazebo is producing frames. Rendering is fine."
		echo "  First image message metadata:"
		grep -E "width:|height:|pixel_format|step:" "${camera_sample}" \
			| head -10 | sed 's/^/    /' || true
		gz_ok=1
	else
		echo
		echo "  NO -- subscriber exited successfully but produced no data."
		gz_ok=0
	fi
else
	rc="${subscriber_rc}"
	echo

	if [[ "${rc}" == "124" ]]; then
		echo "  NO -- timed out waiting 15 s for the first camera frame."
	else
		echo "  NO -- Gazebo topic subscriber failed with exit code ${rc}."
	fi

	echo "  Relevant gazebo log lines:"
	grep -iE "render|ogre|egl|engine|error|fail" "${tmp}/gz.log" \
		| tail -25 | sed 's/^/    /'

	echo
	echo "  Subscriber stderr:"
	tail -20 "${camera_err}" | sed 's/^/    /'

	echo
	echo "  Things to try, in order:"
	echo "    LIBGL_DRI3_DISABLE=1 ./diagnose_camera.sh"
	echo
	echo "    LIBGL_ALWAYS_SOFTWARE=1 ./diagnose_camera.sh"

	gz_ok=0
fi

say "4. the bridge"
if [[ "${gz_ok}" == "1" ]]; then
	# The ROS 2 CLI normally consults a background discovery daemon. A daemon
	# started earlier under another DDS / discovery configuration can make a
	# healthy bridge look invisible. Stop the daemon for this domain and use
	# direct discovery below so the diagnostic does not depend on cached graph
	# state.
	ros2 daemon stop >/dev/null 2>&1 || true

	echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
	echo "  ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-<unset>}"
	echo "  RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<default>}"

	"${BEE_DIR}/bridge.sh" >"${tmp}/bridge.log" 2>&1 </dev/null &
	bridge_pid=$!
	sleep 5

	if ! kill -0 "${bridge_pid}" 2>/dev/null; then
		echo "  BRIDGE PROCESS EXITED during startup."
		echo "  bridge.log:"
		tail -40 "${tmp}/bridge.log" | sed 's/^/    /'
	else
		echo "  bridge process is still alive (pid ${bridge_pid})"

		# Query the graph without the daemon. This tells us whether ROS discovery
		# can see the bridge publisher at all.
		ros_topics="${tmp}/ros_topics.txt"
		ros_nodes="${tmp}/ros_nodes.txt"
		ros2 topic list --no-daemon --spin-time 5 >"${ros_topics}" 2>"${tmp}/ros_topics.err" || true
		ros2 node list --no-daemon --spin-time 5 >"${ros_nodes}" 2>"${tmp}/ros_nodes.err" || true

		if grep -qx "${CAMERA_TOPIC}" "${ros_topics}"; then
			echo "  camera topic DISCOVERED directly on the ROS graph"
		else
			echo "  camera topic NOT discovered directly on the ROS graph"
		fi

		# Supply the type explicitly and bypass the daemon. BEST_EFFORT is safe
		# for sensor data and is compatible with a RELIABLE publisher as well.
		if timeout 20 ros2 topic echo \
				--no-daemon --spin-time 10 --qos-reliability best_effort --once \
				"${CAMERA_TOPIC}" sensor_msgs/msg/Image \
				>"${tmp}/ros_camera.txt" 2>"${tmp}/ros_camera.err"; then
			echo "  ROS side OK -- an image crossed the bridge."
		else
			echo "  NO ROS image received from the bridge."
			echo
			echo "  Directly discovered ROS topics of interest:"
			grep -E '^(/bee_|/platform/|/fmu/)' "${ros_topics}" | head -40 | sed 's/^/    /' \
				|| echo "    (none)"
			echo
			echo "  Directly discovered ROS nodes:"
			head -30 "${ros_nodes}" | sed 's/^/    /' || true
			echo
			echo "  camera subscriber stderr:"
			tail -20 "${tmp}/ros_camera.err" | sed 's/^/    /'
			echo
			echo "  bridge process:"
			pgrep -af 'parameter_bridge|ros_gz_bridge' | sed 's/^/    /' || echo "    (not found)"
			echo
			echo "  bridge.log:"
			tail -30 "${tmp}/bridge.log" | sed 's/^/    /'
		fi
	fi
else
	echo "  skipped: fix the gazebo side first."
fi

say "logs"
echo "  ${tmp}"
