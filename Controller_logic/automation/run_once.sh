#!/usr/bin/env bash
# One run: five processes, started in order, each gated on a real readiness
# signal, all in this script's own process session so teardown can kill the lot
# as a group.
#
# Why we start Gazebo instead of letting PX4 do it
# ------------------------------------------------
# PX4 ignores PX4_GZ_WORLD when a simulation is already running, and the gz
# server it spawns is a sibling process that outlives it. Owning the server
# ourselves gives us three things at once: teardown that actually works, a
# readiness signal to poll instead of a sleep, and a per-run world reached by
# PATH rather than by resource-path lookup.
#
# PX4_GZ_STANDALONE=1 is what makes PX4 wait for our server and attach to it.
# In standalone mode we also set PX4_GZ_MODEL explicitly rather than relying on
# PX4_SIM_MODEL as an alias, because the alias path disables PX4_GZ_MODEL_POSE
# and the spawn pose is part of the scenario.
#
# Usage:
#   run_once.sh --run-dir DIR --world FILE [--gate on|off] [--seed N]
#               [--timeout SEC]
#
# Exit codes are the harness's own vocabulary, distinct from the controller's:
#   0  the node ran to its own terminal outcome (landed/infeasible/aborted)
#   2  a launch stage never became ready        -> status "launch_failed"
#   3  the run exceeded its wall-clock budget   -> status "timeout"
set -uo pipefail

# --------------------------------------------------------------------------
# Own session. Everything started below inherits this process group, so
# teardown.sh can address the whole run with one negative PID.
# --------------------------------------------------------------------------
if [[ "${BEE_RUN_SESSION:-0}" != "1" ]]; then
	export BEE_RUN_SESSION=1
	exec setsid --wait "$0" "$@"
fi

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BEE_DIR="${BEE_DIR:-${PX4_DIR}/BEE_LAND}"
AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_PORT="${AGENT_PORT:-8888}"
WORLD_NAME="bee_platform"
DRONE_INSTANCE="bee_x500_0"

RUN_DIR=""; WORLD=""; GATE="on"; SEED=""; RUN_TIMEOUT="${RUN_TIMEOUT:-300}"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--run-dir) RUN_DIR="$2"; shift 2 ;;
		--world)   WORLD="$2";   shift 2 ;;
		--gate)    GATE="$2";    shift 2 ;;
		--seed)    SEED="$2";    shift 2 ;;
		--timeout) RUN_TIMEOUT="$2"; shift 2 ;;
		*) echo "run_once: unknown argument $1" >&2; exit 64 ;;
	esac
done
[[ -n "${RUN_DIR}" && -n "${WORLD}" ]] || { echo "run_once: --run-dir and --world are required" >&2; exit 64; }
[[ -f "${WORLD}" ]] || { echo "run_once: world not found: ${WORLD}" >&2; exit 64; }

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
log() { printf '[run_once] %s\n' "$*" | tee -a "${LOG_DIR}/run.log" >&2; }

# The trap disarms itself on entry. teardown signals members of this process
# group, and this script is one of them, so without disarming the resulting
# SIGTERM re-enters cleanup and the run collapses into a loop of "Terminated".
_cleanup_done=0
cleanup() {
	local code=$?
	[[ "${_cleanup_done}" == "1" ]] && exit "${code}"
	_cleanup_done=1
	trap - EXIT INT TERM
	log "tearing down (exit ${code})"
	"${AUTOMATION_DIR}/teardown.sh" "$$" >>"${LOG_DIR}/teardown.log" 2>&1
	exit "${code}"
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# Environment. Set up HERE rather than inherited from an interactive shell,
# so the campaign does not depend on anyone's .bashrc.
# --------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${AUTOMATION_DIR}/lib_env.sh"
bee_source_ros_and_venv

# Keep the campaign's DDS traffic off the subnet. Without this, any other ROS 2
# node reachable from this host can join the graph, and "run 88 was weird"
# becomes unexplainable after the fact.
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"

# Do not inherit cached graph state from an interactive ROS 2 daemon. All
# readiness probes below use --no-daemon, but stopping it here also keeps later
# ad-hoc diagnostics in this run from consulting stale discovery information.
ros2 daemon stop >/dev/null 2>&1 || true

bee_setup_gz_paths "${PX4_DIR}" "${BEE_DIR}" "${WORLD}"

# Never `gz` from PATH: sourcing ROS shadows the real Gazebo with a vendored
# CLI that has no `sim` subcommand. See lib_env.sh.
if ! bee_resolve_gz >/dev/null; then
	log "FATAL: no working gz found (binary and GZ_CONFIG_PATH together)"
	bee_gz_not_found_message | tee -a "${LOG_DIR}/run.log" >&2
	exit 2
fi
log "using gz: ${BEE_GZ_BIN} (GZ_CONFIG_PATH=${BEE_GZ_CONFIG})"

for tool in ros2 MicroXRCEAgent; do
	command -v "${tool}" >/dev/null 2>&1 || {
		log "FATAL: ${tool} not on PATH after sourcing the environment"
		exit 2
	}
done
[[ -x "${PX4_DIR}/build/px4_sitl_default/bin/px4" ]] || {
	log "FATAL: PX4 binary not found at ${PX4_DIR}/build/px4_sitl_default/bin/px4"
	exit 2
}
[[ -x "${BEE_DIR}/bridge.sh" ]] || {
	log "FATAL: ${BEE_DIR}/bridge.sh missing or not executable"
	exit 2
}

# --------------------------------------------------------------------------
# Readiness helpers. Every wait is a poll with a deadline; none is a sleep.
# --------------------------------------------------------------------------
# Which log to show when a stage never becomes ready, and which process must
# still be alive for waiting to make any sense.
WATCH_LOG=""
WATCH_PID=""

diagnose() {
	local what="$1"
	log "---- diagnosis: ${what} ----"
	if [[ -n "${WATCH_LOG}" && -s "${WATCH_LOG}" ]]; then
		log "last 40 lines of $(basename "${WATCH_LOG}"):"
		tail -n 40 "${WATCH_LOG}" | sed 's/^/    /' | tee -a "${LOG_DIR}/run.log" >&2
	elif [[ -n "${WATCH_LOG}" ]]; then
		log "$(basename "${WATCH_LOG}") is EMPTY -- the process wrote nothing at all"
	fi
	log "gz topics currently advertised:"
	{ bee_gz topic -l 2>&1 || echo "(gz topic -l failed)"; } | sed 's/^/    /' \
		| tee -a "${LOG_DIR}/run.log" >&2
	log "---- end diagnosis ----"
}

wait_for() {
	local what="$1" timeout="$2"; shift 2
	local deadline=$(( SECONDS + timeout ))
	while (( SECONDS < deadline )); do
		if "$@" >/dev/null 2>&1; then
			log "ready: ${what}"
			return 0
		fi
		# A dead process will never become ready. Failing here turns a silent
		# 90-second wait into an immediate, self-describing error -- which is
		# the difference between "TIMEOUT" and knowing why.
		if [[ -n "${WATCH_PID}" ]] && ! kill -0 "${WATCH_PID}" 2>/dev/null; then
			log "FAILED: the process for '${what}' exited before becoming ready"
			diagnose "${what}"
			return 1
		fi
		sleep 0.25
	done
	log "TIMEOUT waiting for ${what} (${timeout}s)"
	diagnose "${what}"
	return 1
}

gz_topic_exists() { bee_gz topic -l 2>/dev/null | grep -qx "$1"; }

# Gazebo publishes the plain /clock unless something else already publishes it,
# in which case only the world-qualified topic appears. Accept either, and fall
# back to /stats, which the server always advertises once a world is loaded.
gz_is_running() {
	bee_gz topic -l 2>/dev/null | grep -qE \
		"^(/clock|/world/${WORLD_NAME}/clock|/world/${WORLD_NAME}/stats)$"
}
# Query ROS directly instead of relying on the background ROS 2 daemon.  A
# daemon left over from an interactive shell can carry a different DDS / domain
# configuration and make a healthy publisher look invisible. BEST_EFFORT is
# compatible with both the camera sensor path and reliable publishers.
ros_topic_has_data() {
	local topic="$1" type="${2:-}"
	local cmd=(ros2 topic echo --no-daemon --spin-time 5
		--qos-reliability best_effort --once "${topic}")
	[[ -n "${type}" ]] && cmd+=("${type}")
	timeout 8 "${cmd[@]}" >/dev/null 2>&1
}

# `bee_gz` is a shell function, so it cannot be passed directly to `timeout`.
# Run the resolved Gazebo executable itself, preserving the GZ_CONFIG_PATH
# semantics selected by bee_resolve_gz().  `-n 1` makes the subscriber exit
# cleanly after one message; do NOT truncate an endless `gz topic -e` stream
# with `head` under pipefail, because the resulting SIGPIPE looks like failure.
gz_topic_has_data() {
	local topic="$1"
	if [[ "${BEE_GZ_CONFIG}" == "${BEE_GZ_CONFIG_UNSET}" ]]; then
		timeout 8 env -u GZ_CONFIG_PATH "${BEE_GZ_BIN}" \
			topic -e -n 1 -t "${topic}" >/dev/null 2>&1
	else
		timeout 8 env GZ_CONFIG_PATH="${BEE_GZ_CONFIG}" "${BEE_GZ_BIN}" \
			topic -e -n 1 -t "${topic}" >/dev/null 2>&1
	fi
}

# ==========================================================================
# 1. Gazebo server
# ==========================================================================
# Check the world parses before launching it: a malformed world makes gz exit
# almost immediately, and `gz sdf -k` names the offending element, which a
# server crash log often does not.
if bee_gz sdf --help >/dev/null 2>&1; then
	if bee_gz sdf -k "${WORLD}" >"${LOG_DIR}/world_check.log" 2>&1; then
		log "world parses"
	else
		log "FATAL: the generated world does not parse"
		tail -n 20 "${LOG_DIR}/world_check.log" | sed 's/^/    /' >&2
		exit 2
	fi
else
	log "note: this gz has no 'sdf' subcommand; skipping the world pre-check"
fi

log "starting gz sim on ${WORLD} (OGRE2 headless rendering enabled)"
# Camera sensors need an EGL render context during unattended runs. `-s` only
# suppresses the GUI; it does not by itself create a headless rendering surface.
# GZ_SIM_EXTRA_ARGS remains available for extra diagnostics / engine options.
# shellcheck disable=SC2086
bee_gz sim -s -r --headless-rendering ${GZ_SIM_EXTRA_ARGS:-} "${WORLD}" \
	>"${LOG_DIR}/gz.log" 2>&1 </dev/null &
gz_pid=$!
WATCH_LOG="${LOG_DIR}/gz.log"; WATCH_PID="${gz_pid}"
wait_for "gazebo server" 90 gz_is_running || exit 2

# The truth plugin is a world plugin; if its .so is missing the world still
# loads and the topic simply never appears. Catch it here rather than as a
# controller that never receives truth.
wait_for "truth plugin" 30 gz_topic_exists "/bee_land/truth" || exit 2
WATCH_PID=""

# ==========================================================================
# 2. Micro XRCE-DDS agent
# ==========================================================================
log "starting MicroXRCEAgent on udp4 ${AGENT_PORT}"
MicroXRCEAgent udp4 -p "${AGENT_PORT}" >"${LOG_DIR}/agent.log" 2>&1 </dev/null &
WATCH_LOG="${LOG_DIR}/agent.log"; WATCH_PID=$!
wait_for "agent port" 30 bash -c "ss -lun 2>/dev/null | grep -q ':${AGENT_PORT}\b'" || exit 2

# ==========================================================================
# 3. PX4, attaching to our server
# ==========================================================================
SPAWN_POSE="${SPAWN_POSE:-1.2,1.2,0.4,0,0,0}"
# -d runs PX4 without its interactive pxh console. Without it PX4 tries to read
# stdin; a backgrounded process that reads the terminal is stopped by SIGTTIN
# and never becomes ready. setsid happens to hide this here, but relying on
# that makes the same launch hang anywhere it is run without a new session.
log "starting PX4 (standalone, no console, pose ${SPAWN_POSE})"
(
	cd "${PX4_DIR}" || exit 1
	# PX4_SIM_MODEL must be set as well as PX4_GZ_MODEL. Autostart 4001
	# defaults PX4_SIM_MODEL to plain "x500", and leaving it at that default
	# makes PX4 spawn a SECOND vehicle (x500_0) alongside bee_x500_0 -- two
	# drones in one world, with PX4 bound to whichever it spawned last.
	# Setting both to bee_x500 keeps the one model while PX4_GZ_MODEL is what
	# lets PX4_GZ_MODEL_POSE apply at all.
	HEADLESS=1 \
	PX4_GZ_STANDALONE=1 \
	PX4_SYS_AUTOSTART=4001 \
	PX4_SIMULATOR=gz \
	PX4_SIM_MODEL=bee_x500 \
	PX4_GZ_MODEL=bee_x500 \
	PX4_GZ_MODEL_POSE="${SPAWN_POSE}" \
	PX4_GZ_WORLD="${WORLD_NAME}" \
	./build/px4_sitl_default/bin/px4 -d
) >"${LOG_DIR}/px4.log" 2>&1 </dev/null &
WATCH_LOG="${LOG_DIR}/px4.log"; WATCH_PID=$!

# PX4 renames this topic across releases and the running build decides which
# one appears -- v4 on current PX4. These candidates MUST stay in step with
# TopicsConfig.vehicle_status in core/config.py: the controller subscribes to
# all of them and takes whichever exists, so a launcher that waits on a
# narrower list times out while the system is perfectly healthy.
VEHICLE_STATUS_TOPICS=(
	/fmu/out/vehicle_status_v4
	/fmu/out/vehicle_status_v1
	/fmu/out/vehicle_status
)

px4_is_publishing() {
	local topic
	for topic in "${VEHICLE_STATUS_TOPICS[@]}"; do
		# "The topic exists" is not "PX4 is alive": a stale DDS discovery record
		# can advertise a topic with no publisher. Require an actual message.
		if ros_topic_has_data "${topic}"; then
			return 0
		fi
	done
	return 1
}

wait_for "PX4 telemetry" 120 px4_is_publishing || exit 2

# ----------------------------------------------------------------------
# The orphan canary.
#
# bridge.sh and the truth plugin both hard-code bee_x500_0. That _0 is PX4's
# instance suffix, and it is 0 only when Gazebo started clean. If a stale
# server survived a previous run, PX4 spawns bee_x500_1 instead: every process
# keeps running, the bridge keeps bridging, and contacts silently never fire.
# Checking the camera topic exists under the expected name catches it at
# launch, where it costs one run, instead of at analysis, where it costs many.
# ----------------------------------------------------------------------
vehicle_models="$(bee_gz topic -l 2>/dev/null \
	| sed -n "s|^/world/${WORLD_NAME}/model/\([^/]*\)/.*|\1|p" \
	| grep -v "^${WORLD_NAME}$" | sort -u)"

if ! grep -qx "${DRONE_INSTANCE}" <<<"${vehicle_models}"; then
	log "FATAL: ${DRONE_INSTANCE} not present. A stale Gazebo server almost"
	log "       certainly survived a previous run and PX4 attached to it."
	log "       vehicle models found: $(tr '\n' ' ' <<<"${vehicle_models}")"
	exit 2
fi

# Exactly one vehicle, or the run is meaningless. bridge.sh and the truth
# plugin both hard-code bee_x500_0, so a second vehicle is not a harmless
# extra: PX4 may bind to it, the camera and contacts come from the other, and
# every process keeps running while the data describes nobody in particular.
extra_vehicles="$(grep -vx "${DRONE_INSTANCE}" <<<"${vehicle_models}" || true)"
if [[ -n "${extra_vehicles}" ]]; then
	log "FATAL: more than one vehicle in the world."
	log "       expected only ${DRONE_INSTANCE}, also found: $(tr '\n' ' ' <<<"${extra_vehicles}")"
	log "       Usually PX4_SIM_MODEL fell back to its autostart default and"
	log "       PX4 spawned a stock airframe beside the bee model."
	exit 2
fi
log "ready: exactly one vehicle (${DRONE_INSTANCE})"

# Prove the camera renders BEFORE the bridge exists, so a rendering failure
# cannot be mistaken for a bridge failure. The camera is always_on, so frames
# should appear promptly after the model spawns; a longer wait
# here means the Sensors system never produced an image, which on a headless
# WSL host usually means the render engine failed to initialise.
WATCH_LOG="${LOG_DIR}/gz.log"; WATCH_PID=""
if ! wait_for "camera frames from Gazebo" 45 gz_topic_has_data "/bee_x500/camera/image"; then
	log "The camera topic is advertised but Gazebo is publishing no frames."
	log "This is a RENDERING problem, not a bridge problem -- the bridge has"
	log "not been started yet. Headless OGRE2 is already enabled. Check gz.log"
	log "for EGL / OpenGL errors; on WSL useful fallbacks are:"
	log "    LIBGL_DRI3_DISABLE=1 ./run_once.sh ..."
	log "    LIBGL_ALWAYS_SOFTWARE=1 ./run_once.sh ..."
	exit 2
fi

# ==========================================================================
# 4. Bridge
# ==========================================================================
log "starting bridge"
"${BEE_DIR}/bridge.sh" >"${LOG_DIR}/bridge.log" 2>&1 </dev/null &
WATCH_LOG="${LOG_DIR}/bridge.log"; WATCH_PID=$!

wait_for "camera images" 60 ros_topic_has_data \
	"/bee_x500/camera/image" "sensor_msgs/msg/Image" || exit 2
wait_for "truth packets" 30 ros_topic_has_data \
	"/bee_land/truth" "ros_gz_interfaces/msg/Float32Array" || exit 2
wait_for "wind packets" 30 ros_topic_has_data \
	"/bee_land/wind_cmd" "ros_gz_interfaces/msg/Float32Array" || exit 2
# The gear contact topics are the touchdown evidence. They are bridged from
# sensors that only publish on contact, so we check the SUBSCRIPTION exists
# rather than waiting for a message that will not arrive until landing.
wait_for "gear contact bridge" 30 \
	bash -c 'ros2 topic list --no-daemon --spin-time 5 2>/dev/null | grep -q "^/bee/gear_contacts/left$"' || exit 2

# ==========================================================================
# 5. The controller
# ==========================================================================
NODE_ARGS=(--outdir "${RUN_DIR}" --gate "${GATE}")
[[ -n "${SEED}" ]] && NODE_ARGS+=(--seed "${SEED}")

log "starting bee_node (gate=${GATE}, timeout=${RUN_TIMEOUT}s)"
(
	cd "${BEE_DIR}/controller" || exit 1
	python3 -m bee_control.bee_node "${NODE_ARGS[@]}"
) >"${LOG_DIR}/node.log" 2>&1 </dev/null &
node_pid=$!

# The node ends itself on any terminal outcome. This budget exists only for the
# case where it does not -- a hang no operational gate catches.
deadline=$(( SECONDS + RUN_TIMEOUT ))
while kill -0 "${node_pid}" 2>/dev/null; do
	if (( SECONDS >= deadline )); then
		log "run exceeded ${RUN_TIMEOUT}s; killing the node"
		kill -TERM "${node_pid}" 2>/dev/null
		sleep 3
		kill -KILL "${node_pid}" 2>/dev/null
		exit 3
	fi
	sleep 0.5
done

wait "${node_pid}"
node_exit=$?
log "bee_node exited ${node_exit}"
exit 0
