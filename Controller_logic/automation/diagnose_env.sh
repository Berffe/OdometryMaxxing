#!/usr/bin/env bash
# Answer "why did the world not launch?" without flying anything.
#
# Uses the same environment setup as run_once.sh (lib_env.sh), so what it finds
# is what the launcher will find. Runs in the foreground with output on the
# terminal, so a broken launch reports in seconds instead of after two
# 90-second timeouts.
#
#   ./diagnose_env.sh [path/to/world.sdf]
#
# With no argument it uses the nominal BEE_LAND world.
set -uo pipefail

AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BEE_DIR="${BEE_DIR:-${PX4_DIR}/BEE_LAND}"
WORLD="${1:-${BEE_DIR}/worlds/bee_platform.sdf}"
WORLD_NAME="bee_platform"

# shellcheck disable=SC1091
source "${AUTOMATION_DIR}/lib_env.sh"

say() { printf '\n=== %s ===\n' "$*"; }

say "environment"
bee_source_ros_and_venv
bee_setup_gz_paths "${PX4_DIR}" "${BEE_DIR}" "${WORLD}"
echo "ROS_DISTRO                : ${ROS_DISTRO:-unset}"
echo "BEE_VENV                  : ${BEE_VENV:-unset}"
echo "GZ_SIM_RESOURCE_PATH      : ${GZ_SIM_RESOURCE_PATH}"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH : ${GZ_SIM_SYSTEM_PLUGIN_PATH}"

say "gazebo"
echo "gz binaries on PATH, in order:"
type -a -p gz 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo
echo "GZ_CONFIG_PATH (inherited) : ${GZ_CONFIG_PATH:-<unset>}"
echo "sim*.yaml there            : $(ls "${GZ_CONFIG_PATH:-/nonexistent}"/sim*.yaml 2>/dev/null | tr '\n' ' ')"
echo "sim*.yaml in /usr/share/gz : $(ls /usr/share/gz/sim*.yaml 2>/dev/null | tr '\n' ' ')"
if bee_resolve_gz >/dev/null; then
	echo
	echo "  RESOLVED binary         : ${BEE_GZ_BIN}"
	echo "  RESOLVED GZ_CONFIG_PATH : ${BEE_GZ_CONFIG}"
	echo "  gz sim versions         : $(bee_gz sim --versions 2>/dev/null | tr '\n' ' ')"
	if [[ "${BEE_GZ_BIN}" != "$(command -v gz 2>/dev/null)" ]]; then
		echo
		echo "  NOTE: this is NOT the gz first on PATH. Sourcing ROS puts the"
		echo "        vendored CLI ahead of the real Gazebo."
	fi
	if [[ "${BEE_GZ_CONFIG}" != "${GZ_CONFIG_PATH:-}" ]]; then
		echo
		echo "  NOTE: the inherited GZ_CONFIG_PATH does not work; using"
		echo "        ${BEE_GZ_CONFIG} instead. The gz CLI reads its subcommand"
		echo "        definitions from YAML files in that ONE directory, and ROS"
		echo "        points it somewhere that has no sim*.yaml -- which is why"
		echo "        even /usr/bin/gz loses the 'sim' subcommand."
	fi
else
	echo
	bee_gz_not_found_message | sed 's/^/  /'
	exit 1
fi

say "required files"
for f in "${WORLD}" \
         "${PX4_DIR}/build/px4_sitl_default/bin/px4" \
         "${BEE_DIR}/bridge.sh"; do
	[[ -e "${f}" ]] && printf '  OK      %s\n' "${f}" || printf '  MISSING %s\n' "${f}"
done
for so in libWindController.so libOscillatingPlatformController.so libBeeLandingTruth.so; do
	found="$(find "${PX4_DIR}/build/px4_sitl_default" -name "${so}" 2>/dev/null | head -1)"
	[[ -n "${found}" ]] && printf '  OK      %s\n' "${found}" \
		|| printf '  MISSING %s (world loads, plugin silently does nothing)\n' "${so}"
done

say "other tools"
for tool in ros2 MicroXRCEAgent python3; do
	printf '  %-16s %s\n' "${tool}" "$(command -v "${tool}" || echo 'NOT FOUND')"
done

say "does the world parse?"
if bee_gz sdf --help >/dev/null 2>&1; then
	bee_gz sdf -k "${WORLD}" && echo "  parses cleanly"
else
	echo "  this gz has no 'sdf' subcommand; skipped"
fi

say "starting gz sim for 25s -- output below is the real answer"
bee_gz sim -s -r -v 4 "${WORLD}" &
gz_pid=$!
for i in $(seq 1 25); do
	if ! kill -0 "${gz_pid}" 2>/dev/null; then
		echo
		echo "  gz sim EXITED after ~${i}s. The output above is why."
		exit 1
	fi
	sleep 1
done

say "topics after 25s"
bee_gz topic -l 2>&1 | sort | sed 's/^/  /'

say "verdict"
rc=0
if bee_gz topic -l 2>/dev/null | grep -qE \
		"^(/clock|/world/${WORLD_NAME}/clock|/world/${WORLD_NAME}/stats)$"; then
	echo "  Gazebo is up and publishing. The launch itself is fine."
else
	echo "  Gazebo is running but advertises no clock/stats topic."
	echo "  That points at gz transport discovery rather than the world."
	echo "  Try: export GZ_IP=127.0.0.1  and re-run."
	rc=1
fi
if bee_gz topic -l 2>/dev/null | grep -q "^/bee_land/truth$"; then
	echo "  truth plugin is publishing."
else
	echo "  /bee_land/truth ABSENT -- libBeeLandingTruth.so did not load."
	echo "  Check GZ_SIM_SYSTEM_PLUGIN_PATH above against where the .so lives."
	rc=1
fi

kill -TERM "${gz_pid}" 2>/dev/null; sleep 1; kill -KILL "${gz_pid}" 2>/dev/null
echo
echo "done."
exit "${rc}"
