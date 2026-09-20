#!/usr/bin/env bash
# Shared environment setup. Sourced by run_once.sh and diagnose_env.sh so the
# launch and the diagnostic can never disagree about what the environment is.
#
# The Gazebo resolution problem
# -----------------------------
# Two separate things break when ROS is sourced, and either alone is enough to
# make `gz sim` disappear.
#
# 1. PATH. /opt/ros/<distro>/setup.bash prepends the ROS-vendored `gz` CLI
#    (gz_tools_vendor), which provides only log/msg/param/service/topic.
#
# 2. GZ_CONFIG_PATH. The `gz` CLI does not have subcommands built in: it
#    discovers them from YAML files in the directory GZ_CONFIG_PATH names, and
#    that variable holds a SINGLE path, not a list. ROS points it at the
#    vendored share directory, which carries transport/msgs YAMLs but no
#    sim8.yaml -- so the system /usr/share/gz is masked completely and EVERY
#    gz on PATH loses `sim`, including /usr/bin/gz from a perfectly good
#    Gazebo install.
#
# The second is the nastier one, because the obvious fix -- "call /usr/bin/gz
# directly" -- does not work. The binary is fine; it simply cannot see its own
# subcommand definitions.
#
# Both are invisible in manual use: the terminal that launches PX4 typically
# has not sourced ROS. Only automation that sources ROS *and* calls `gz sim`
# trips over them.
#
# So we resolve a (binary, config directory) PAIR by probing combinations, and
# route every gz call through it -- sim and topic alike, since mixing CLIs or
# config paths across installs risks the two not seeing each other's transport.

bee_source_ros_and_venv() {
	# `set -u` has to come off across these: ROS's setup.bash reads unbound
	# variables (AMENT_TRACE_SETUP_FILES among them) as a matter of course, and
	# so do most venv activate scripts. With -u on, sourcing aborts the launch
	# before a single process starts.
	local had_u=0
	[[ $- == *u* ]] && had_u=1
	set +u
	# shellcheck disable=SC1090
	source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
	if [[ -n "${BEE_VENV:-}" && -f "${BEE_VENV}/bin/activate" ]]; then
		# shellcheck disable=SC1090
		source "${BEE_VENV}/bin/activate"
	fi
	[[ "${had_u}" == "1" ]] && set -u
	return 0
}

# Sentinel meaning "run with GZ_CONFIG_PATH unset entirely", which is distinct
# from running it set to the empty string.
BEE_GZ_CONFIG_UNSET="__unset__"

# Run gz once with a specific config path. Kept separate so the probe and the
# wrapper cannot drift apart.
bee_gz_with() {
	local config="$1"; shift
	local binary="$1"; shift
	if [[ "${config}" == "${BEE_GZ_CONFIG_UNSET}" ]]; then
		env -u GZ_CONFIG_PATH "${binary}" "$@"
	else
		GZ_CONFIG_PATH="${config}" "${binary}" "$@"
	fi
}

# Probes (binary, config) pairs and, on success, sets BEE_GZ_BIN and
# BEE_GZ_CONFIG so bee_gz can be used afterwards. Returns 1 if nothing works.
bee_resolve_gz() {
	local binary config output
	local binaries=() configs=()

	# An explicit GZ_BIN always wins, so an unusual layout can be fixed without
	# editing any script.
	[[ -n "${GZ_BIN:-}" ]] && binaries+=("${GZ_BIN}")
	while IFS= read -r binary; do
		binaries+=("${binary}")
	done < <(type -a -p gz 2>/dev/null || true)
	binaries+=(/usr/bin/gz /usr/local/bin/gz /snap/bin/gz)

	# Move ROS-vendored binaries to the back. Pairing that CLI with the system
	# gz-sim libraries does work, but it mixes two installs, and it made
	# run_once.sh and diagnose_camera.sh resolve to different binaries on the
	# same machine -- which is exactly the sort of difference that makes a
	# failure reproduce in one script and not the other.
	local preferred=() deferred=()
	for binary in "${binaries[@]}"; do
		case "${binary}" in
			*/opt/ros/*) deferred+=("${binary}") ;;
			*)           preferred+=("${binary}") ;;
		esac
	done
	binaries=("${preferred[@]}" "${deferred[@]}")

	[[ -n "${GZ_CONFIG_PATH_OVERRIDE:-}" ]] && configs+=("${GZ_CONFIG_PATH_OVERRIDE}")
	# Whatever the environment already says, first: if it works, changing it
	# would only risk pointing at a different Gazebo than the rest of the setup.
	[[ -n "${GZ_CONFIG_PATH:-}" ]] && configs+=("${GZ_CONFIG_PATH}")
	configs+=(/usr/share/gz /usr/local/share/gz "${BEE_GZ_CONFIG_UNSET}")

	for binary in "${binaries[@]}"; do
		[[ -x "${binary}" ]] || continue
		for config in "${configs[@]}"; do
			[[ "${config}" == "${BEE_GZ_CONFIG_UNSET}" || -d "${config}" ]] || continue
			# Exit status alone is not enough: a CLI without the subcommand may
			# print its help text and still exit 0, and would then be selected
			# -- which is the exact situation we are trying to detect. Require
			# output that actually looks like a version number.
			output="$(bee_gz_with "${config}" "${binary}" sim --versions \
				2>/dev/null | head -n 1)"
			if [[ "${output}" =~ ^[0-9]+\.[0-9]+ ]]; then
				BEE_GZ_BIN="${binary}"
				BEE_GZ_CONFIG="${config}"
				export BEE_GZ_BIN BEE_GZ_CONFIG
				printf '%s' "${binary}"
				return 0
			fi
		done
	done
	return 1
}

# The only way any script should invoke gz after resolution.
bee_gz() {
	bee_gz_with "${BEE_GZ_CONFIG:-${BEE_GZ_CONFIG_UNSET}}" \
		"${BEE_GZ_BIN:-gz}" "$@"
}

# Deduplicate a colon-separated path, preserving order. Repeated entries are
# harmless but make the diagnostic output hard to read.
bee_dedupe_path() {
	local input="$1" out="" item
	local IFS=':'
	for item in ${input}; do
		[[ -z "${item}" ]] && continue
		case ":${out}:" in
			*":${item}:"*) continue ;;
		esac
		out="${out:+${out}:}${item}"
	done
	printf '%s' "${out}"
}

bee_setup_gz_paths() {
	local px4_dir="$1" bee_dir="$2" world="${3:-}"

	# Our bee_x500 model is an overlay: its model.sdf includes PX4's stock
	# model://x500_base.  A normal PX4 `make px4_sitl gz_*` launch gets the
	# stock model directory from build/px4_sitl_default/rootfs/gz_env.sh, but
	# this automation starts Gazebo directly and therefore has to reproduce
	# that part of the environment explicitly.
	#
	# Keep BEE_LAND first so a custom model with the same name wins over a PX4
	# or ~/.simulation-gazebo copy, while nested stock includes (x500_base,
	# meshes, etc.) can still resolve afterwards.
	local px4_models="${px4_dir}/Tools/simulation/gz/models"
	local px4_worlds="${px4_dir}/Tools/simulation/gz/worlds"
	local standalone_models="${HOME}/.simulation-gazebo/models"
	local standalone_worlds="${HOME}/.simulation-gazebo/worlds"
	local resource="${bee_dir}/worlds:${bee_dir}/models"

	[[ -d "${px4_models}" ]] && resource="${resource}:${px4_models}"
	[[ -d "${px4_worlds}" ]] && resource="${resource}:${px4_worlds}"
	[[ -d "${standalone_models}" ]] && resource="${resource}:${standalone_models}"
	[[ -d "${standalone_worlds}" ]] && resource="${resource}:${standalone_worlds}"
	[[ -n "${GZ_SIM_RESOURCE_PATH:-}" ]] && resource="${resource}:${GZ_SIM_RESOURCE_PATH}"

	if [[ -n "${world}" && -f "${world}" ]]; then
		# The generated world references materials/ relatively, so its own
		# directory must be searchable too.
		resource="$(cd "$(dirname "${world}")" && pwd):${resource}"
	fi
	export GZ_SIM_RESOURCE_PATH="$(bee_dedupe_path "${resource}")"
	export GZ_SIM_SYSTEM_PLUGIN_PATH="$(bee_dedupe_path \
		"${px4_dir}/build/px4_sitl_default/src/modules/simulation/gz_plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}")"
}

# Printed whenever Gazebo cannot be found, because the failure is unhelpful on
# its own -- `gz` exists, it just cannot simulate.
bee_gz_not_found_message() {
	cat <<'MSG'
No working (gz binary, GZ_CONFIG_PATH) combination was found.

Two independent things can cause this, and both come from sourcing ROS:

  1. PATH -- the ROS-vendored CLI (gz_tools_vendor) has no `sim` subcommand.
  2. GZ_CONFIG_PATH -- the `gz` CLI reads its subcommand definitions from
     YAML files in the ONE directory this variable names. ROS points it at
     the vendored share directory, which has no sim*.yaml, so even a perfectly
     good /usr/bin/gz cannot see `sim`.

Check what is actually installed and where the YAMLs are:

    type -a gz
    echo "GZ_CONFIG_PATH=${GZ_CONFIG_PATH:-<unset>}"
    ls /usr/share/gz/*.yaml 2>/dev/null
    apt list --installed 2>/dev/null | grep -i 'gz-\(harmonic\|garden\|sim\)'

Then try the pair by hand:

    GZ_CONFIG_PATH=/usr/share/gz /usr/bin/gz sim --versions

If that works, this script should have found it -- please report it. If the
YAMLs live elsewhere, point at them:

    export GZ_CONFIG_PATH_OVERRIDE=/path/to/share/gz
    export GZ_BIN=/path/to/gz
MSG
}
