#!/usr/bin/env bash
# Return the machine to a state where the next run starts clean.
#
# Orphan accumulation is the failure that silently corrupts a long campaign.
# A surviving `gz sim` is the worst of them: PX4 ignores PX4_GZ_WORLD when a
# simulation is already running, so the next run attaches to the OLD world --
# previous platform state, previous wreckage -- and spawns its vehicle as
# bee_x500_1 instead of bee_x500_0. The bridge keeps running happily, bridging
# contact topics nobody publishes, and touchdown detection silently stops
# working. Nothing errors. The runs just quietly stop meaning anything.
#
# Usage:
#   teardown.sh [PGID]      kill that process group first, then sweep
#   teardown.sh             sweep only
#
# The PGID form is normally called from INSIDE the group it is clearing, by
# run_once.sh's exit trap. That makes self-protection mandatory: a plain
# `kill -TERM -- -PGID` would kill this script and its caller along with the
# simulation, the caller's own TERM trap would re-enter, and the run would
# collapse into a loop of "Terminated". So the group is enumerated and this
# process, its caller, and every ancestor are removed before anything is
# signalled.
#
# Exit 0 if the machine ends clean, 1 if anything survived. The campaign runner
# treats a non-zero exit as fatal, because one orphan contaminates every run
# after it.
set -uo pipefail

AGENT_PORT="${AGENT_PORT:-8888}"
SWEEP_PATTERN='px4|gz sim|gz-sim|ruby.*gz|MicroXRCEAgent|parameter_bridge|bee_node'

log() { printf '[teardown] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Ancestry, so we never signal ourselves or whoever called us.
# ---------------------------------------------------------------------------
ancestors_of() {
	local pid="$1"
	while [[ -n "${pid}" && "${pid}" != "0" && "${pid}" != "1" ]]; do
		printf '%s\n' "${pid}"
		pid="$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')"
	done
}

PROTECTED="$(ancestors_of "$$" | sort -u)"
is_protected() { grep -qx "$1" <<<"${PROTECTED}"; }

# ---------------------------------------------------------------------------
# 1. The run's own process group, if we were given one.
#    Enumerated and filtered rather than signalled as a group: SIGTERM first so
#    PX4 and Gazebo get to flush, SIGKILL for whatever ignores it.
# ---------------------------------------------------------------------------
if [[ $# -ge 1 && -n "${1:-}" ]]; then
	pgid="$1"

	# Snapshot the group ONCE. Re-enumerating on every poll would never
	# terminate: the `ps` that does the enumerating is itself a member of the
	# group, so the list always contains at least one live process and the
	# wait runs its full duration on every single run.
	targets=""
	while read -r pid; do
		[[ -z "${pid}" ]] && continue
		is_protected "${pid}" && continue
		targets+="${pid} "
	done < <(ps -o pid= -g "${pgid}" 2>/dev/null | tr -d ' ')

	alive() {
		local pid rest=""
		for pid in $1; do
			kill -0 "${pid}" 2>/dev/null && rest+="${pid} "
		done
		printf '%s' "${rest}"
	}

	targets="$(alive "${targets}")"
	if [[ -n "${targets}" ]]; then
		log "SIGTERM to group ${pgid}: ${targets}"
		# shellcheck disable=SC2086
		kill -TERM ${targets} 2>/dev/null
		for _ in $(seq 1 100); do
			targets="$(alive "${targets}")"
			[[ -z "${targets}" ]] && break
			sleep 0.1
		done
		if [[ -n "${targets}" ]]; then
			log "group ${pgid} ignored SIGTERM; SIGKILL: ${targets}"
			# shellcheck disable=SC2086
			kill -KILL ${targets} 2>/dev/null
			sleep 0.3
		fi
	fi
fi

# ---------------------------------------------------------------------------
# 2. Sweep by name. Catches anything that escaped the group -- notably a
#    gz server PX4 spawned before we took ownership of it.
# ---------------------------------------------------------------------------
sweep() {
	local signal="$1"
	local pids raw
	# Exclude this process and every ancestor, or the sweep kills the teardown
	# and the launcher that invoked it.
	raw="$(pgrep -f "${SWEEP_PATTERN}" 2>/dev/null || true)"
	pids=""
	while read -r pid; do
		[[ -z "${pid}" ]] && continue
		is_protected "${pid}" && continue
		pids+="${pid}"$'\n'
	done <<<"${raw}"
	pids="$(printf '%s' "${pids}")"
	[[ -z "${pids}" ]] && return 0
	log "${signal} to: $(echo "${pids}" | tr '\n' ' ')"
	# shellcheck disable=SC2086
	kill "-${signal}" ${pids} 2>/dev/null
	return 0
}

sweep TERM
for _ in $(seq 1 50); do
	pgrep -f "${SWEEP_PATTERN}" >/dev/null 2>&1 || break
	sleep 0.1
done
sweep KILL
sleep 0.5

# ---------------------------------------------------------------------------
# 3. The agent's UDP port. A socket in TIME_WAIT is harmless, but a live
#    listener means something survived the sweep under a name we do not match.
# ---------------------------------------------------------------------------
if command -v fuser >/dev/null 2>&1; then
	fuser -k "${AGENT_PORT}/udp" >/dev/null 2>&1 || true
	sleep 0.2
fi

# ---------------------------------------------------------------------------
# 4. Assert, do not hope.
# ---------------------------------------------------------------------------
survivors=""
while read -r line; do
	[[ -z "${line}" ]] && continue
	pid="${line%% *}"
	is_protected "${pid}" && continue
	# run_once.sh's own command line carries the world path; do not count the
	# launcher as an orphan of the thing it launched.
	[[ "${line}" == *"teardown.sh"* || "${line}" == *"run_once.sh"* ]] && continue
	survivors+="${line}"$'\n'
done <<<"$(pgrep -af "${SWEEP_PATTERN}" 2>/dev/null || true)"
survivors="$(printf '%s' "${survivors}")"
if [[ -n "${survivors}" ]]; then
	log "MACHINE NOT CLEAN -- these survived:"
	printf '%s\n' "${survivors}" >&2
	exit 1
fi

if command -v ss >/dev/null 2>&1; then
	if ss -lunp 2>/dev/null | grep -q ":${AGENT_PORT}\b"; then
		log "MACHINE NOT CLEAN -- UDP ${AGENT_PORT} still bound"
		exit 1
	fi
fi

log "clean"
exit 0
