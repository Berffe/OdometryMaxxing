"""
takeoff.py — PX4 SITL takeoff via MAVSDK
Usage: python3 takeoff.py [target_altitude_m]
"""

import asyncio
import sys
from mavsdk import System
import subprocess, os, signal

def free_mavsdk_port(port=14540):
    result = subprocess.run(
        ["lsof", "-t", f"-i:UDP:{port}"],
        capture_output=True, text=True
    )
    for pid in result.stdout.strip().split():
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Killed stale process {pid} on port {port}")
        except ProcessLookupError:
            pass

free_mavsdk_port()

TARGET_ALTITUDE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0e-3

# How long to wait after home is set before arming.
# Gives EKF2 time to converge while the drone sits on the platform.
EKF2_SETTLE_TIME = 5.0  # seconds


async def run():
	drone = System()
	await drone.connect(system_address="udp://0.0.0.0:14540")

	# ── Wait for connection ──────────────────────────────────────────────────
	print("Waiting for drone connection...")
	async for state in drone.core.connection_state():
		if state.is_connected:
			print("Connected to drone.")
			break

	# ── Wait for global + home position, then let EKF2 settle ───────────────
	print("Waiting for global position estimate...")
	# Replace the current health check loop with this
	async for health in drone.telemetry.health():
		if (health.is_global_position_ok 
			and health.is_home_position_ok
			and health.is_local_position_ok):   # ← add this
			print("All position estimates OK.")
			break
		
	# This delay is critical: EKF2 altitude can still drift for a few seconds
	# after home_position_ok fires, especially on an oscillating platform.
	await asyncio.sleep(EKF2_SETTLE_TIME)
	async for position in drone.telemetry.position():
		home_baro_offset = position.relative_altitude_m  # should be ~0 but may drift
		print(f"Home altitude offset: {home_baro_offset:.2f} m")
		break

	# ── Set takeoff altitude via param (guaranteed applied before takeoff) ───
	print(f"Setting takeoff altitude to {TARGET_ALTITUDE} m...")
	await drone.param.set_param_float("MIS_TAKEOFF_ALT", TARGET_ALTITUDE)
	# Small pause to ensure PX4 acknowledges the param write
	await asyncio.sleep(0.5)

	# ── Arm ──────────────────────────────────────────────────────────────────
	print("Arming...")
	await drone.action.arm()
	print("Armed.")

	# ── Take off ─────────────────────────────────────────────────────────────
	print(f"Taking off to {TARGET_ALTITUDE} m above home...")
	await drone.action.takeoff()

	# ── Monitor altitude until target is reached ─────────────────────────────
	async for position in drone.telemetry.position():
		alt = position.relative_altitude_m
		print(f"  Altitude: {alt:.2f}", end="\r")
		if (alt - home_baro_offset) >= TARGET_ALTITUDE - 0.2:
			print(f"\nReached {alt:.2f} m — hovering.")
			break

	# ── Hover then land ──────────────────────────────────────────────────────
	# await asyncio.sleep(5)
	# print("Landing...")
	# await drone.action.land()

	# async for armed in drone.telemetry.armed():
	# 	if not armed:
	# 		print("Disarmed. Done.")
	# 		break


if __name__ == "__main__":
	asyncio.run(run())