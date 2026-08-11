"""Measurements to an attitude/thrust setpoint.

One module today (``control_law``). It stays in its own folder because the
control law is a distinct responsibility from the mission that schedules its
gains -- the mission decides WHAT gains, this decides what command they produce.
"""
