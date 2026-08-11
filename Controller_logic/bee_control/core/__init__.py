"""Shared primitives. Imports nothing else from ``bee_control``.

``clock``             the only three time sources in the system
``config``            every tuning knob, as frozen dataclasses
``state``             the ROS-free dataclasses exchanged between subsystems
``controller_state``  what the controller currently knows and last commanded
"""
