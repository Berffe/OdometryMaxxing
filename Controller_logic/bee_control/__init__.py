"""BEE_LAND: bio-inspired visual landing controller.

Layout
------
``bee_node.py``   the ROS 2 node. Entry point, top level, on its own.
``core/``         shared primitives: clock, config, exchanged dataclasses, live state
``vision/``       camera -> measurements. Runs in a separate process.
``mission/``      the visual landing sequence (a package; see mission/routine.py)
``control/``      the control law that turns measurements into a setpoint
``interfaces/``   talking to the vehicle: PX4, MAVSDK, the outer lifecycle
``diagnostics/``  the log schema contract and the CSV writers
``tests/``        ROS-free contract tests

Design notes live in ``docs/``: ``WIND_REJECTION.md`` covers the static/dynamic
acceleration split that gives the near field its steady-wind rejection.

Dependency direction is one-way, top of this list to bottom: ``core`` imports
nothing else in the package; ``bee_node`` imports everything and is imported by
nothing. If you ever need an import that points back up this list, that is the
signal a seam is in the wrong place.
"""
