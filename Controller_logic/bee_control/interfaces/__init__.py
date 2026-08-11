"""Everything that talks to the vehicle rather than about it.

``px4_interface``     uORB message construction and publication
``mavsdk_worker``     takeoff and termination side channel
``flight_sequencer``  the outer lifecycle: takeoff -> offboard -> handoff -> terminal

``flight_sequencer`` lives here rather than in ``core`` because its entire job is
negotiating with PX4 and MAVSDK; it is the policy layer over the other two.
"""
