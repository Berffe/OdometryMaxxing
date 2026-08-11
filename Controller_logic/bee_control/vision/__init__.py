"""Camera frames to visual measurements.

``target_acquisition``  NN-free colourfulness detector -> TargetEstimate
``optical_flow``        Farneback + affine divergence fit -> FlowResult
``derotation``          body-rate compensation and the angular-rate buffer
``vision_worker``       the out-of-process loop that runs the two above

Nothing in here may import rclpy or touch DDS: ``vision_worker`` is spawned as a
clean interpreter and must stay light and DDS-free.
"""
