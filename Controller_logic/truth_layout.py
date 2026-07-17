"""Fixed field layout for /bee_land/truth (ros_gz_interfaces/Float32Array).

This is intentionally dependency-free. The future truth logger can import
TRUTH_FIELDS and turn ``msg.data`` into a named CSV row.
"""

TRUTH_SCHEMA_VERSION = 1

TRUTH_FIELDS = [
    "truth_schema_version",
    "truth_sequence",
    "truth_sim_time_sec",
    "truth_physics_dt_sec",
    "truth_entities_ready",
    "truth_drone_position_x_m",
    "truth_drone_position_y_m",
    "truth_drone_position_z_m",
    "truth_drone_orientation_x",
    "truth_drone_orientation_y",
    "truth_drone_orientation_z",
    "truth_drone_orientation_w",
    "truth_drone_linear_velocity_x_m_s",
    "truth_drone_linear_velocity_y_m_s",
    "truth_drone_linear_velocity_z_m_s",
    "truth_drone_angular_velocity_x_rad_s",
    "truth_drone_angular_velocity_y_rad_s",
    "truth_drone_angular_velocity_z_rad_s",
    "truth_drone_linear_acceleration_x_m_s2",
    "truth_drone_linear_acceleration_y_m_s2",
    "truth_drone_linear_acceleration_z_m_s2",
    "truth_platform_position_x_m",
    "truth_platform_position_y_m",
    "truth_platform_position_z_m",
    "truth_platform_orientation_x",
    "truth_platform_orientation_y",
    "truth_platform_orientation_z",
    "truth_platform_orientation_w",
    "truth_platform_linear_velocity_x_m_s",
    "truth_platform_linear_velocity_y_m_s",
    "truth_platform_linear_velocity_z_m_s",
    "truth_platform_angular_velocity_x_rad_s",
    "truth_platform_angular_velocity_y_rad_s",
    "truth_platform_angular_velocity_z_rad_s",
    "truth_platform_linear_acceleration_x_m_s2",
    "truth_platform_linear_acceleration_y_m_s2",
    "truth_platform_linear_acceleration_z_m_s2",
    "truth_deck_point_x_m",
    "truth_deck_point_y_m",
    "truth_deck_point_z_m",
    "truth_deck_normal_x",
    "truth_deck_normal_y",
    "truth_deck_normal_z",
    "truth_left_pad_position_x_m",
    "truth_left_pad_position_y_m",
    "truth_left_pad_position_z_m",
    "truth_left_pad_velocity_x_m_s",
    "truth_left_pad_velocity_y_m_s",
    "truth_left_pad_velocity_z_m_s",
    "truth_left_pad_signed_distance_m",
    "truth_left_pad_closing_rate_m_s",
    "truth_right_pad_position_x_m",
    "truth_right_pad_position_y_m",
    "truth_right_pad_position_z_m",
    "truth_right_pad_velocity_x_m_s",
    "truth_right_pad_velocity_y_m_s",
    "truth_right_pad_velocity_z_m_s",
    "truth_right_pad_signed_distance_m",
    "truth_right_pad_closing_rate_m_s",
    "truth_min_pad_signed_distance_m",
    "truth_contact_pad_closing_rate_m_s",
    "truth_camera_position_x_m",
    "truth_camera_position_y_m",
    "truth_camera_position_z_m",
    "truth_camera_velocity_x_m_s",
    "truth_camera_velocity_y_m_s",
    "truth_camera_velocity_z_m_s",
    "truth_camera_normal_distance_m",
    "truth_camera_normal_closing_rate_m_s",
    "truth_camera_optical_range_m",
    "truth_camera_view_alignment",
    "truth_normal_expansion_rate_1_s",
    "truth_frontoparallel_flow_divergence_1_s",
    "truth_expansion_truth_valid",
    "truth_left_contact",
    "truth_right_contact",
    "truth_any_contact",
    "truth_contact_confirmed",
    "truth_left_contact_force_magnitude_n",
    "truth_right_contact_force_magnitude_n",
    "truth_left_contact_sensor_sim_time_sec",
    "truth_right_contact_sensor_sim_time_sec",
    "truth_left_contact_age_sec",
    "truth_right_contact_age_sec",
    "truth_contact_dwell_sec",
    "truth_geometric_crossing_latched",
    "truth_first_geometric_crossing_sim_time_sec",
    "truth_first_any_contact_sim_time_sec",
    "truth_contact_confirmed_sim_time_sec",
]

EXPECTED_FIELD_COUNT = len(TRUTH_FIELDS)


def decode_truth_array(data):
    """Return a named dict and fail loudly if producer / consumer disagree."""
    values = list(data)
    if len(values) != EXPECTED_FIELD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FIELD_COUNT} truth fields, got {len(values)}. "
            "The plugin and truth_layout.py schemas are out of sync."
        )
    if int(round(values[0])) != TRUTH_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported truth schema {values[0]!r}; expected {TRUTH_SCHEMA_VERSION}."
        )
    return dict(zip(TRUTH_FIELDS, values))
