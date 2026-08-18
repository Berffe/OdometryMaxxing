"""Fixed field layout for /bee_land/wind_cmd."""

WIND_SCHEMA_VERSION = 1

WIND_FIELDS = [
    "wind_schema_version",
    "wind_sequence",
    "wind_sim_time_sec",
    "wind_enabled",
    "wind_command_x_enu_m_s",
    "wind_command_y_enu_m_s",
    "wind_command_z_enu_m_s",
    "wind_command_norm_m_s",
    "wind_raw_x_enu_m_s",
    "wind_raw_y_enu_m_s",
    "wind_raw_z_enu_m_s",
    "wind_raw_norm_m_s",
    "wind_clamped",
]

EXPECTED_WIND_FIELD_COUNT = len(WIND_FIELDS)


def decode_wind_array(data):
    values = list(data)

    if len(values) != EXPECTED_WIND_FIELD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_WIND_FIELD_COUNT} wind fields, "
            f"got {len(values)}."
        )

    if int(round(values[0])) != WIND_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported wind schema {values[0]!r}; "
            f"expected {WIND_SCHEMA_VERSION}."
        )

    return dict(zip(WIND_FIELDS, values))