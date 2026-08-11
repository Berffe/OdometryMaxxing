"""The log schema contract and the CSV writers.

``telemetry``           the TelemetrySource protocol every logged subsystem implements
``diagnostics_writer``  assembles the header from its sources; writes both CSVs
``truth_layout``        the fixed Gazebo truth field layout, shared with the plugin

This package declares NO column names of its own beyond the eight base fields.
Every other column is owned by the subsystem that produces it.
"""
