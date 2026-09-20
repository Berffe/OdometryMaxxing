"""Controller-event, Gazebo-truth and wind-command CSV logging.

The controller CSV records what the visual controller knew and commanded.
The truth CSV records every atomic Gazebo truth packet without reconstruction.
The wind CSV records every bridged WindController diagnostic packet.
All three files share a run id and are merged later by ``analyse_log.py`` on
Gazebo SIM time.

Alongside them sits one JSON outcome record: not a time series but a single
verdict per run, which is why it is neither a fourth CSV nor a row appended to
an existing one. It carries the same run id, so the post-run pass that fills in
the truth-derived fields finds its inputs from the record itself.

Schema ownership
----------------
This module no longer knows what a mission phase, a probe or an optical-flow
stage timing is called.  It takes a list of ``TelemetrySource`` objects, builds
its header from ``source.telemetry_fields()``, and pulls ``source.telemetry()``
once per row.  Adding a column is a change in the subsystem that produces it.

Two failure modes that used to be silent are now loud:

* a value whose column was never declared raises ``TelemetrySchemaError``
  instead of being dropped (the old ``if col in row`` guard);
* two sources claiming the same column name raise at construction.

The dense diagnostic sinks keep independent schemas: physical truth comes from
``truth_layout.TRUTH_FIELDS`` and commanded wind from ``wind_layout.WIND_FIELDS``.
Neither schema is mixed into the controller log.
"""
from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

from bee_control.diagnostics.telemetry import (
    TelemetrySource,
    collect_fields,
    schema_fingerprint,
    snapshot,
)
from bee_control.diagnostics.truth_layout import TRUTH_FIELDS
from bee_control.diagnostics.wind_layout import WIND_FIELDS


class _AsyncCsvSink:
    """Small non-blocking CSV sink used for dense diagnostic streams."""

    def __init__(self, path: Path, fieldnames, *, queue_size: int = 2048,
                 flush_every_rows: int = 100):
        self.path = str(path)
        self._fieldnames = list(fieldnames)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._flush_every_rows = max(1, int(flush_every_rows))
        self._stop = object()
        self.dropped_rows = 0
        self._thread = threading.Thread(
            target=self._run, name=f"csv:{path.name}", daemon=True)
        self._thread.start()

    def submit(self, row: Mapping):
        try:
            self._queue.put_nowait(dict(row))
        except queue.Full:
            # Diagnostic streams must never block the ROS executor / controller.
            self.dropped_rows += 1

    def close(self):
        try:
            self._queue.put(self._stop, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)

    def _run(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writeheader()
            count = 0
            while True:
                item = self._queue.get()
                if item is self._stop:
                    break
                writer.writerow({k: item.get(k, "") for k in self._fieldnames})
                count += 1
                if count % self._flush_every_rows == 0:
                    handle.flush()
            handle.flush()


def _json_safe(value):
    """Map a non-finite float to None.

    ``rho`` is legitimately ``inf`` when the ceiling is unmeasurable, and NaN
    reaches these fields wherever an estimator never seeded. Both are real
    states worth recording, but neither is representable in JSON, so they are
    written as null rather than as a number that would read as a measurement.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DiagnosticsWriter:
    #: Bumped by hand when the MEANING of the base columns changes. The
    #: per-run fingerprint below covers accidental column drift automatically.
    CONTROLLER_SCHEMA_VERSION = "6.0-controller"
    TRUTH_LOG_SCHEMA_VERSION = "1.0-truth-log"
    WIND_LOG_SCHEMA_VERSION = "1.0-wind-log"
    OUTCOME_SCHEMA_VERSION = "1.0-outcome"

    #: Columns this writer owns outright. Everything else comes from sources.
    BASE_FIELDS = (
        "diagnostics_schema_version",
        "log_wall_timestamp_sec",
        "log_monotonic_timestamp_sec",
        "log_elapsed_wall_sec",
        "log_elapsed_monotonic_sec",
        "event",
        "event_detail",
        "controller_phase",
    )

    def __init__(self, sources: Sequence[TelemetrySource], output_dir="logs",
                 filename=None, *, strict: bool = True,
                 controller_flush_every_rows: int = 25,
                 truth_queue_size: int = 2048,
                 wind_queue_size: int = 2048):
        self._sources = list(sources)
        self._strict = bool(strict)

        source_fields = collect_fields(self._sources)
        self._fieldnames = list(self.BASE_FIELDS) + source_fields
        self.schema_fingerprint = schema_fingerprint(self._fieldnames)
        self.schema_version = (
            f"{self.CONTROLLER_SCHEMA_VERSION}+{self.schema_fingerprint}"
        )

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        filename = filename or f"bee_controller_{run_id}.csv"
        controller_path = root / filename
        if filename.startswith("bee_controller_"):
            suffix = filename[len("bee_controller_"):]
            truth_name = "bee_truth_" + suffix
            wind_name = "bee_wind_" + suffix
        else:
            truth_name = f"bee_truth_{run_id}.csv"
            wind_name = f"bee_wind_{run_id}.csv"
        truth_path = root / truth_name
        wind_path = root / wind_name
        if filename.startswith("bee_controller_"):
            outcome_name = "bee_outcome_" + Path(suffix).stem + ".json"
        else:
            outcome_name = f"bee_outcome_{run_id}.json"
        outcome_path = root / outcome_name

        self.run_id = run_id
        self.filepath = str(controller_path)
        self.truth_filepath = str(truth_path)
        self.wind_filepath = str(wind_path)
        self.outcome_filepath = str(outcome_path)
        self._start_wall = time.time()
        self._start_mono = time.monotonic()
        self._controller_flush_every_rows = max(1, int(controller_flush_every_rows))
        self._controller_row_count = 0
        self._controller_file = open(
            self.filepath, "w", newline="", encoding="utf-8")
        self._controller_writer = csv.DictWriter(
            self._controller_file, fieldnames=self._fieldnames)
        self._controller_writer.writeheader()
        self._truth_sink = _AsyncCsvSink(
            truth_path,
            [
                "truth_log_schema_version",
                "truth_receipt_wall_timestamp_sec",
                "truth_receipt_monotonic_timestamp_sec",
            ] + list(TRUTH_FIELDS),
            queue_size=truth_queue_size,
            flush_every_rows=100,
        )
        self._wind_sink = _AsyncCsvSink(
            wind_path,
            [
                "wind_log_schema_version",
                "wind_receipt_wall_timestamp_sec",
                "wind_receipt_monotonic_timestamp_sec",
            ] + list(WIND_FIELDS),
            queue_size=wind_queue_size,
            flush_every_rows=100,
        )

    @property
    def fieldnames(self) -> Sequence[str]:
        return tuple(self._fieldnames)

    @property
    def truth_dropped_rows(self) -> int:
        return int(self._truth_sink.dropped_rows)

    @property
    def wind_dropped_rows(self) -> int:
        return int(self._wind_sink.dropped_rows)

    def write_truth(self, truth: Mapping, *, receipt_wall_sec: float,
                    receipt_monotonic_sec: float):
        row = {
            "truth_log_schema_version": self.TRUTH_LOG_SCHEMA_VERSION,
            "truth_receipt_wall_timestamp_sec": float(receipt_wall_sec),
            "truth_receipt_monotonic_timestamp_sec": float(receipt_monotonic_sec),
        }
        row.update(truth)
        self._truth_sink.submit(row)

    def write_wind(self, wind: Mapping, *, receipt_wall_sec: float,
                   receipt_monotonic_sec: float):
        row = {
            "wind_log_schema_version": self.WIND_LOG_SCHEMA_VERSION,
            "wind_receipt_wall_timestamp_sec": float(receipt_wall_sec),
            "wind_receipt_monotonic_timestamp_sec": float(receipt_monotonic_sec),
        }
        row.update(wind)
        self._wind_sink.submit(row)

    def write_outcome(self, record: Mapping) -> str:
        """Write the run's outcome record. Returns the path written.

        Written atomically -- a temporary file then ``os.replace`` -- because
        the campaign runner keys resumability on this file's existence. A
        partially written record left behind by a crash mid-write would look
        like a finished run and the run would be skipped forever.

        ``postprocessed`` is False: this is the controller's half. The fields
        that come from the truth and controller CSVs (contact velocities, the
        realised peak relative acceleration, the measured loop period) are
        filled in by an offline pass, so nothing here has to compute physics
        from truth data mid-flight.
        """
        payload = {
            "outcome_schema_version": self.OUTCOME_SCHEMA_VERSION,
            "run_id": self.run_id,
            "postprocessed": False,
            "controller_csv": self.filepath,
            "truth_csv": self.truth_filepath,
            "wind_csv": self.wind_filepath,
        }
        payload.update(dict(record))
        payload = {k: _json_safe(v) for k, v in payload.items()}

        path = Path(self.outcome_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        with open(tmp, "w", encoding="utf-8") as handle:
            # allow_nan=False so a non-finite value that escaped _json_safe
            # raises here rather than emitting bare Infinity, which json.dump
            # accepts by default and strict parsers reject.
            json.dump(payload, handle, indent=2, sort_keys=False,
                      allow_nan=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return str(path)

    def write(self, *, event: str = "", event_detail: str = "",
              controller_phase: str = ""):
        """Snapshot every registered source and emit one controller row."""
        wall, mono = time.time(), time.monotonic()
        row = {k: "" for k in self._fieldnames}
        row.update({
            "diagnostics_schema_version": self.schema_version,
            "log_wall_timestamp_sec": wall,
            "log_monotonic_timestamp_sec": mono,
            "log_elapsed_wall_sec": wall - self._start_wall,
            "log_elapsed_monotonic_sec": mono - self._start_mono,
            "event": event,
            "event_detail": event_detail,
            "controller_phase": controller_phase,
        })
        for source in self._sources:
            row.update(snapshot(source, strict=self._strict))
        self._controller_writer.writerow(row)
        self._controller_row_count += 1
        if self._controller_row_count % self._controller_flush_every_rows == 0:
            self._controller_file.flush()

    def close(self):
        self._truth_sink.close()
        self._wind_sink.close()
        if not self._controller_file.closed:
            self._controller_file.flush()
            self._controller_file.close()
