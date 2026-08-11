"""Controller-event and Gazebo-truth CSV logging.

The controller CSV records what the visual controller knew and commanded.
The truth CSV records every atomic Gazebo truth packet without reconstruction.
The two files share a run id and are merged later by ``analyse_log.py`` on
Gazebo SIM time.

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

The truth sink is unchanged: its layout comes from ``truth_layout.TRUTH_FIELDS``,
which is already a single source of truth shared with the Gazebo plugin.
"""
from __future__ import annotations

import csv
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
class _AsyncCsvSink:
    """Small non-blocking CSV sink used for the dense truth stream."""

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
            # Truth is dense. Dropping a row under pathological disk pressure is
            # safer than blocking the ROS executor / controller.
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


class DiagnosticsWriter:
    #: Bumped by hand when the MEANING of the base columns changes. The
    #: per-run fingerprint below covers accidental column drift automatically.
    CONTROLLER_SCHEMA_VERSION = "5.0-controller"
    TRUTH_LOG_SCHEMA_VERSION = "1.0-truth-log"

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
                 truth_queue_size: int = 2048):
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
            truth_name = "bee_truth_" + filename[len("bee_controller_"):]
        else:
            truth_name = f"bee_truth_{run_id}.csv"
        truth_path = root / truth_name

        self.filepath = str(controller_path)
        self.truth_filepath = str(truth_path)
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

    @property
    def fieldnames(self) -> Sequence[str]:
        return tuple(self._fieldnames)

    @property
    def truth_dropped_rows(self) -> int:
        return int(self._truth_sink.dropped_rows)

    def write_truth(self, truth: Mapping, *, receipt_wall_sec: float,
                    receipt_monotonic_sec: float):
        row = {
            "truth_log_schema_version": self.TRUTH_LOG_SCHEMA_VERSION,
            "truth_receipt_wall_timestamp_sec": float(receipt_wall_sec),
            "truth_receipt_monotonic_timestamp_sec": float(receipt_monotonic_sec),
        }
        row.update(truth)
        self._truth_sink.submit(row)

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
        if not self._controller_file.closed:
            self._controller_file.flush()
            self._controller_file.close()
