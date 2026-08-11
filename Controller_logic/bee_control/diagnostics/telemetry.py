"""Telemetry contract: every subsystem owns its own log schema.

Before this module, a new mission phase (or a new tuning knob worth logging)
had to be declared in three places: the subsystem that produced it,
``bee_node._mission_dict()`` which hand-transcribed it, and
``DiagnosticsWriter._fieldnames()`` which hardcoded the column list.  Anything
missed in step three was *silently dropped* at write time.

A ``TelemetrySource`` collapses those three into one.  A subsystem declares:

    TELEMETRY_PREFIX = "mission"                 # column namespace, "" for none
    @classmethod
    def telemetry_fields(cls) -> Sequence[str]   # STATIC, header-time
    def telemetry(self) -> Mapping[str, object]  # live, per-row

``DiagnosticsWriter`` assembles its header from the registered sources and
pulls a snapshot from each one per row.  Nothing else needs to know the names.

Why ``telemetry_fields`` must be static
---------------------------------------
A CSV header is written once, at file-open time, and ``analyse_log.py`` reads
positionally-stable named columns.  A source is therefore not allowed to
invent a column mid-run.  ``strict=True`` on the writer turns that from a
silent truncation into a loud ``TelemetrySchemaError``.

Schema fingerprint
------------------
``schema_fingerprint()`` hashes the assembled column list.  It rides in the
``diagnostics_schema_version`` cell, so a log that lost or renamed a column
is recognisable offline without diffing headers by eye.
"""
from __future__ import annotations

import hashlib
from typing import ClassVar, Iterable, Mapping, Protocol, Sequence, runtime_checkable


class TelemetrySchemaError(RuntimeError):
    """A source emitted a key it never declared, or two sources collided."""


@runtime_checkable
class TelemetrySource(Protocol):
    """Anything that contributes named columns to the controller CSV."""

    TELEMETRY_PREFIX: ClassVar[str]

    @classmethod
    def telemetry_fields(cls) -> Sequence[str]:
        """Column names WITHOUT the prefix. Must be constant for a run."""

    def telemetry(self) -> Mapping[str, object]:
        """Current values, keyed by the same unprefixed names."""


def prefixed(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def collect_fields(sources: Iterable[TelemetrySource]) -> list[str]:
    """Assemble the full, prefixed column list and reject collisions."""
    fields: list[str] = []
    seen: dict[str, str] = {}
    for source in sources:
        prefix = getattr(source, "TELEMETRY_PREFIX", "")
        owner = type(source).__name__
        for name in source.telemetry_fields():
            column = prefixed(prefix, name)
            if column in seen:
                raise TelemetrySchemaError(
                    f"Column {column!r} claimed by both {seen[column]} and "
                    f"{owner}. Give one of them a distinct TELEMETRY_PREFIX."
                )
            seen[column] = owner
            fields.append(column)
    return fields


def snapshot(source: TelemetrySource, *, strict: bool = True) -> dict[str, object]:
    """Pull one prefixed row fragment from a source.

    With ``strict``, a key the source never declared raises instead of being
    dropped on the floor -- that silent drop is exactly how columns used to go
    missing between a mission edit and the next log analysis.
    """
    prefix = getattr(source, "TELEMETRY_PREFIX", "")
    declared = set(source.telemetry_fields())
    values = source.telemetry()
    if strict:
        undeclared = set(values) - declared
        if undeclared:
            raise TelemetrySchemaError(
                f"{type(source).__name__}.telemetry() emitted undeclared keys "
                f"{sorted(undeclared)}. Add them to telemetry_fields()."
            )
    return {prefixed(prefix, k): v for k, v in values.items() if k in declared}


def schema_fingerprint(fields: Sequence[str]) -> str:
    """Short stable hash of a column list, for the schema-version cell."""
    digest = hashlib.blake2b("\x1f".join(fields).encode("utf-8"), digest_size=4)
    return digest.hexdigest()
