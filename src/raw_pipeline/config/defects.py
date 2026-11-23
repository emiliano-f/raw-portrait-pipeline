"""Conversion between the ``defects.map`` config entries and core defect objects."""

from __future__ import annotations

from collections.abc import Iterable

from raw_pipeline.config.models import DefectEntry
from raw_pipeline.core.raw import ColumnDefect, Defect, DefectMap, PixelDefect, RowDefect


def to_defect(entry: DefectEntry) -> Defect:
    """Convert one validated config entry to a core defect."""
    if entry.type == "pixel":
        assert entry.x is not None  # guaranteed by DefectEntry validation
        assert entry.y is not None
        return PixelDefect(x=entry.x, y=entry.y)
    assert entry.index is not None  # guaranteed by DefectEntry validation
    cls = ColumnDefect if entry.type == "column" else RowDefect
    return cls(index=entry.index, width=entry.width, start=entry.start, end=entry.end)


def to_defect_map(entries: Iterable[DefectEntry]) -> DefectMap:
    """Convert config entries to a :class:`DefectMap`."""
    return DefectMap(tuple(to_defect(e) for e in entries))


def to_entry(defect: Defect) -> DefectEntry:
    """Convert a core defect back to a config entry (for ``detect-defects`` output)."""
    if isinstance(defect, PixelDefect):
        return DefectEntry(type="pixel", x=defect.x, y=defect.y)
    kind = "column" if isinstance(defect, ColumnDefect) else "row"
    return DefectEntry(type=kind, index=defect.index, width=defect.width, start=defect.start, end=defect.end)


def entry_to_yaml_dict(entry: DefectEntry) -> dict[str, object]:
    """Compact mapping for YAML output: drops unset optional fields."""
    data = entry.model_dump(exclude_none=True)
    if entry.type == "pixel":
        data.pop("width", None)
    return data
