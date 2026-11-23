from __future__ import annotations

import csv
from pathlib import Path

import pytest

from raw_pipeline.core.errors import OutputExistsError
from raw_pipeline.core.registries import reporter_registry
from raw_pipeline.exporters.reporters.csv_report import CsvReporter, format_value
from tests.unit.exporters.helpers import build_env
from tests.unit.exporters.reporters.fakes import failed_outcome, ok_outcome


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_round_trip_every_parameter(tmp_path: Path) -> None:
    reporter = CsvReporter("report.csv", overwrite=False)
    outcome = ok_outcome("IMG_0002.CR2", ev=0.1 + 0.2)
    reporter.add(outcome)
    [path] = reporter.finalize(tmp_path)
    _, [row] = _read(path)
    expected = {key: format_value(value) for key, value in outcome.metadata.flatten().items()}
    assert {key: row[key] for key in expected} == expected
    assert float(row["meter.correction_ev"]) == 0.1 + 0.2
    assert row["exif.has_gps"] == "true"
    assert row["qa.skipped"] == "true"
    assert row["qa.flags"] == ""
    assert (row["file"], row["status"], row["outputs"]) == ("IMG_0002.CR2", "ok", "IMG_0002.jpg")


def test_failures_included_and_columns_stable(tmp_path: Path) -> None:
    reporter = CsvReporter("r.csv", overwrite=False)
    for outcome in (ok_outcome("IMG_0003.CR2"), failed_outcome("IMG_0001.CR2"), ok_outcome("IMG_0002.CR2")):
        reporter.add(outcome)
    columns, rows = _read(reporter.finalize(tmp_path)[0])
    assert [r["file"] for r in rows] == ["IMG_0001.CR2", "IMG_0002.CR2", "IMG_0003.CR2"]
    assert (rows[0]["status"], rows[0]["error_type"], rows[0]["error"]) == ("failed", "DecodeError", "truncated file")
    assert rows[0]["meter.correction_ev"] == ""
    assert columns[:5] == ["file", "status", "error_type", "error", "outputs"]
    stages = [c.split(".")[0] for c in columns[5:]]
    assert list(dict.fromkeys(stages)) == ["exif", "decode", "defects", "meter", "qa"]
    assert columns[5:] == [
        c for s in dict.fromkeys(stages) for c in sorted(c for c in columns[5:] if c.startswith(s + "."))
    ]


def test_order_of_add_does_not_change_bytes() -> None:
    outcomes = [ok_outcome("B.CR2"), failed_outcome("A.CR2"), ok_outcome("C.CR2", ev=-1.25)]
    a, b = CsvReporter("r.csv", overwrite=False), CsvReporter("r.csv", overwrite=False)
    for outcome in outcomes:
        a.add(outcome)
    for outcome in reversed(outcomes):
        b.add(outcome)
    assert a.render() == b.render()


def test_no_silent_overwrite(tmp_path: Path) -> None:
    CsvReporter("r.csv", overwrite=False).finalize(tmp_path)
    with pytest.raises(OutputExistsError):
        CsvReporter("r.csv", overwrite=False).finalize(tmp_path)
    CsvReporter("r.csv", overwrite=True).finalize(tmp_path)


def test_built_from_config(tmp_path: Path) -> None:
    reporter = reporter_registry.create("csv", build_env(tmp_path, overwrite=True, report={"csv_filename": "x.csv"}))
    assert isinstance(reporter, CsvReporter)
    assert (reporter.filename, reporter.overwrite) == ("x.csv", True)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, ""),
        (True, "true"),
        (False, "false"),
        (3, "3"),
        (-1.5, "-1.5"),
        (0.1, "0.1"),
        ("a,b", "a,b"),
        ("=1+1", "'=1+1"),
        ("-x", "'-x"),
    ],
)
def test_format_value(value, text) -> None:
    assert format_value(value) == text


def test_formula_like_text_is_neutralised(tmp_path: Path) -> None:
    reporter = CsvReporter("r.csv", overwrite=False)
    reporter.add(failed_outcome("=HYPERLINK(1).CR2"))
    reporter.add(ok_outcome("IMG_0001.CR2", ev=-1.25))
    _, rows = _read(reporter.finalize(tmp_path)[0])
    assert rows[0]["file"] == "'=HYPERLINK(1).CR2"
    assert rows[1]["meter.correction_ev"] == "-1.25"
