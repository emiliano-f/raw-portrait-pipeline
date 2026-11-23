from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from raw_pipeline.config.models import AppConfig
from raw_pipeline.runtime.report_files import (
    duplicate_output_stems,
    report_filenames,
    report_name_collisions,
    stale_contact_sheet_pages,
)


def _config(**report: str) -> AppConfig:
    return AppConfig.model_validate({"report": report})


def test_report_filenames_lists_the_enabled_reporters_files() -> None:
    assert report_filenames(_config()) == ("report.csv", "contact_sheet.jpg")


def test_report_name_collisions_ignore_disabled_reporters() -> None:
    sources = [Path("report.CR2"), Path("contact_sheet.CR2"), Path("contact_sheet-2.CR2")]
    csv_only = AppConfig.model_validate({"report": {"reporters": ["csv"]}})
    assert report_filenames(csv_only) == ("report.csv",)
    assert report_name_collisions(sources, csv_only) == ["report.CR2 -> report.csv"]
    sheet_only = AppConfig.model_validate({"report": {"reporters": ["contact_sheet"]}})
    assert report_name_collisions(sources, sheet_only) == [
        "contact_sheet.CR2 -> contact_sheet.jpg",
        "contact_sheet-2.CR2 -> contact sheet page contact_sheet-2",
    ]
    assert report_name_collisions(sources, AppConfig.model_validate({"report": {"reporters": []}})) == []


@pytest.mark.parametrize(
    ("source", "clash"),
    [
        ("contact_sheet.CR2", "contact_sheet.CR2 -> contact_sheet.jpg"),
        ("REPORT.cr2", "REPORT.cr2 -> report.csv"),
        ("contact_sheet-3.CR2", "contact_sheet-3.CR2 -> contact sheet page contact_sheet-3"),
    ],
)
def test_report_name_collisions_finds_clashing_stems(source: str, clash: str) -> None:
    sources = [Path("in/IMG_0001.CR2"), Path("in") / source]
    assert report_name_collisions(sources, _config()) == [clash]


def test_report_name_collisions_follows_configured_names_and_suffix() -> None:
    config = AppConfig.model_validate({"report": {"csv_filename": "summary.csv"}, "export": {"filename_suffix": "_x"}})
    sources = [Path("summary.CR2"), Path("report.CR2"), Path("summary_x.CR2"), Path("contact_sheet-x.CR2")]
    assert report_name_collisions(sources, config) == []
    clashing = config.model_copy(update={"report": config.report.model_copy(update={"csv_filename": "summary_x.csv"})})
    assert report_name_collisions([Path("summary.CR2")], clashing) == ["summary.CR2 -> summary_x.csv"]


def test_stale_contact_sheet_pages_lists_pages_this_run_did_not_write(tmp_path: Path) -> None:
    for name in ("contact_sheet.jpg", "contact_sheet-2.jpg", "contact_sheet-3.jpg", "contact_sheet-x.jpg", "a.jpg"):
        (tmp_path / name).write_bytes(b"x")
    written = [tmp_path / "contact_sheet.jpg", tmp_path / "contact_sheet-2.jpg"]
    assert stale_contact_sheet_pages(tmp_path, _config(), written) == [tmp_path / "contact_sheet-3.jpg"]


def test_duplicate_output_stems_ignore_case() -> None:
    sources = [Path("in/IMG_1.CR2"), Path("in/img_1.cr2"), Path("in/IMG_2.CR2")]
    assert duplicate_output_stems(sources) == ["IMG_1.CR2 / img_1.cr2"]
    assert duplicate_output_stems(sources[1:]) == []


@pytest.mark.parametrize("name", ["../in/IMG_1.CR2", "sub/report.csv", "..", "", "a\\b.csv"])
def test_report_file_names_must_be_plain(name: str) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"report": {"csv_filename": name}})
