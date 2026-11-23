from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import yaml

from raw_pipeline import cli
from raw_pipeline.config.models import AppConfig
from raw_pipeline.runtime.components import ComponentRegistries
from tests.unit.runtime.fakes import EmptyDetector, FakeDecoder, FakeServices, make_registries, write_inputs


class Run:
    def __init__(self, registries: ComponentRegistries, services: FakeServices) -> None:
        self.registries = registries
        self.services = services
        self.answers: list[str] = []
        self.questions: list[str] = []
        self.out = io.StringIO()
        self.err = io.StringIO()

    def ask(self, question: str) -> str:
        self.questions.append(question)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def __call__(self, *argv: object) -> int:
        self.out.seek(0)
        self.out.truncate()
        self.err.seek(0)
        self.err.truncate()
        return cli.main(
            [str(a) for a in argv],
            registries=self.registries,
            services=self.services.services(),
            stdout=self.out,
            stderr=self.err,
            ask=self.ask,
        )


@pytest.fixture
def run(registries: ComponentRegistries, services: FakeServices) -> Run:
    return Run(registries, services)


def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_process_writes_outputs_and_exits_zero(run: Run, input_dir: Path, config_path: Path, tmp_path: Path) -> None:
    before = _digests(input_dir)
    out = tmp_path / "out"

    assert run("process", input_dir, out, "--config", config_path, "--workers", 1) == 0

    names = ["IMG_0001.txt", "IMG_0002.txt", "IMG_0003.txt", "summary.txt"]
    assert sorted(p.name for p in out.iterdir()) == names
    assert _digests(input_dir) == before
    progress = run.err.getvalue()
    assert "[1/3] IMG_0001.CR2: ok" in progress
    assert "3 ok, 0 failed" in progress


def test_process_exits_non_zero_when_an_image_fails(
    run: Run, input_dir: Path, config_path: Path, tmp_path: Path
) -> None:
    write_inputs(input_dir, {"IMG_0004.CR2": b"BAD"})
    assert run("process", input_dir, tmp_path / "out", "--config", config_path, "-q") == 1
    assert (tmp_path / "out" / "IMG_0001.txt").exists()
    assert "[" not in run.err.getvalue()


def test_process_refuses_to_overwrite_without_flag(
    run: Run, input_dir: Path, config_path: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    assert run("process", input_dir, out, "--config", config_path) == 0
    (out / "IMG_0001.txt").write_text("keep me", encoding="utf-8")

    assert run("process", input_dir, out, "--config", config_path) == 1
    assert (out / "IMG_0001.txt").read_text() == "keep me"

    assert run("process", input_dir, out, "--config", config_path, "--overwrite") == 0
    assert (out / "IMG_0001.txt").read_text() != "keep me"


def test_process_does_not_read_its_own_output_folder(run: Run, input_dir: Path, config_path: Path) -> None:
    out = input_dir / "out"
    assert run("process", input_dir, out, "--config", config_path) == 0
    write_inputs(out, {"IMG_9999.CR2": b"BAD"})
    assert run("process", input_dir, out, "--config", config_path, "--overwrite") == 0
    assert "3 ok, 0 failed" in run.err.getvalue()


@pytest.mark.parametrize(
    "case",
    ["missing_input", "same_dir", "output_is_file", "no_raws", "unknown_component", "bad_config"],
)
def test_process_usage_errors_exit_two_before_writing(
    run: Run, input_dir: Path, config_path: Path, tmp_path: Path, case: str
) -> None:
    out = tmp_path / "out"
    args: list[object] = ["process", input_dir, out, "--config", config_path]
    if case == "missing_input":
        args[1] = tmp_path / "nope"
    elif case == "same_dir":
        args[2] = input_dir
    elif case == "output_is_file":
        out.write_text("not a folder", encoding="utf-8")
    elif case == "no_raws":
        args[1] = tmp_path / "empty"
        (tmp_path / "empty").mkdir()
    elif case == "unknown_component":
        data = yaml.safe_load(config_path.read_text())
        data["pipeline"]["stages"].append("not_registered")
        config_path.write_text(yaml.safe_dump(data))
    else:
        config_path.write_text("pipeline: {workers: -3}\n")

    assert run(*args) == 2
    assert run.err.getvalue().startswith("error: ")
    assert not out.is_dir()


def test_invalid_workers_is_an_argparse_error(run: Run, input_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        run("process", input_dir, tmp_path / "out", "--workers", 0)
    assert info.value.code == 2


def test_detect_defects_prints_findings_and_merges_after_confirmation(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices
) -> None:
    before = _digests(input_dir)
    run.answers = ["y"]

    assert run("detect-defects", input_dir, "--config", config_path) == 0

    text = run.out.getvalue()
    assert "analysed 3 frame(s), 0 unreadable" in text
    assert "found 2 defect(s)" in text
    assert "{type: column, index: 21, width: 2}" in text
    assert "{type: pixel, x: 7, y: 9}" in text
    assert len(run.questions) == 1
    assert [(len(m), p) for m, p in services.merged] == [(2, config_path)]
    assert services.written == []
    assert _digests(input_dir) == before


@pytest.mark.parametrize("answers", [["n"], [""], []])
def test_detect_defects_declined_writes_nothing(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices, answers: list[str]
) -> None:
    run.answers = answers
    assert run("detect-defects", input_dir, "--config", config_path) == 0
    assert services.merged == []
    assert "not written" in run.out.getvalue()


def test_detect_defects_yes_skips_prompt(run: Run, input_dir: Path, config_path: Path, services: FakeServices) -> None:
    assert run("detect-defects", input_dir, "--config", config_path, "--yes") == 0
    assert run.questions == []
    assert len(services.merged) == 1


def test_detect_defects_output_writes_standalone_file(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices, tmp_path: Path
) -> None:
    target = tmp_path / "defects.yaml"
    assert run("detect-defects", input_dir, "--config", config_path, "--output", target, "--overwrite") == 0
    assert run.questions == []
    assert services.merged == []
    assert [(p, o) for _, p, o in services.written] == [(target, True)]


def test_detect_defects_dark_frame_uses_only_that_file(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices
) -> None:
    dark = input_dir / "IMG_0002.CR2"
    assert run("detect-defects", input_dir, "--config", config_path, "--dark-frame", dark, "--yes") == 0
    assert "analysed 1 frame(s)" in run.out.getvalue()
    assert services.detectors[0].frames == 1


def test_detect_defects_without_findings_writes_nothing(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices
) -> None:
    services.detector_cls = EmptyDetector
    assert run("detect-defects", input_dir, "--config", config_path) == 0
    assert "no defects found" in run.out.getvalue()
    assert run.questions == []
    assert services.merged == []


def test_detect_defects_needs_a_destination(input_dir: Path, services: FakeServices) -> None:
    run = Run(make_registries(AppConfig()), services)
    assert run("detect-defects", input_dir) == 2
    assert "--output" in run.err.getvalue()


def test_detect_defects_fails_when_no_frame_is_readable(run: Run, tmp_path: Path, config_path: Path) -> None:
    folder = tmp_path / "bad"
    write_inputs(folder, {"A.CR2": b"BAD"})
    assert run("detect-defects", folder, "--config", config_path) == 2


def test_detect_defects_output_must_be_yaml(
    run: Run, input_dir: Path, config_path: Path, services: FakeServices
) -> None:
    raw = input_dir / "IMG_0001.CR2"
    before = raw.read_bytes()
    assert run("detect-defects", input_dir, "--config", config_path, "--output", raw, "--overwrite") == 2
    assert services.written == []
    assert raw.read_bytes() == before


def test_detect_defects_only_needs_the_decoder(input_dir: Path, config_path: Path, services: FakeServices) -> None:
    regs = ComponentRegistries.empty()
    regs.raw_decoders.register(FakeDecoder.name)(FakeDecoder)
    run = Run(regs, services)
    assert run("detect-defects", input_dir, "--config", config_path, "--yes") == 0
    assert len(services.merged) == 1
    assert run("detect-defects", input_dir, "--yes") == 2
    assert "decode.decoder" in run.err.getvalue()
