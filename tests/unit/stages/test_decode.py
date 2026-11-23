from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from raw_pipeline.config import parse_config
from raw_pipeline.core.artifacts import BEFORE_IMAGE, DEFECT_CROPS
from raw_pipeline.core.context import ImageContext
from raw_pipeline.core.errors import DefectCorrectionError
from raw_pipeline.core.protocols import Stage
from raw_pipeline.core.raw import (
    CFAFrame,
    ColumnDefect,
    DefectCorrectionResult,
    DefectMap,
    DevelopedImage,
    PixelDefect,
    RowDefect,
)
from raw_pipeline.core.registries import defect_corrector_registry, stage_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import ExifInfo
from raw_pipeline.io.rawpy_decoder import RawpyDecoder, sensor_to_image
from raw_pipeline.stages.decode import DecodeStage, crop_window, defect_center, downscale
from tests.contract.stage_contract import StageContract
from tests.support.synthetic import cfa_frame, decoded_context, inject_hot

HEIGHT, WIDTH = 64, 96
EXIF = ExifInfo(camera_make="Canon", camera_model="Canon EOS 800D", iso=100)
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SAMPLES = sorted(FIXTURES.glob("*.CR2"))


class FakeHandle:
    """Develops a CFA by replicating it into RGB (normalised); counts developments."""

    def __init__(self, frame: CFAFrame, flip: int = 0) -> None:
        self.frame = frame
        self.flip = flip
        self.developed: list[CFAFrame] = []

    def read_cfa(self) -> CFAFrame:
        return self.frame.with_data(self.frame.data.copy())

    def develop(self, cfa: CFAFrame) -> DevelopedImage:
        self.developed.append(cfa)
        mono = cfa.data.astype(np.float32) / np.float32(cfa.white_level)
        rgb = np.repeat(mono[:, :, None], 3, axis=2)
        if self.flip == 6:
            rgb = np.rot90(rgb, -1)
        return DevelopedImage(image=np.ascontiguousarray(rgb, dtype=np.float32), params={"decoder": "fake"})

    def sensor_to_image(self, x: int, y: int) -> tuple[int, int]:
        return sensor_to_image(x, y, width=self.frame.width, height=self.frame.height, flip=self.flip)


class FakeDecoder:
    def __init__(self, frame: CFAFrame | None = None, flip: int = 0) -> None:
        self.frame = frame or inject_hot(cfa_frame(HEIGHT, WIDTH), _column_mask(10))
        self.flip = flip
        self.handles: list[FakeHandle] = []
        self.opened: list[Path] = []

    @contextmanager
    def open(self, path: Path) -> Iterator[FakeHandle]:
        self.opened.append(path)
        handle = FakeHandle(self.frame, self.flip)
        self.handles.append(handle)
        yield handle


class NullCorrector:
    name: ClassVar[str] = "none"

    def correct(self, frame: CFAFrame, defects: DefectMap) -> DefectCorrectionResult:  # noqa: ARG002
        return DefectCorrectionResult(frame=frame, corrected_pixel_count=0, method=self.name)


class LeftNeighbourCorrector:
    """Replaces masked photosites with the same-colour neighbour two columns to the left."""

    name: ClassVar[str] = "fake"

    def correct(self, frame: CFAFrame, defects: DefectMap) -> DefectCorrectionResult:
        mask = defects.mask(frame.height, frame.width)
        data = frame.data.copy()
        data[mask] = np.roll(frame.data, 2, axis=1)[mask]
        return DefectCorrectionResult(
            frame=frame.with_data(data), corrected_pixel_count=int(mask.sum()), method=self.name
        )


class FakeMetadata:
    def read(self, path: Path) -> ExifInfo:  # noqa: ARG002
        return EXIF


def _column_mask(x: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, x] = True
    return mask


def make_stage(
    decoder: FakeDecoder | None = None,
    *,
    corrector: NullCorrector | LeftNeighbourCorrector | None = None,
    defects: DefectMap | None = None,
    report: bool = True,
    crop_px: int = 16,
) -> DecodeStage:
    return DecodeStage(
        decoder=decoder or FakeDecoder(),
        corrector=corrector or LeftNeighbourCorrector(),
        defect_map=DefectMap((ColumnDefect(10),)) if defects is None else defects,
        metadata_reader=FakeMetadata(),
        before_preview_long_edge_px=48,
        defect_crop_px=crop_px,
        produce_report_artifacts=report,
    )


class TestDecodeStageContract(StageContract):
    def make_stage(self) -> Stage:
        return make_stage()


class TestNullDecodeStageContract(StageContract):
    def make_stage(self) -> Stage:
        return make_stage(corrector=NullCorrector(), defects=DefectMap(()))


def test_corrects_before_demosaicing_and_records() -> None:
    decoder = FakeDecoder()
    out = make_stage(decoder).process(decoded_context())
    (handle,) = decoder.handles
    corrected, uncorrected = handle.developed
    assert not np.array_equal(corrected.data[:, 10], decoder.frame.data[:, 10])
    np.testing.assert_array_equal(uncorrected.data, decoder.frame.data)
    np.testing.assert_array_equal(out.image[..., 0], corrected.data / np.float32(corrected.white_level))
    records = out.metadata.records
    assert records["decode"]["decoder"] == "fake"
    assert records["decode"]["before_preview"] is True
    assert dict(records["defects"]) == {"method": "fake", "corrected_pixel_count": HEIGHT, "count": 1}
    assert out.metadata.exif == EXIF


def test_before_image_is_the_uncorrected_downscaled_develop() -> None:
    decoder = FakeDecoder()
    out = make_stage(decoder).process(decoded_context())
    before = out.artifact(BEFORE_IMAGE)
    assert before.shape == (32, 48, 3)
    assert before.dtype == np.float32
    expected = downscale(decoder.handles[0].developed[1].data[:, :, None].repeat(3, 2).astype(np.float32) / 15000, 48)
    np.testing.assert_allclose(before, expected, rtol=1e-6)


def test_defect_crops_cover_the_first_defect() -> None:
    out = make_stage(defects=DefectMap((ColumnDefect(10), PixelDefect(80, 5)))).process(decoded_context())
    crops = out.artifact(DEFECT_CROPS)
    assert crops.sensor_xy == (10, 31)
    assert crops.image_xy == (10, 31)
    assert crops.before.shape == crops.after.shape == (16, 16, 3)
    assert crops.before.max() == pytest.approx(1.0)
    assert crops.after.max() < 0.9


def test_defect_crops_follow_the_rotation() -> None:
    out = make_stage(FakeDecoder(flip=6), defects=DefectMap((PixelDefect(10, 3),))).process(decoded_context())
    crops = out.artifact(DEFECT_CROPS)
    assert crops.image_xy == (HEIGHT - 1 - 3, 10)
    assert out.image.shape == (WIDTH, HEIGHT, 3)
    assert crops.before.shape == (16, 16, 3)


def test_null_corrector_develops_once_and_reuses_the_image() -> None:
    decoder = FakeDecoder()
    out = make_stage(decoder, corrector=NullCorrector()).process(decoded_context())
    assert len(decoder.handles[0].developed) == 1
    assert out.metadata.records["defects"]["method"] == "none"
    assert out.metadata.records["defects"]["corrected_pixel_count"] == 0
    np.testing.assert_allclose(out.artifact(BEFORE_IMAGE), downscale(out.image, 48))
    assert out.artifact(DEFECT_CROPS) is None


def test_without_report_stage_skips_report_artifacts() -> None:
    decoder = FakeDecoder()
    out = make_stage(decoder, report=False).process(decoded_context())
    assert len(decoder.handles[0].developed) == 1
    assert out.artifact(BEFORE_IMAGE) is None
    assert out.artifact(DEFECT_CROPS) is None
    assert out.metadata.records["decode"]["before_preview"] is False


def test_defect_outside_the_sensor_fails_the_image() -> None:
    with pytest.raises(DefectCorrectionError, match=r"IMG_0001\.CR2"):
        make_stage(defects=DefectMap((ColumnDefect(500),))).process(decoded_context())


def test_input_frame_is_not_mutated() -> None:
    decoder = FakeDecoder()
    original = decoder.frame.data.copy()
    make_stage(decoder).process(decoded_context())
    np.testing.assert_array_equal(decoder.frame.data, original)


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        (PixelDefect(3, 4), (3, 4)),
        (ColumnDefect(10, width=3), (11, 31)),
        (ColumnDefect(10, start=20, end=30), (10, 24)),
        (RowDefect(7, start=0, end=500), (47, 7)),
    ],
)
def test_defect_center(defect: PixelDefect | ColumnDefect | RowDefect, expected: tuple[int, int]) -> None:
    assert defect_center(defect, HEIGHT, WIDTH) == expected


def test_crop_window_stays_inside() -> None:
    assert crop_window(2, 16, 100) == slice(0, 16)
    assert crop_window(98, 16, 100) == slice(84, 100)
    assert crop_window(50, 16, 100) == slice(42, 58)
    assert crop_window(5, 16, 10) == slice(0, 10)


def test_downscale_never_upscales() -> None:
    img = np.ones((10, 20, 3), dtype=np.float32) * 2.0
    same = downscale(img, 100)
    assert same.shape == img.shape
    assert not np.shares_memory(same, img)
    small = downscale(img, 10)
    assert small.shape == (5, 10, 3)
    np.testing.assert_allclose(small, 2.0)


def test_registered_and_built_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(defect_corrector_registry._factories, "fake", lambda _env: LeftNeighbourCorrector())
    cfg = parse_config(
        {
            "defects": {"method": "fake", "map": [{"type": "column", "index": 10}]},
            "decode": {"before_preview_long_edge_px": 256},
            "report": {"defect_crop_px": 64},
        }
    )
    stage = stage_registry.create("decode", BuildEnv(config=cfg, output_dir=Path("out")))
    assert isinstance(stage, DecodeStage)
    assert stage.correction_enabled
    assert isinstance(stage._decoder, RawpyDecoder)
    assert stage._crop_px == 64
    assert stage._before_long_edge_px == 256


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLES, reason="no tests/fixtures/*.CR2")
def test_real_cr2_input_is_unchanged() -> None:
    path = SAMPLES[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stage = DecodeStage(
        decoder=RawpyDecoder(),
        corrector=LeftNeighbourCorrector(),
        defect_map=DefectMap((ColumnDefect(1001),)),
        metadata_reader=FakeMetadata(),
    )
    out = stage.process(ImageContext.initial(path))
    assert out.image.dtype == np.float32
    assert max(out.artifact(BEFORE_IMAGE).shape[:2]) == 1024
    assert out.artifact(DEFECT_CROPS).before.shape == (192, 192, 3)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
