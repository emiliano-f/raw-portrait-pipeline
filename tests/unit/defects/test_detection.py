from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import DefectDetectionConfig
from raw_pipeline.core.raw import CFAFrame, ColumnDefect, DefectMap, PixelDefect, RowDefect
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.defects import DirectionalCorrector, StatisticalDefectDetector
from tests.support.synthetic import cfa_frame, inject_hot

SIZE = 256
HOT = 3000  # raw units; the dark level is ~2048 with 20 units of noise
INJECTED = (
    ColumnDefect(index=41),
    ColumnDefect(index=100, width=2),
    ColumnDefect(index=200, start=64, end=192),
    RowDefect(index=150, width=3),
    RowDefect(index=77, end=128),
    PixelDefect(x=13, y=9),
    PixelDefect(x=180, y=230),
)


def _dark(seed: int = 0, defects: tuple = INJECTED) -> CFAFrame:
    frame = cfa_frame(SIZE, SIZE, level=0, seed=seed)
    return inject_hot(frame, DefectMap(defects).mask(SIZE, SIZE), HOT) if defects else frame


def _found(detector: StatisticalDefectDetector, frames: Iterator[CFAFrame] | list[CFAFrame]) -> set:
    return {d.defect for d in detector.detect(frames)}


def test_dark_frame_finds_every_injected_defect_exactly() -> None:
    detected = StatisticalDefectDetector().detect([_dark()])
    assert {d.defect for d in detected} == set(INJECTED)
    assert all(d.score > 8 and d.frames_flagged == d.frames_total == 1 for d in detected)
    kinds = [type(d.defect) for d in detected]
    assert kinds == sorted(kinds, key=[ColumnDefect, RowDefect, PixelDefect].index)


def test_stuck_at_white_level_line_is_found() -> None:
    frame = inject_hot(cfa_frame(SIZE, SIZE, level=0), DefectMap((ColumnDefect(index=7),)).mask(SIZE, SIZE))
    assert _found(StatisticalDefectDetector(), [frame]) == {ColumnDefect(index=7)}


@pytest.mark.parametrize("level", [0, 4000])
def test_no_false_positives_on_clean_noise(level: int) -> None:
    frames = [cfa_frame(SIZE, SIZE, level=level, seed=s) for s in range(3)]
    assert StatisticalDefectDetector().detect(frames) == []


def test_batch_mode_rejects_scene_edges_and_keeps_consistent_defects() -> None:
    defects = DefectMap(INJECTED[:2] + INJECTED[-1:])
    frames = (
        inject_hot(cfa_frame(SIZE, SIZE, level=3000, seed=s, edge_at_x=60 + 30 * s), defects.mask(SIZE, SIZE), 9000)
        for s in range(5)
    )
    assert _found(StatisticalDefectDetector(min_frame_fraction=0.8), frames) == set(defects)


def test_inconsistent_defect_below_frame_fraction_is_dropped() -> None:
    frames = [_dark(0), _dark(1, ()), _dark(2, ())]
    assert _found(StatisticalDefectDetector(min_frame_fraction=0.5), frames) == set()
    assert _found(StatisticalDefectDetector(min_frame_fraction=0.3), frames) == set(INJECTED)


def test_pixel_defects_are_capped_to_the_strongest() -> None:
    pixels = tuple(PixelDefect(x=20 + 10 * i, y=30) for i in range(6))
    frame = cfa_frame(SIZE, SIZE, level=0)
    data = frame.data.copy()
    for i, p in enumerate(pixels):
        data[p.y, p.x] += 500 + 200 * i
    detected = StatisticalDefectDetector(max_pixel_defects=2).detect([frame.with_data(data)])
    assert [d.defect for d in detected] == list(pixels[-2:])
    assert StatisticalDefectDetector(max_pixel_defects=0).detect([frame.with_data(data)]) == []


def test_dead_pixels_are_found_too() -> None:
    frame = cfa_frame(SIZE, SIZE, level=4000)
    data = frame.data.copy()
    data[50, 60] = 0
    assert _found(StatisticalDefectDetector(), [frame.with_data(data)]) == {PixelDefect(x=60, y=50)}


def test_frames_are_streamed_from_a_generator() -> None:
    consumed: list[int] = []

    def frames() -> Iterator[CFAFrame]:
        for seed in range(3):
            consumed.append(seed)
            yield _dark(seed)

    assert _found(StatisticalDefectDetector(), frames()) == set(INJECTED)
    assert consumed == [0, 1, 2]


def test_empty_input_finds_nothing() -> None:
    assert StatisticalDefectDetector().detect([]) == []


def test_mixed_frame_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same size"):
        StatisticalDefectDetector().detect([cfa_frame(64, 64), cfa_frame(64, 96)])


def test_detected_map_drives_the_corrector() -> None:
    clean = cfa_frame(SIZE, SIZE, level=0, seed=5)
    hot = inject_hot(clean, DefectMap(INJECTED).mask(SIZE, SIZE), HOT)
    found = DefectMap(tuple(d.defect for d in StatisticalDefectDetector().detect([hot])))
    corrected = DirectionalCorrector().correct(hot, found).frame
    error = np.abs(corrected.data.astype(np.int64) - clean.data.astype(np.int64))
    assert error.max() < 120


def test_built_from_config(tmp_path: Path) -> None:
    detection = DefectDetectionConfig(outlier_threshold_sigma=5, min_frame_fraction=0.5, max_pixel_defects=7)
    config = AppConfig(defects={"detection": detection})
    detector = StatisticalDefectDetector.from_env(BuildEnv(config=config, output_dir=tmp_path))
    assert (detector.outlier_threshold_sigma, detector.min_frame_fraction, detector.max_pixel_defects) == (5, 0.5, 7)


def test_exact_frame_fraction_is_accepted() -> None:
    frames = [_dark(s, INJECTED if s < 14 else ()) for s in range(25)]
    assert _found(StatisticalDefectDetector(min_frame_fraction=0.56), frames) == set(INJECTED)


def test_staggered_adjacent_lines_keep_their_own_extents() -> None:
    defects = (ColumnDefect(index=60), ColumnDefect(index=61, end=128))
    assert _found(StatisticalDefectDetector(), [_dark(defects=defects)]) == set(defects)


@pytest.mark.parametrize("defect", [ColumnDefect(index=0, width=4), RowDefect(index=SIZE - 4, width=4)])
def test_band_at_the_sensor_border_is_not_widened(defect: ColumnDefect | RowDefect) -> None:
    assert _found(StatisticalDefectDetector(), [_dark(defects=(defect,))]) == {defect}


def test_stationary_scene_edge_is_not_a_defect() -> None:
    frames = [cfa_frame(SIZE, SIZE, level=3000, seed=s, edge_at_x=60) for s in range(5)]
    assert StatisticalDefectDetector().detect(frames) == []


def test_six_photosite_band_is_found_whole() -> None:
    band = ColumnDefect(index=100, width=6)
    assert _found(StatisticalDefectDetector(), [_dark(defects=(band,))]) == {band}


def test_stationary_scene_edge_near_the_border_is_not_a_defect() -> None:
    frames = [cfa_frame(SIZE, SIZE, level=3000, seed=s, edge_at_x=6, edge_level=3600) for s in range(5)]
    assert StatisticalDefectDetector().detect(frames) == []


def test_unevenly_hot_wide_band_is_found_whole() -> None:
    frame = cfa_frame(SIZE, SIZE, level=0)
    data = frame.data.copy()
    for offset, value in enumerate((12000, 12000, 9000, 9000, 6000, 6000)):
        data[:, 100 + offset] = value
    assert _found(StatisticalDefectDetector(), [frame.with_data(data)]) == {ColumnDefect(index=100, width=6)}


def test_stationary_scene_edge_beside_a_wide_band_is_not_flagged() -> None:
    band = ColumnDefect(index=100, width=6)
    frames = [
        inject_hot(
            cfa_frame(SIZE, SIZE, level=3600, seed=s, edge_at_x=108, edge_level=3000),
            DefectMap((band,)).mask(SIZE, SIZE),
            9000,
        )
        for s in range(5)
    ]
    assert _found(StatisticalDefectDetector(), frames) == {band}


def test_stationary_stripe_beside_a_wide_band_is_not_flagged() -> None:
    band = ColumnDefect(index=100, width=6)
    frames = []
    for s in range(5):
        frame = cfa_frame(SIZE, SIZE, level=3000, seed=s)
        data = frame.data.copy()
        data[:, 106:112] += 600
        frames.append(inject_hot(frame.with_data(data), DefectMap((band,)).mask(SIZE, SIZE), 9000))
    assert _found(StatisticalDefectDetector(), frames) == {band}
