"""Pydantic models of the YAML configuration.

Every field documents its unit and default; the configuration reference in
``docs/functional/configuration.md`` is generated from these descriptions.
Component *names* (stages, strategies, exporters, ...) are checked against
the registries by :func:`raw_pipeline.config.validation.validate_components`.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineConfig(_Section):
    """Which stages run, in which order, and how many images in parallel."""

    stages: tuple[str, ...] = Field(
        default=("decode", "meter", "tone", "retouch", "export", "report", "qa"),
        description="Registered stage names, executed in this order for every image.",
    )
    workers: int | None = Field(
        default=None,
        ge=1,
        description="Parallel worker processes. Default: CPU count - 1 (at least 1). Overridden by --workers.",
    )


class DecodeConfig(_Section):
    """RAW development with LibRaw (rawpy)."""

    decoder: str = Field(default="rawpy", description="Registered RAW decoder name.")
    demosaic: Literal["ahd", "aahd", "dcb", "dht", "linear", "vng", "ppg"] = Field(
        default="ahd", description="LibRaw demosaic algorithm."
    )
    highlight_mode: Literal["clip", "ignore", "blend", "reconstruct"] = Field(
        default="blend",
        description="LibRaw treatment of clipped photosites before tone mapping. "
        "'blend' keeps highlight detail for the tone stage to roll off.",
    )
    before_preview_long_edge_px: int = Field(
        default=1024,
        ge=64,
        description="Long edge in pixels of the uncorrected 'before' preview kept for the contact sheet.",
    )


class DefectEntry(_Section):
    """One sensor defect in raw sensor coordinates (visible area, before rotation)."""

    type: Literal["column", "row", "pixel"] = Field(description="Defect kind.")
    index: int | None = Field(default=None, ge=0, description="Column (x) or row (y) index; column/row only.")
    width: int = Field(default=1, ge=1, description="Adjacent defective columns/rows starting at index (pixels).")
    start: int | None = Field(
        default=None, ge=0, description="First affected row (column defect) or column (row defect), inclusive."
    )
    end: int | None = Field(
        default=None, ge=0, description="Last affected row (column defect) or column (row defect), exclusive."
    )
    x: int | None = Field(default=None, ge=0, description="Column of a pixel defect.")
    y: int | None = Field(default=None, ge=0, description="Row of a pixel defect.")

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.type == "pixel":
            if self.x is None or self.y is None:
                msg = "pixel defects need x and y"
                raise ValueError(msg)
        elif self.index is None:
            msg = f"{self.type} defects need index"
            raise ValueError(msg)
        if self.end is not None and self.end <= (self.start or 0):
            msg = "end must be greater than start (start defaults to 0)"
            raise ValueError(msg)
        return self


class DefectDetectionConfig(_Section):
    """Parameters of ``raw-pipeline detect-defects``."""

    outlier_threshold_sigma: float = Field(
        default=8.0,
        gt=0,
        description="Minimum deviation, in robust standard deviations (MAD), of a row/column/pixel "
        "from its same-colour neighbours to count as an outlier in one frame.",
    )
    min_frame_fraction: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Fraction of analysed frames in which a candidate must be an outlier (batch mode).",
    )
    max_pixel_defects: int = Field(default=500, ge=0, description="Upper bound on individual pixel defects reported.")


class DefectsConfig(_Section):
    """Raw-level hot/stuck pixel correction applied before demosaicing."""

    method: str = Field(
        default="none",
        description="Registered defect corrector: none | neighbor_average | directional.",
    )
    map: tuple[DefectEntry, ...] = Field(
        default=(), description="Known defects in raw sensor coordinates (see detect-defects)."
    )
    detection: DefectDetectionConfig = Field(default_factory=DefectDetectionConfig)


class MeterConfig(_Section):
    """Per-image exposure metering on the face."""

    strategy: str = Field(default="face_weighted", description="Registered metering strategy.")
    detector: str = Field(default="mediapipe", description="Registered face detector.")
    fallback_detector: str | None = Field(
        default="opencv", description="Detector used when the primary one is unavailable or fails."
    )
    fallback_strategy: str = Field(default="center_weighted", description="Strategy used when no face is found.")
    target_face_luminance: float = Field(
        default=0.18, gt=0, description="Target scene-linear luminance of the face (Rec.709 weights)."
    )
    target_center_luminance: float = Field(
        default=0.18, gt=0, description="Target scene-linear luminance for center-weighted metering."
    )
    max_correction_ev: float = Field(
        default=2.0, ge=0, description="Absolute limit of the exposure correction, in EV stops."
    )
    detection_long_edge_px: int = Field(
        default=1024, ge=128, description="Long edge in pixels of the preview used for face detection."
    )
    min_face_confidence: float = Field(default=0.6, ge=0, le=1, description="Minimum detector score.")
    center_weight_sigma_fraction: float = Field(
        default=0.25,
        gt=0,
        description="Gaussian sigma of center-weighted metering, as a fraction of the image long edge.",
    )


class ToneConfig(_Section):
    """Exposure correction, highlight recovery and shadow compression (scene-linear)."""

    highlight_rolloff_start_ev: float = Field(
        default=-1.0,
        lt=0,
        description="Where the highlight shoulder starts, in EV relative to 1.0 (diffuse white); must be below 0.",
    )
    highlight_rolloff_ev: float = Field(
        default=1.5, gt=0, description="Scene range above the shoulder compressed into the output range, in EV."
    )
    shadow_lift_ev: float = Field(
        default=0.3, ge=0, le=2.0, description="Maximum lift applied to deep shadows, in EV (0 to 2)."
    )
    shadow_pivot_luminance: float = Field(
        default=0.05, gt=0, description="Scene-linear luminance below which shadows are lifted."
    )


class RetouchConfig(_Section):
    """Optional subtle skin smoothing inside a face-mesh skin mask."""

    method: str = Field(
        default="frequency_separation", description="Registered retoucher: none | frequency_separation."
    )
    strength: float = Field(
        default=0.25, ge=0, le=1, description="Blend of the smoothed low-frequency layer (0 = off, 1 = full)."
    )
    blur_radius_face_fraction: float = Field(
        default=0.015, gt=0, description="Low-pass radius as a fraction of the face box width."
    )
    mask_feather_px: int = Field(default=8, ge=0, description="Feather of the skin mask edge, in pixels.")


class ExportConfig(_Section):
    """Output encoding."""

    formats: tuple[str, ...] = Field(default=("jpeg",), description="Registered exporters to run (jpeg, tiff).")
    jpeg_quality: int = Field(default=92, ge=1, le=100, description="JPEG quality (1-100).")
    long_edge_px: int | None = Field(
        default=None, ge=64, description="Resize so the long edge is this many pixels; null keeps full size."
    )
    color_space: Literal["srgb"] = Field(default="srgb", description="Output colour space (ICC embedded).")
    keep_gps: bool = Field(default=False, description="Copy GPS EXIF tags to outputs.")
    filename_suffix: str = Field(default="", description="Appended to the input stem for output files.")


class ReportConfig(_Section):
    """Contact sheet and CSV report."""

    reporters: tuple[str, ...] = Field(default=("csv", "contact_sheet"), description="Registered batch reporters.")
    thumbnail_long_edge_px: int = Field(default=512, ge=64, description="Contact sheet thumbnail long edge (px).")
    defect_crop_px: int = Field(default=192, ge=16, description="Edge of the 100 % defect crop (px).")
    csv_filename: str = Field(default="report.csv", description="CSV file name inside the output directory.")
    contact_sheet_filename: str = Field(
        default="contact_sheet.jpg", description="Contact sheet file name inside the output directory."
    )


class QAConfig(_Section):
    """Optional vision-LLM QA. Runs only when ``provider`` is not ``none`` and an API key is set."""

    provider: str = Field(default="none", description="Registered QA provider; 'none' disables QA.")
    model: str | None = Field(default=None, description="Provider model identifier.")
    endpoint: str | None = Field(default=None, description="API base URL (provider default when null).")
    api_key_env: str = Field(default="QA_API_KEY", description="Environment variable holding the API key.")
    preview_long_edge_px: int = Field(
        default=1024, ge=64, le=1024, description="Long edge of the JPEG preview sent (px, at most 1024)."
    )
    preview_jpeg_quality: int = Field(default=85, ge=1, le=100, description="JPEG quality of the preview.")
    timeout_s: float = Field(default=30.0, gt=0, description="Request timeout in seconds.")


class AppConfig(_Section):
    """Root of the configuration file."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    defects: DefectsConfig = Field(default_factory=DefectsConfig)
    meter: MeterConfig = Field(default_factory=MeterConfig)
    tone: ToneConfig = Field(default_factory=ToneConfig)
    retouch: RetouchConfig = Field(default_factory=RetouchConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    plugins: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Free-form sections for third-party plugins, keyed by plugin name."
    )
