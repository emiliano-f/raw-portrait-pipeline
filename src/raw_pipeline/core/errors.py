"""Exception hierarchy shared by every layer of the pipeline.

Processing code raises a subclass of :class:`PipelineError`. The batch runner
catches :class:`ImageError` subclasses per image (the image is logged, recorded
in the report and skipped) while :class:`ConfigError` aborts before any image
is processed.
"""


class PipelineError(Exception):
    """Base class for every error raised on purpose by ``raw_pipeline``."""


class ConfigError(PipelineError):
    """The configuration file is missing, malformed or names unknown components."""


class RegistryError(PipelineError):
    """A component name is not registered, or registered twice."""


class ImageError(PipelineError):
    """Processing of a single image failed; the batch continues with the next one."""


class DecodeError(ImageError):
    """The RAW file could not be read or developed."""


class DefectCorrectionError(ImageError):
    """The defect map does not fit the sensor or correction failed."""


class DetectionError(ImageError):
    """Face detection failed in a way that is not simply "no face found"."""


class MeteringError(ImageError):
    """Exposure could not be measured (e.g. empty or non-finite image)."""


class StageError(ImageError):
    """A stage failed while transforming the image."""


class ExportError(ImageError):
    """Writing an output file failed."""


class OutputExistsError(ExportError):
    """The output file already exists and overwriting was not requested."""


class QAError(PipelineError):
    """A QA provider failed. Always handled: QA never fails an image."""
