from __future__ import annotations

from raw_pipeline.core import errors


def test_hierarchy() -> None:
    assert issubclass(errors.DecodeError, errors.ImageError)
    assert issubclass(errors.OutputExistsError, errors.ExportError)
    assert issubclass(errors.ImageError, errors.PipelineError)
    assert issubclass(errors.ConfigError, errors.PipelineError)
    assert not issubclass(errors.ConfigError, errors.ImageError)
    assert not issubclass(errors.QAError, errors.ImageError)
