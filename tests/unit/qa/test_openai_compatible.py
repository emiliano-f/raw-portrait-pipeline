from __future__ import annotations

import base64
import io
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raw_pipeline.config import AppConfig
from raw_pipeline.config.models import QAConfig
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.protocols import QAProvider
from raw_pipeline.core.registries import qa_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.qa.openai_compatible import DEFAULT_ENDPOINT, DEFAULT_MODEL, OpenAICompatibleProvider
from raw_pipeline.qa.preview import encode_preview
from tests.contract.qa_provider_contract import QAProviderContract
from tests.support.synthetic import flat_image
from tests.unit.qa.fakes import FailingTransport, RecordingTransport, chat_response

SECRET = "sk-test-not-a-real-key-1234"
OK_BODY = chat_response('{"flags": ["blur"], "notes": "Soft focus."}')


def _provider(transport: object, api_key: str | None = SECRET, **kwargs: object) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(api_key, transport=transport, **kwargs)  # type: ignore[arg-type]


class TestOpenAICompatibleContract(QAProviderContract):
    def make_provider(self) -> QAProvider:
        return _provider(RecordingTransport(OK_BODY))


class TestOpenAICompatibleNoKeyContract(QAProviderContract):
    def make_provider(self) -> QAProvider:
        return _provider(FailingTransport(), api_key=None)


class TestOpenAICompatibleFailingContract(QAProviderContract):
    def make_provider(self) -> QAProvider:
        return _provider(FailingTransport())


def test_success_parses_flags(tiny_jpeg: bytes) -> None:
    result = _provider(RecordingTransport(OK_BODY), model="vision-1").review(tiny_jpeg)
    assert not result.skipped
    assert result.flags == {"blur"}
    assert result.notes == "Soft focus."
    assert result.model == "vision-1"


def test_request_shape(tiny_jpeg: bytes) -> None:
    transport = RecordingTransport(OK_BODY)
    _provider(transport, endpoint="https://llm.example/v1/", model="m", timeout_s=7.0).review(tiny_jpeg)
    (req,) = transport.requests
    assert req.url == "https://llm.example/v1/chat/completions"
    assert req.timeout_s == 7.0
    assert req.headers["Authorization"] == f"Bearer {SECRET}"
    payload = json.loads(req.body)
    assert payload["model"] == "m"
    assert payload["temperature"] == 0
    image_part = payload["messages"][1]["content"][1]
    prefix = "data:image/jpeg;base64,"
    assert image_part["image_url"]["url"].startswith(prefix)
    assert base64.b64decode(image_part["image_url"]["url"][len(prefix) :]) == tiny_jpeg


def test_request_body_deterministic(tiny_jpeg: bytes) -> None:
    a, b = RecordingTransport(OK_BODY), RecordingTransport(OK_BODY)
    _provider(a).review(tiny_jpeg)
    _provider(b).review(tiny_jpeg)
    assert a.requests[0].body == b.requests[0].body


def test_no_key_skips_without_request(tiny_jpeg: bytes, caplog: pytest.LogCaptureFixture) -> None:
    transport = RecordingTransport(OK_BODY)
    with caplog.at_level(logging.WARNING):
        provider = _provider(transport, api_key="", api_key_env="MY_KEY")
    assert "no API key in $MY_KEY" in caplog.text
    result = provider.review(tiny_jpeg)
    assert result.skipped
    assert "MY_KEY" in result.notes
    assert transport.requests == []


def test_transport_error_skips_with_warning(tiny_jpeg: bytes, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = _provider(FailingTransport("TimeoutError: timed out")).review(tiny_jpeg)
    assert result.skipped
    assert "timed out" in result.notes
    assert "QA skipped" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize("body", [b"<html>502</html>", chat_response("I cannot help."), chat_response('{"flags": 3}')])
def test_malformed_response_skips(tiny_jpeg: bytes, body: bytes) -> None:
    result = _provider(RecordingTransport(body)).review(tiny_jpeg)
    assert result.skipped
    assert result.flags == frozenset()


def test_refuses_oversized_preview() -> None:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((600, 1500, 3), np.uint8)).save(buf, format="JPEG")
    transport = RecordingTransport(OK_BODY)
    result = _provider(transport).review(buf.getvalue())
    assert result.skipped
    assert "exceeds 1024" in result.notes
    assert transport.requests == []


def test_refuses_non_jpeg_payload() -> None:
    transport = RecordingTransport(OK_BODY)
    result = _provider(transport).review(b"II*\x00 raw sensor bytes")
    assert result.skipped
    assert transport.requests == []


def test_accepts_max_size_preview() -> None:
    transport = RecordingTransport(OK_BODY)
    preview = encode_preview(flat_image(0.18, 2000, 3000), 1024, 85)
    assert not _provider(transport).review(preview).skipped
    assert len(transport.requests) == 1


@pytest.mark.parametrize("endpoint", ["http://llm.example/v1", "ftp://llm.example", "https://", "not a url"])
def test_rejects_insecure_endpoints(endpoint: str) -> None:
    with pytest.raises(ConfigError, match="https"):
        _provider(RecordingTransport(OK_BODY), endpoint=endpoint)


@pytest.mark.parametrize("endpoint", ["http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "https://x.example"])
def test_accepts_secure_or_local_endpoints(endpoint: str) -> None:
    _provider(RecordingTransport(OK_BODY), endpoint=endpoint)


def test_repr_hides_key() -> None:
    provider = _provider(RecordingTransport(OK_BODY))
    assert SECRET not in repr(provider)
    assert SECRET not in str(provider)


def test_picklable(tiny_jpeg: bytes) -> None:
    clone = pickle.loads(pickle.dumps(_provider(RecordingTransport(OK_BODY))))
    assert clone.review(tiny_jpeg).flags == {"blur"}


def test_from_env_reads_config_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_KEY", SECRET)
    cfg = AppConfig(qa=QAConfig(provider="openai_compatible", api_key_env="VISION_KEY", model="m", timeout_s=5))
    provider = qa_registry.create("openai_compatible", BuildEnv(config=cfg, output_dir=Path("out")))
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "m"
    assert provider.timeout_s == 5
    assert provider.endpoint == DEFAULT_ENDPOINT


def test_from_env_defaults_without_key(monkeypatch: pytest.MonkeyPatch, tiny_jpeg: bytes) -> None:
    monkeypatch.delenv("QA_API_KEY", raising=False)
    cfg = AppConfig(qa=QAConfig(provider="openai_compatible"))
    provider = OpenAICompatibleProvider.from_env(BuildEnv(config=cfg, output_dir=Path("out")))
    assert provider.model == DEFAULT_MODEL
    assert provider.review(tiny_jpeg).skipped


@pytest.mark.parametrize("bad_key", ["sk-abc\nInjected: 1", "sk abc", "sk-\u00e9t\u00e9", "sk-\x00"])
def test_invalid_key_skips_without_leaking(tiny_jpeg: bytes, bad_key: str, caplog: pytest.LogCaptureFixture) -> None:
    transport = RecordingTransport(OK_BODY)
    with caplog.at_level(logging.DEBUG):
        provider = _provider(transport, api_key=bad_key)
        result = provider.review(tiny_jpeg)
    assert result.skipped
    assert "control characters" in result.notes
    assert transport.requests == []
    assert bad_key not in caplog.text
    assert bad_key not in result.notes
