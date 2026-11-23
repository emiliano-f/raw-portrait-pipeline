from __future__ import annotations

import io
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from raw_pipeline.qa import transport as transport_module
from raw_pipeline.qa.transport import MAX_RESPONSE_BYTES, HttpRequest, RefuseRedirects, TransportError, UrllibTransport

REQUEST = HttpRequest(
    url="https://llm.example/v1/chat/completions",
    body=b"{}",
    timeout_s=3.0,
    headers={"Authorization": "Bearer secret-123", "Content-Type": "application/json"},
)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, behaviour: Any) -> list[tuple[urllib.request.Request, float]]:
    calls: list[tuple[urllib.request.Request, float]] = []

    def fake_open(req: urllib.request.Request, timeout_s: float) -> _Response:
        calls.append((req, timeout_s))
        if isinstance(behaviour, BaseException):
            raise behaviour
        return _Response(behaviour)

    # open_url is the adapter's boundary to the network; no socket is opened.
    monkeypatch.setattr(transport_module, "open_url", fake_open)
    return calls


def test_posts_body_headers_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_urlopen(monkeypatch, b'{"ok": true}')
    assert UrllibTransport().post(REQUEST) == b'{"ok": true}'
    ((req, timeout),) = calls
    assert req.get_method() == "POST"
    assert req.full_url == REQUEST.url
    assert req.data == b"{}"
    assert req.get_header("Authorization") == "Bearer secret-123"
    assert timeout == 3.0


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        urllib.error.URLError("no route"),
        urllib.error.HTTPError(REQUEST.url, 401, "Unauthorized", Message(), None),
        ValueError("unknown url type"),
    ],
)
def test_errors_become_transport_errors(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    _patch_urlopen(monkeypatch, error)
    with pytest.raises(TransportError) as info:
        UrllibTransport().post(REQUEST)
    assert "secret-123" not in str(info.value)


def test_oversized_response_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, b"x" * (MAX_RESPONSE_BYTES + 10))
    with pytest.raises(TransportError, match="larger than"):
        UrllibTransport().post(REQUEST)


def test_request_repr_hides_headers() -> None:
    assert "secret-123" not in repr(REQUEST)


def test_redirects_are_refused() -> None:
    req = urllib.request.Request(REQUEST.url, data=b"{}", headers=dict(REQUEST.headers), method="POST")
    with pytest.raises(urllib.error.HTTPError, match="redirect refused") as info:
        RefuseRedirects().redirect_request(req, io.BytesIO(), 302, "Found", Message(), "http://evil.example/")
    assert info.value.code == 302


def test_opener_uses_refuse_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers: list[object] = []

    class _Opener:
        def open(self, req: urllib.request.Request, timeout: float) -> str:
            del req, timeout
            return "response"

    def fake_build_opener(*h: object) -> _Opener:
        handlers.extend(h)
        return _Opener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    req = urllib.request.Request(REQUEST.url)
    assert transport_module.open_url(req, 1.0) == "response"
    assert any(isinstance(h, RefuseRedirects) for h in handlers)
