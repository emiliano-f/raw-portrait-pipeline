"""Fake HTTP transports for QA provider tests (no network)."""

from __future__ import annotations

import json

from raw_pipeline.qa.transport import HttpRequest, TransportError


def chat_response(content: object) -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


class RecordingTransport:
    """Returns a canned body and keeps every request it was given."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests: list[HttpRequest] = []

    def post(self, request: HttpRequest) -> bytes:
        self.requests.append(request)
        return self.body


class FailingTransport:
    """Always fails like a timeout or an HTTP error would."""

    def __init__(self, message: str = "TimeoutError: timed out") -> None:
        self.message = message
        self.calls = 0

    def post(self, request: HttpRequest) -> bytes:
        del request
        self.calls += 1
        raise TransportError(self.message)
