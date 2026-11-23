"""HTTP transport used by network QA providers (an adapter, so tests inject fakes)."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

MAX_RESPONSE_BYTES: Final = 1_048_576
"""Responses larger than this (bytes) are rejected instead of being read into memory."""


class TransportError(Exception):
    """The request could not be completed (network, HTTP status, oversized response)."""


@dataclass(frozen=True)
class HttpRequest:
    """One JSON POST request.

    Attributes:
        url: Absolute URL.
        body: Request body (UTF-8 JSON).
        timeout_s: Total timeout in seconds.
        headers: Request headers. May contain credentials: never log them.
    """

    url: str
    body: bytes
    timeout_s: float
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class HttpTransport(Protocol):
    """Sends an :class:`HttpRequest` and returns the response body."""

    def post(self, request: HttpRequest) -> bytes:
        """POST ``request`` and return the body of a 2xx response.

        Raises:
            TransportError: On any network, timeout, HTTP-status or size problem.
        """
        ...


class UrllibTransport:
    """Standard-library transport (``urllib.request``); no third-party HTTP client needed."""

    def post(self, request: HttpRequest) -> bytes:
        """POST with ``urllib``; see :meth:`HttpTransport.post`."""
        req = urllib.request.Request(request.url, data=request.body, headers=dict(request.headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=request.timeout_s) as resp:
                body: bytes = resp.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, ValueError) as exc:
            # HTTPError/URLError/TimeoutError are OSErrors; their text never includes request headers.
            msg = f"{type(exc).__name__}: {exc}"
            raise TransportError(msg) from exc
        if len(body) > MAX_RESPONSE_BYTES:
            msg = f"response larger than {MAX_RESPONSE_BYTES} bytes"
            raise TransportError(msg)
        return body
