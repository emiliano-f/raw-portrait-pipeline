"""``openai_compatible`` QA provider: any vision model behind a chat-completions API."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import ClassVar, Final
from urllib.parse import urlsplit

from raw_pipeline.config import app_config
from raw_pipeline.core.errors import ConfigError
from raw_pipeline.core.registries import qa_registry
from raw_pipeline.core.registry import BuildEnv
from raw_pipeline.core.values import QAResult
from raw_pipeline.qa.preview import MAX_PREVIEW_LONG_EDGE_PX, inspect_preview
from raw_pipeline.qa.response import SYSTEM_PROMPT, USER_PROMPT, ResponseFormatError, message_content, parse_verdict
from raw_pipeline.qa.target import require_endpoint_and_model
from raw_pipeline.qa.transport import HttpRequest, HttpTransport, TransportError, UrllibTransport

logger = logging.getLogger(__name__)

MAX_RESPONSE_TOKENS: Final = 200
"""Upper bound on the length of the model answer, in tokens."""

_LOCAL_HOSTS: Final = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_endpoint(endpoint: str) -> str:
    """Return ``endpoint`` without a trailing slash if it is safe to send portraits to.

    HTTPS is required, except for plain HTTP to the local machine (self-hosted models).

    Raises:
        ConfigError: For any other scheme or a URL without host.
    """
    parts = urlsplit(endpoint)
    secure = parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS)
    if not parts.hostname or not secure:
        msg = f"qa.endpoint must be an https:// URL (http:// only for localhost), got {endpoint!r}"
        raise ConfigError(msg)
    return endpoint.rstrip("/")


def is_valid_api_key(api_key: str) -> bool:
    """Return ``True`` if ``api_key`` is printable ASCII without spaces (safe in an HTTP header)."""
    return api_key.isascii() and api_key.isprintable() and " " not in api_key


@qa_registry.register("openai_compatible")
class OpenAICompatibleProvider:
    """Sends one downscaled JPEG preview to a chat-completions endpoint and parses flags.

    Works with any service implementing the OpenAI chat-completions API with
    image inputs (``image_url`` data URLs). It never raises: a missing API key,
    an oversized or invalid preview, network/HTTP/timeout errors and malformed
    answers all produce ``QAResult(skipped=True)`` and a warning. The API key is
    only placed in the ``Authorization`` header; it is never logged or reported.

    Args:
        api_key: Bearer token, or ``None`` to skip every review.
        api_key_env: Name of the environment variable the key came from (for messages).
        endpoint: Validated API base URL; ``/chat/completions`` is appended.
        model: Model identifier sent with each request.
        timeout_s: Request timeout in seconds.
        transport: HTTP adapter (injected; :class:`UrllibTransport` in production).
    """

    name: ClassVar[str] = "openai_compatible"

    def __init__(
        self,
        api_key: str | None,
        *,
        api_key_env: str = "QA_API_KEY",
        endpoint: str,
        model: str,
        timeout_s: float = 30.0,
        transport: HttpTransport | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.endpoint = validate_endpoint(endpoint)
        self.model = model
        self.timeout_s = timeout_s
        self._transport: HttpTransport = transport or UrllibTransport()
        self._api_key: str | None = None
        self._key_problem: str | None = None
        if not api_key:
            self._key_problem = f"no API key in ${api_key_env}"
        elif not is_valid_api_key(api_key):
            self._key_problem = f"${api_key_env} contains whitespace or control characters"
        else:
            self._api_key = api_key
        if self._key_problem is not None:
            logger.warning("QA enabled but %s: QA will be skipped for every image", self._key_problem)

    @classmethod
    def from_env(cls, env: BuildEnv) -> OpenAICompatibleProvider:
        """Build from ``qa.*``; the key is read from the variable named by ``qa.api_key_env``.

        Raises:
            ConfigError: If ``qa.endpoint`` or ``qa.model`` is null, or the endpoint is not an acceptable URL.
        """
        cfg = app_config(env).qa
        endpoint, model = require_endpoint_and_model(cfg)
        return cls(
            os.environ.get(cfg.api_key_env),
            api_key_env=cfg.api_key_env,
            endpoint=endpoint,
            model=model,
            timeout_s=cfg.timeout_s,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(endpoint={self.endpoint!r}, model={self.model!r})"

    def review(self, preview_jpeg: bytes) -> QAResult:
        """Send ``preview_jpeg`` for review; see the class docstring for failure handling."""
        if self._api_key is None:
            return self._skipped(self._key_problem or "no API key", warn=False)
        try:
            info = inspect_preview(preview_jpeg)
        except ValueError as exc:
            return self._skipped(str(exc))
        if info.long_edge_px > MAX_PREVIEW_LONG_EDGE_PX:
            return self._skipped(f"preview long edge {info.long_edge_px} px exceeds {MAX_PREVIEW_LONG_EDGE_PX} px")
        try:
            body = self._transport.post(self._request(preview_jpeg))
            verdict = parse_verdict(message_content(body))
        except (TransportError, ResponseFormatError) as exc:
            return self._skipped(f"request failed: {exc}")
        return QAResult(
            provider=self.name,
            flags=frozenset(self._redact(flag) for flag in verdict.flags),
            notes=self._redact(verdict.notes),
            model=self.model,
        )

    def _redact(self, text: str) -> str:
        """Remove the API key from server-controlled text (a service may echo it)."""
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _request(self, preview_jpeg: bytes) -> HttpRequest:
        image_url = "data:image/jpeg;base64," + base64.b64encode(preview_jpeg).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": MAX_RESPONSE_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        return HttpRequest(
            url=f"{self.endpoint}/chat/completions",
            body=json.dumps(payload, sort_keys=True).encode("utf-8"),
            timeout_s=self.timeout_s,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )

    def _skipped(self, reason: str, *, warn: bool = True) -> QAResult:
        reason = self._redact(reason)
        if warn:
            logger.warning("QA skipped: %s", reason)
        return QAResult(provider=self.name, skipped=True, notes=reason, model=self.model)
