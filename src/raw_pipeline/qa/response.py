"""Prompt sent to vision LLMs and parsing of their JSON verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

KNOWN_FLAGS: Final = frozenset({"closed_eyes", "blur", "skin_tone"})
"""Issue identifiers a provider may report; anything else in a response is dropped."""

MAX_NOTES_CHARS: Final = 300
"""Provider notes are truncated to this many characters before reaching the report."""

SYSTEM_PROMPT: Final = (
    "You review portrait photographs for technical problems. You never edit images. "
    "Look only for these issues: "
    "'closed_eyes' (a subject's eyes are closed or half closed), "
    "'blur' (the main face is out of focus or motion-blurred), "
    "'skin_tone' (skin looks unnatural: colour cast, too orange, grey, magenta or green). "
    'Answer with a single JSON object and nothing else: {"flags": [...], "notes": "..."} '
    "where flags lists zero or more of closed_eyes, blur, skin_tone and notes is one short sentence."
)
USER_PROMPT: Final = "Review this portrait preview."


class ResponseFormatError(ValueError):
    """The provider answered, but not with the expected JSON verdict."""


@dataclass(frozen=True)
class Verdict:
    """Parsed provider answer.

    Attributes:
        flags: Recognised issue identifiers (subset of :data:`KNOWN_FLAGS`).
        notes: Short explanation, at most :data:`MAX_NOTES_CHARS` characters, single line.
    """

    flags: frozenset[str]
    notes: str


def parse_verdict(content: str) -> Verdict:
    """Parse the model's message text into a :class:`Verdict`.

    Accepts the bare JSON object, optionally wrapped in a Markdown code fence or
    surrounded by prose. Unknown flags are dropped.

    Raises:
        ResponseFormatError: If no JSON object with a ``flags`` list is found.
    """
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        msg = "no JSON object in the model answer"
        raise ResponseFormatError(msg)
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON in the model answer: {exc.msg}"
        raise ResponseFormatError(msg) from exc
    # The slice starts with "{", so a successful parse is always a dict.
    flags = data.get("flags")
    if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
        msg = "model answer has no 'flags' list of strings"
        raise ResponseFormatError(msg)
    notes = data.get("notes", "")
    notes_text = " ".join(notes.split())[:MAX_NOTES_CHARS] if isinstance(notes, str) else ""
    return Verdict(flags=frozenset(f.strip().lower() for f in flags) & KNOWN_FLAGS, notes=notes_text)


def message_content(response_body: bytes) -> str:
    """Extract the assistant message text from a chat-completions response body.

    Handles ``content`` given as a string or as a list of ``{"type": "text"}`` parts.

    Raises:
        ResponseFormatError: If the body is not a chat-completions response.
    """
    try:
        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError) as exc:
        msg = f"unexpected chat-completions response ({type(exc).__name__})"
        raise ResponseFormatError(msg) from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    msg = "chat-completions message has no text content"
    raise ResponseFormatError(msg)
