from __future__ import annotations

import pytest

from raw_pipeline.qa.response import MAX_NOTES_CHARS, ResponseFormatError, message_content, parse_verdict
from tests.unit.qa.fakes import chat_response


def test_parses_bare_json() -> None:
    v = parse_verdict('{"flags": ["closed_eyes", "blur"], "notes": "Eyes shut."}')
    assert v.flags == {"closed_eyes", "blur"}
    assert v.notes == "Eyes shut."


def test_parses_fenced_json_and_normalises_flags() -> None:
    v = parse_verdict('Sure:\n```json\n{"flags": [" Skin_Tone ", "red_eye"], "notes": "a\\nb"}\n```')
    assert v.flags == {"skin_tone"}
    assert v.notes == "a b"


def test_empty_flags_and_missing_notes() -> None:
    v = parse_verdict('{"flags": []}')
    assert v.flags == frozenset()
    assert v.notes == ""


def test_notes_truncated() -> None:
    v = parse_verdict('{"flags": [], "notes": "' + "x" * 1000 + '"}')
    assert len(v.notes) == MAX_NOTES_CHARS


@pytest.mark.parametrize(
    "content",
    [
        "no json here",
        "{not json}",
        "[1, 2]",
        "[{}]",
        '{"notes": "no flags"}',
        '{"flags": "blur"}',
        '{"flags": [1]}',
    ],
)
def test_malformed_verdicts(content: str) -> None:
    with pytest.raises(ResponseFormatError):
        parse_verdict(content)


def test_message_content_string_and_parts() -> None:
    assert message_content(chat_response("hi")) == "hi"
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, "junk"]
    assert message_content(chat_response(parts)) == "ab"


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"choices": []}', b'{"choices": [{"message": null}]}', b"\xff"])
def test_message_content_malformed(body: bytes) -> None:
    with pytest.raises(ResponseFormatError):
        message_content(body)


def test_message_content_without_text() -> None:
    with pytest.raises(ResponseFormatError, match="no text"):
        message_content(chat_response(None))
