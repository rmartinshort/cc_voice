"""Tests for EventFilter rule logic."""

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from voice_observer.event_filter import EventFilter
from voice_observer.models import ClaudeCodeEvent


def make_event(event_type: str, name: str = "", input_data: dict | None = None, is_error: bool = False) -> ClaudeCodeEvent:
    raw: dict = {}
    if name:
        raw["name"] = name
    if input_data:
        raw["input"] = input_data
    if is_error:
        raw["is_error"] = True
    return ClaudeCodeEvent(event_type=event_type, summary="test summary", raw_data=raw)


async def collect(stream: AsyncIterator[ClaudeCodeEvent]) -> list[ClaudeCodeEvent]:
    return [e async for e in stream]


async def as_stream(*events: ClaudeCodeEvent) -> AsyncIterator[ClaudeCodeEvent]:
    for e in events:
        yield e


class TestShouldEmit:
    def test_bash_is_emitted(self):
        f = EventFilter()
        event = make_event("tool_use", name="Bash")
        assert f.should_emit(event) is True

    def test_write_is_emitted(self):
        f = EventFilter()
        event = make_event("tool_use", name="Write", input_data={"path": "foo.py"})
        assert f.should_emit(event) is True

    def test_thinking_is_emitted(self):
        f = EventFilter()
        event = make_event("thinking")
        assert f.should_emit(event) is True

    def test_complete_is_emitted(self):
        f = EventFilter()
        event = make_event("complete")
        assert f.should_emit(event) is True

    def test_error_is_emitted(self):
        f = EventFilter()
        event = make_event("error")
        assert f.should_emit(event) is True

    def test_text_is_dropped(self):
        f = EventFilter()
        event = make_event("text")
        assert f.should_emit(event) is False

    def test_tool_result_without_error_is_dropped(self):
        f = EventFilter()
        event = make_event("tool_result", is_error=False)
        assert f.should_emit(event) is False

    def test_tool_result_with_error_is_emitted(self):
        f = EventFilter()
        event = make_event("tool_result", is_error=True)
        assert f.should_emit(event) is True

    def test_first_read_is_emitted(self):
        f = EventFilter()
        event = make_event("tool_use", name="Read", input_data={"path": "main.py"})
        assert f.should_emit(event) is True

    def test_repeated_read_within_30s_is_suppressed(self):
        f = EventFilter()
        event = make_event("tool_use", name="Read", input_data={"path": "main.py"})
        f.should_emit(event)  # first call records the timestamp
        assert f.should_emit(event) is False

    def test_read_after_30s_is_emitted(self):
        f = EventFilter()
        event = make_event("tool_use", name="Read", input_data={"path": "main.py"})
        f.should_emit(event)
        # Manually backdate the timestamp
        f._last_read_file_times["main.py"] = time.monotonic() - 31.0
        assert f.should_emit(event) is True


class TestEnrichSummary:
    def test_bash_summary(self):
        f = EventFilter()
        event = make_event("tool_use", name="Bash")
        result = f.enrich_summary(event)
        assert result.summary == "Running a shell command"

    def test_write_summary_with_path(self):
        f = EventFilter()
        event = make_event("tool_use", name="Write", input_data={"path": "auth.py"})
        result = f.enrich_summary(event)
        assert "auth.py" in result.summary

    def test_read_summary_with_path(self):
        f = EventFilter()
        event = make_event("tool_use", name="Read", input_data={"path": "config.py"})
        result = f.enrich_summary(event)
        assert "config.py" in result.summary

    def test_non_tool_event_unchanged(self):
        f = EventFilter()
        event = make_event("thinking")
        result = f.enrich_summary(event)
        assert result.summary == event.summary


@pytest.mark.asyncio
async def test_filter_events_passes_complete_individually():
    """Complete events should never be batched."""
    f = EventFilter()
    events = [
        make_event("tool_use", name="Write", input_data={"path": "a.py"}),
        make_event("complete"),
    ]
    result = await collect(f.filter_events(as_stream(*events)))
    assert any(e.event_type == "complete" for e in result)


@pytest.mark.asyncio
async def test_filter_events_drops_text():
    f = EventFilter()
    events = [make_event("text"), make_event("complete")]
    result = await collect(f.filter_events(as_stream(*events)))
    assert not any(e.event_type == "text" for e in result)
