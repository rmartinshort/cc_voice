"""Tests for ContextMux priority queue and context state."""

import asyncio

import pytest

from voice_observer.context_mux import ContextMux
from voice_observer.models import ClaudeCodeEvent, NarrationTurn, UserVoiceEvent


def make_claude_event(summary: str = "test") -> ClaudeCodeEvent:
    return ClaudeCodeEvent(event_type="tool_use", summary=summary, raw_data={})


def make_voice_event(transcript: str = "hello") -> UserVoiceEvent:
    return UserVoiceEvent(transcript=transcript)


def make_turn(event: ClaudeCodeEvent | UserVoiceEvent, mode: str = "proactive") -> NarrationTurn:
    return NarrationTurn(mode=mode, trigger=event, response_text="response")


@pytest.mark.asyncio
async def test_user_voice_beats_claude_event_even_when_enqueued_later():
    """Priority 0 (user voice) must be returned before priority 1 (Claude) even if Claude was put first."""
    mux = ContextMux()
    claude_event = make_claude_event()
    voice_event = make_voice_event()

    await mux.put_claude_event(claude_event)
    await mux.put_user_voice(voice_event)

    first = await mux.get()
    assert isinstance(first, UserVoiceEvent), "User voice should come out first"

    second = await mux.get()
    assert isinstance(second, ClaudeCodeEvent)


@pytest.mark.asyncio
async def test_same_priority_events_are_fifo():
    """Two Claude events should come out in insertion order."""
    mux = ContextMux()
    e1 = make_claude_event("first")
    e2 = make_claude_event("second")

    await mux.put_claude_event(e1)
    await mux.put_claude_event(e2)

    out1 = await mux.get()
    out2 = await mux.get()

    assert out1.summary == "first"
    assert out2.summary == "second"


def test_event_log_capped_at_50():
    mux = ContextMux()
    for i in range(60):
        event = make_claude_event(f"event {i}")
        turn = make_turn(event)
        mux.record_turn(turn)

    assert len(mux._event_log) == 50
    # Should retain the most recent 50
    assert mux._event_log[-1].summary == "event 59"


def test_conversation_capped_at_20():
    mux = ContextMux()
    for i in range(25):
        event = make_claude_event(f"event {i}")
        turn = make_turn(event)
        mux.record_turn(turn)

    assert len(mux._conversation) == 20


def test_get_context_snapshot_includes_task_prompt():
    mux = ContextMux()
    mux.set_task_prompt("write a CSV parser")
    snapshot = mux.get_context_snapshot()
    assert snapshot["task_prompt"] == "write a CSV parser"


def test_get_context_snapshot_includes_event_log():
    mux = ContextMux()
    event = make_claude_event("reading config.py")
    mux.record_turn(make_turn(event))
    snapshot = mux.get_context_snapshot()
    assert "reading config.py" in snapshot["event_log"]


@pytest.mark.asyncio
async def test_empty_returns_true_when_queue_empty():
    mux = ContextMux()
    assert mux.empty() is True
    await mux.put_claude_event(make_claude_event())
    assert mux.empty() is False
    await mux.get()
    assert mux.empty() is True
