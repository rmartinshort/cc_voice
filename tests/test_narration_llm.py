"""Tests for NarrationLLM prompt assembly and mode selection.

Uses unittest.mock to patch the Anthropic client so no real API calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_observer.models import ClaudeCodeEvent, NarrationTurn, UserVoiceEvent
from voice_observer.narration_llm import NarrationLLM


def make_claude_event(summary: str = "Writing auth.py") -> ClaudeCodeEvent:
    return ClaudeCodeEvent(event_type="tool_use", summary=summary, raw_data={})


def make_voice_event(transcript: str = "Why is he doing that?") -> UserVoiceEvent:
    return UserVoiceEvent(transcript=transcript)


def make_mock_response(text: str = "He's writing the auth module.") -> MagicMock:
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture
def llm_with_mock_client():
    """Return a NarrationLLM with the Anthropic client mocked."""
    with patch("voice_observer.narration_llm.anthropic.AsyncAnthropic") as mock_class:
        mock_client = AsyncMock()
        mock_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=make_mock_response())
        llm = NarrationLLM(model="claude-haiku-4-5-20251001")
        llm._client = mock_client
        yield llm, mock_client


@pytest.mark.asyncio
async def test_proactive_mode_for_claude_event(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    event = make_claude_event("Writing auth.py")
    turn = await llm.narrate(event, context={})
    assert turn.mode == "proactive"


@pytest.mark.asyncio
async def test_reactive_mode_for_voice_event(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    event = make_voice_event("Why is he doing that?")
    turn = await llm.narrate(event, context={})
    assert turn.mode == "reactive"


@pytest.mark.asyncio
async def test_proactive_prompt_contains_observed(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    event = make_claude_event("Writing auth.py")
    await llm.narrate(event, context={})

    call_kwargs = mock_client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any("observed" in m["content"].lower() for m in messages)


@pytest.mark.asyncio
async def test_reactive_prompt_contains_asked(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    event = make_voice_event("Why is he doing that?")
    await llm.narrate(event, context={})

    call_kwargs = mock_client.messages.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any("asked" in m["content"].lower() for m in messages)


@pytest.mark.asyncio
async def test_narration_turn_has_response_text(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    mock_client.messages.create = AsyncMock(
        return_value=make_mock_response("He's writing the auth module.")
    )
    event = make_claude_event()
    turn = await llm.narrate(event, context={})
    assert turn.response_text == "He's writing the auth module."


@pytest.mark.asyncio
async def test_narration_turn_returned_on_api_error(llm_with_mock_client):
    """NarrationLLM should return a turn even when the API call fails."""
    llm, mock_client = llm_with_mock_client
    mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))
    event = make_claude_event()
    turn = await llm.narrate(event, context={})
    assert isinstance(turn, NarrationTurn)
    assert "unavailable" in turn.response_text


@pytest.mark.asyncio
async def test_max_tokens_is_300(llm_with_mock_client):
    llm, mock_client = llm_with_mock_client
    await llm.narrate(make_claude_event(), context={})
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 300
