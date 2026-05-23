"""Async wrapper around the Claude Code SDK.

Yields ClaudeCodeEvent objects so nothing downstream ever touches the SDK directly.
"""

import json
import logging
import os
from collections.abc import AsyncIterator

from claude_code_sdk import ClaudeCodeOptions, query
from claude_code_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from voice_observer.models import ClaudeCodeEvent

logger = logging.getLogger(__name__)


async def stream_claude_code_events(
    prompt: str,
    cwd: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[ClaudeCodeEvent]:
    """Run a Claude Code task and yield ClaudeCodeEvent objects.

    Uses include_partial_messages=False (default) so only complete messages arrive,
    which are already structured enough for the event filter.

    Pass session_id to resume a previous Claude Code session and preserve context
    across multiple tasks.
    """
    options = ClaudeCodeOptions(cwd=cwd or os.getcwd(), resume=session_id)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    event = _block_to_event(block)
                    if event is not None:
                        logger.debug(
                            "SDK block: %s -> %s",
                            type(block).__name__,
                            event.event_type,
                        )
                        yield event

            elif isinstance(message, ResultMessage):
                summary = "Task complete."
                if message.is_error:
                    summary = (
                        f"Task ended with an error. Cost: ${message.total_cost_usd:.4f}"
                    )
                yield ClaudeCodeEvent(
                    event_type="complete",
                    summary=summary,
                    raw_data={
                        "is_error": message.is_error,
                        "total_cost_usd": message.total_cost_usd,
                        "session_id": message.session_id,
                    },
                )
            else:
                logger.debug("Skipping message type: %s", type(message).__name__)

    except Exception as exc:
        logger.exception("Claude Code SDK error")
        yield ClaudeCodeEvent(
            event_type="error",
            summary=f"An error occurred: {exc}",
            raw_data={"exception": repr(exc)},
        )


def _block_to_event(block: object) -> ClaudeCodeEvent | None:
    """Convert a single SDK content block to a ClaudeCodeEvent, or None to skip."""
    if isinstance(block, ThinkingBlock):
        preview = (block.thinking or "")[:300]
        return ClaudeCodeEvent(
            event_type="thinking",
            summary=f"Thinking: {preview}",
            raw_data={"thinking": block.thinking},
        )

    if isinstance(block, ToolUseBlock):
        input_preview = json.dumps(block.input)[:200] if block.input else ""
        return ClaudeCodeEvent(
            event_type="tool_use",
            summary=f"Tool call: {block.name}({input_preview})",
            raw_data={"name": block.name, "input": block.input, "id": block.id},
        )

    if isinstance(block, ToolResultBlock):
        is_error = getattr(block, "is_error", False)
        content_str = _extract_tool_result_content(block.content)
        content_preview = content_str[:300]
        if is_error:
            return ClaudeCodeEvent(
                event_type="tool_result",
                summary=f"Tool error: {content_preview}",
                raw_data={"is_error": True, "content": content_str},
            )
        return ClaudeCodeEvent(
            event_type="tool_output",
            summary=content_preview,
            raw_data={"is_error": False, "content": content_str},
        )

    if isinstance(block, TextBlock):
        text = (block.text or "").strip()
        if len(text) < 20:
            return None  # skip short/noise text
        return ClaudeCodeEvent(
            event_type="text",
            summary=text[:300],
            raw_data={"text": block.text},
        )

    logger.debug("Unknown block type: %s", type(block).__name__)
    return None


def _extract_tool_result_content(content: object) -> str:
    """Extract text from ToolResultBlock.content (str | list[dict] | None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content[:1000]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)[:1000]
    return str(content)[:1000]
