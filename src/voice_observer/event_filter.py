"""Event filter: rules engine that converts the raw SDK event stream into a
sparser, narratively significant stream.

Phase 1 rules:
- tool_use: always emit, except repeated Read on same file within 30s
- thinking: always emit
- tool_result (is_error only): always emit
- text: drop in Phase 1 (token noise)
- complete / error: always emit
- batch: if 3+ events within 5s, combine into a single summary
"""

import logging
import time
from collections.abc import AsyncIterator

from voice_observer.models import ClaudeCodeEvent

logger = logging.getLogger(__name__)

_READ_TOOL_NAMES = {"Read", "read_file", "read"}
_WRITE_TOOL_NAMES = {"Write", "write_file", "create_file"}
_EDIT_TOOL_NAMES = {
    "Edit",
    "edit_file",
    "str_replace_editor",
    "str_replace_based_edit_tool",
}
_BASH_TOOL_NAMES = {"Bash", "bash", "execute_bash", "computer"}

# Events within this window get batched
_BATCH_WINDOW_SECONDS = 5.0
_BATCH_THRESHOLD = 3


class EventFilter:
    def __init__(self) -> None:
        self._last_read_file_times: dict[str, float] = {}
        self._recent_events: list[ClaudeCodeEvent] = []
        self._last_batch_emit: float = 0.0

    def should_emit(self, event: ClaudeCodeEvent) -> bool:
        """Return True if the event is narratively significant."""
        if event.event_type in ("complete", "error"):
            return True

        if event.event_type == "thinking":
            return True

        if event.event_type == "tool_result":
            # Only emit error results
            return bool(event.raw_data.get("is_error"))

        if event.event_type == "text":
            return False  # drop in Phase 1

        if event.event_type == "tool_output":
            return False  # display-only; not narrated

        if event.event_type == "tool_use":
            name = event.raw_data.get("name", "")
            if name in _READ_TOOL_NAMES:
                filename = _extract_filename(event.raw_data.get("input", {}))
                now = time.monotonic()
                last = self._last_read_file_times.get(filename, 0.0)
                if now - last < 30.0:
                    logger.debug("Suppressing repeated read of %s", filename)
                    return False
                self._last_read_file_times[filename] = now
            return True

        return True  # pass unknown event types through

    def enrich_summary(self, event: ClaudeCodeEvent) -> ClaudeCodeEvent:
        """Return a copy of the event with a human-readable summary."""
        if event.event_type != "tool_use":
            return event
        name = event.raw_data.get("name", "")
        inp = event.raw_data.get("input", {})
        return event.model_copy(update={"summary": summarize_tool_use(name, inp)})

    async def filter_events(
        self, raw_stream: AsyncIterator[ClaudeCodeEvent]
    ) -> AsyncIterator[ClaudeCodeEvent]:
        """Async generator. Consumes raw stream, yields significant events.

        Applies the batching rule: if 3+ events arrive within 5s, emit a combined
        summary instead of individual narrations.
        """
        buffer: list[ClaudeCodeEvent] = []
        buffer_start: float = 0.0

        async for event in raw_stream:
            if not self.should_emit(event):
                continue

            enriched = self.enrich_summary(event)

            # Pass-through events that always get individual narration
            if event.event_type in ("complete", "error"):
                if buffer:
                    yield _make_batch_event(buffer)
                    buffer = []
                yield enriched
                continue

            now = time.monotonic()
            if not buffer:
                buffer_start = now

            buffer.append(enriched)

            if (
                len(buffer) >= _BATCH_THRESHOLD
                and (now - buffer_start) < _BATCH_WINDOW_SECONDS
            ):
                # Accumulate — don't emit yet; wait until window expires or stream ends
                continue

            if now - buffer_start >= _BATCH_WINDOW_SECONDS:
                if len(buffer) >= _BATCH_THRESHOLD:
                    yield _make_batch_event(buffer)
                else:
                    for e in buffer:
                        yield e
                buffer = []

        # Flush remaining buffer
        if buffer:
            if len(buffer) >= _BATCH_THRESHOLD:
                yield _make_batch_event(buffer)
            else:
                for e in buffer:
                    yield e


def summarize_tool_use(tool_name: str, tool_input: dict) -> str:
    """Build a human-readable one-line summary for a tool use event.

    Used by both the event filter (to enrich summaries) and the hook receiver
    (to label PreToolUse events before they enter the pipeline).
    """
    if tool_name in _READ_TOOL_NAMES:
        path = _extract_filename(tool_input)
        return f"Reading {path}" if path else "Reading a file"
    if tool_name in _WRITE_TOOL_NAMES:
        path = _extract_filename(tool_input)
        return f"Writing {path}" if path else "Writing a new file"
    if tool_name in _EDIT_TOOL_NAMES:
        path = _extract_filename(tool_input)
        return f"Editing {path}" if path else "Editing a file"
    if tool_name in _BASH_TOOL_NAMES:
        command = str(tool_input.get("command", "")).strip()
        return f"Running: {command[:120]}" if command else "Running a shell command"
    return f"Using tool: {tool_name}"


def _extract_filename(inp: dict) -> str:
    """Extract a filepath from a tool input dict."""
    for key in ("path", "file_path", "filename", "filepath"):
        if key in inp:
            return str(inp[key])
    return ""


def _make_batch_event(events: list[ClaudeCodeEvent]) -> ClaudeCodeEvent:
    summaries = "; ".join(e.summary for e in events)
    return ClaudeCodeEvent(
        event_type="batch",
        summary=f"{len(events)} things happened: {summaries}",
        raw_data={"events": [e.model_dump() for e in events]},
    )
