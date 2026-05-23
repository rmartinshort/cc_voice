"""Unix domain socket server that receives Claude Code hook events.

Each hook connection sends a single JSON object on stdin, which we parse
into a ClaudeCodeEvent and push to an asyncio.Queue for the sidecar to consume.

Hook event types:
- PreToolUse:  {tool_name, tool_input, session_id, turn}
- PostToolUse: {tool_name, tool_input, tool_output, session_id, turn}
- PostToolUseFailure: same as PostToolUse + error info
- Stop: {response_text, session_id, turn, total_cost}
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from voice_observer.event_filter import summarize_tool_use
from voice_observer.models import ClaudeCodeEvent
from voice_observer.transcript_reader import read_from_transcript

logger = logging.getLogger(__name__)

SOCKET_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_NAME = "voice-observer.sock"


def socket_path() -> Path:
    return SOCKET_DIR / SOCKET_NAME


def hook_event_to_claude_event(hook_type: str, data: dict) -> ClaudeCodeEvent:
    """Convert a raw hook JSON payload into a ClaudeCodeEvent."""
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "")

    if hook_type == "PreToolUse":
        summary = summarize_tool_use(tool_name, tool_input)
        # Try to capture the user's prompt from the transcript
        user_prompt = ""
        transcript_path = data.get("transcript_path", "")
        if transcript_path:
            user_prompt, _ = read_from_transcript(transcript_path)
        return ClaudeCodeEvent(
            event_type="tool_use",
            summary=summary,
            raw_data={
                "name": tool_name,
                "input": tool_input,
                "session_id": session_id,
                "user_prompt": user_prompt,
            },
        )

    if hook_type == "PostToolUse":
        tool_output = str(data.get("tool_output", ""))[:1000]
        return ClaudeCodeEvent(
            event_type="tool_output",
            summary=tool_output[:300],
            raw_data={
                "name": tool_name,
                "input": tool_input,
                "content": tool_output,
                "is_error": False,
                "session_id": session_id,
            },
        )

    if hook_type == "PostToolUseFailure":
        tool_output = str(data.get("tool_output", ""))[:1000]
        return ClaudeCodeEvent(
            event_type="tool_result",
            summary=f"Tool error: {tool_output[:300]}",
            raw_data={
                "name": tool_name,
                "input": tool_input,
                "content": tool_output,
                "is_error": True,
                "session_id": session_id,
            },
        )

    if hook_type == "Stop":
        response_text = data.get("response_text", "")
        total_cost = data.get("total_cost", 0.0)
        transcript_path = data.get("transcript_path", "")

        logger.debug(
            "Stop hook: response_text=%d chars, transcript_path=%s, all_keys=%s",
            len(response_text), transcript_path, list(data.keys()),
        )

        # If response_text is empty, try reading from transcript
        user_prompt = ""
        if transcript_path:
            t_prompt, t_response = read_from_transcript(transcript_path)
            logger.debug(
                "Transcript read: prompt=%d chars, response=%d chars",
                len(t_prompt), len(t_response),
            )
            if not response_text:
                response_text = t_response
            user_prompt = t_prompt
        else:
            logger.warning("No transcript_path in Stop hook data")

        # Include response preview in summary so narrator sees what CC said
        summary = "Task complete."
        if response_text:
            preview = response_text[:500].replace("\n", " ").strip()
            summary = f"Task complete. Response: {preview}"
        return ClaudeCodeEvent(
            event_type="complete",
            summary=summary,
            raw_data={
                "response_text": response_text,
                "user_prompt": user_prompt,
                "session_id": session_id,
                "total_cost_usd": total_cost,
            },
        )

    # Unknown hook type — pass through
    return ClaudeCodeEvent(
        event_type="hook_event",
        summary=f"Hook: {hook_type}",
        raw_data=data,
    )


class HookReceiver:
    """Async Unix domain socket server for receiving hook events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ClaudeCodeEvent] = asyncio.Queue()
        self._server: asyncio.Server | None = None
        self._path = socket_path()

    @property
    def queue(self) -> asyncio.Queue[ClaudeCodeEvent]:
        return self._queue

    async def start(self) -> None:
        # Clean up stale socket
        if self._path.exists():
            self._path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=str(self._path)
        )
        # Make socket world-writable so hooks can connect
        os.chmod(self._path, 0o777)
        logger.info("Hook receiver listening on %s", self._path)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._path.exists():
            self._path.unlink()
        logger.info("Hook receiver stopped")

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle one hook connection: read JSON, parse, enqueue, close."""
        try:
            raw = await asyncio.wait_for(reader.read(1024 * 1024), timeout=2.0)
            writer.close()
            await writer.wait_closed()

            if not raw:
                return

            payload = json.loads(raw.decode("utf-8", errors="replace"))
            hook_type = payload.get("hook_type", payload.get("type", "unknown"))

            event = hook_event_to_claude_event(hook_type, payload)
            logger.debug("Received hook event: %s -> %s", hook_type, event.event_type)
            await self._queue.put(event)

        except (json.JSONDecodeError, TimeoutError) as exc:
            logger.warning("Bad hook connection: %s", exc)
        except Exception:
            logger.exception("Hook connection error")
        finally:
            try:
                writer.close()
            except Exception:
                pass
