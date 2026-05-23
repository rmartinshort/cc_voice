"""Narration LLM: Claude Haiku wrapper for generating spoken commentary.

Phase 1: single-turn calls with no rolling history.
Phase 2: will inject full rolling context (event log + conversation history).
"""

import logging
import os
from pathlib import Path

import anthropic

from voice_observer.models import (
    ClaudeCodeEvent,
    ContextEvent,
    NarrationTurn,
    UserVoiceEvent,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "narration_system_prompt.txt"
)
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 300


class NarrationLLM:
    def __init__(self, model: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic()
        self._model = model or os.environ.get("NARRATION_MODEL", _DEFAULT_MODEL)
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning(
                "System prompt not found at %s; using fallback", _PROMPT_PATH
            )
            return (
                "You are a senior engineer narrating what you observe another developer doing. "
                "Be concise, conversational, and never read code verbatim."
            )

    async def narrate(
        self,
        event: ContextEvent,
        context: dict,
    ) -> NarrationTurn:
        """Call Haiku and return a NarrationTurn."""
        mode, user_content = self._build_user_message(event, context)
        messages = [{"role": "user", "content": user_content}]

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=self._system_prompt,
                messages=messages,
            )
            text = response.content[0].text if response.content else "(no response)"
        except Exception as exc:
            logger.exception("Narration LLM call failed")
            text = f"(narration unavailable: {exc})"

        return NarrationTurn(
            mode=mode,
            trigger=event,
            response_text=text,
        )

    def _build_user_message(
        self, event: ContextEvent, context: dict
    ) -> tuple[str, str]:
        """Return (mode, user_message_content) for the given event."""
        lines: list[str] = []

        task_prompt = context.get("task_prompt", "")
        if task_prompt:
            lines.append(f"Task being worked on: {task_prompt}")

        event_log: list[str] = context.get("event_log", [])
        if event_log:
            recent = event_log[-12:]
            lines.append("Recent events:\n" + "\n".join(f"- {e}" for e in recent))

        preamble = "\n\n".join(lines)

        if isinstance(event, UserVoiceEvent):
            mode = "reactive"
            current = f"The developer just asked: '{event.transcript}'"
        elif isinstance(event, ClaudeCodeEvent):
            mode = "proactive"
            current = f"You just observed: {event.summary}"
        else:
            mode = "proactive"
            current = f"You just observed: {event!r}"

        content = f"{preamble}\n\n{current}" if preamble else current
        return mode, content
