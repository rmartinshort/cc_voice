"""Context Mux: priority queue + rolling context state.

The mux is the single shared state object in the system. It owns:
- The asyncio.PriorityQueue that serialises voice and Claude Code events
- The rolling event log (last 50 events)
- The rolling conversation history (last 20 turns)

Queue tuple format: (priority: int, seq: int)
Events are stored in self._pending keyed by seq to avoid Pydantic model
comparison errors (asyncio.PriorityQueue requires comparable items).
"""

import asyncio
import logging

from voice_observer.models import (
    ClaudeCodeEvent,
    ContextEvent,
    NarrationTurn,
    UserVoiceEvent,
)

logger = logging.getLogger(__name__)

_MAX_EVENT_LOG = 50
_MAX_CONVERSATION = 20


class ContextMux:
    def __init__(
        self,
        maxsize: int = 200,
        prior_event_log: list["ClaudeCodeEvent"] | None = None,
        prior_conversation: list["NarrationTurn"] | None = None,
    ) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int]] = asyncio.PriorityQueue(
            maxsize=maxsize
        )
        self._pending: dict[int, ContextEvent] = {}
        self._seq: int = 0

        self._event_log: list[ClaudeCodeEvent] = list(prior_event_log or [])
        self._conversation: list[NarrationTurn] = list(prior_conversation or [])
        self._task_prompt: str = ""

    def set_task_prompt(self, prompt: str) -> None:
        self._task_prompt = prompt

    async def put_claude_event(self, event: ClaudeCodeEvent) -> None:
        seq = self._next_seq()
        self._pending[seq] = event
        await self._queue.put((event.priority, seq))
        logger.debug("Enqueued Claude event seq=%d type=%s", seq, event.event_type)

    async def put_user_voice(self, event: UserVoiceEvent) -> None:
        seq = self._next_seq()
        self._pending[seq] = event
        await self._queue.put((event.priority, seq))
        logger.debug("Enqueued user voice event seq=%d", seq)

    async def get(self) -> ContextEvent:
        _, seq = await self._queue.get()
        event = self._pending.pop(seq)
        return event

    def task_done(self) -> None:
        """Must be called after each get() to allow queue.join() to work."""
        self._queue.task_done()

    def log_event(self, event: ClaudeCodeEvent) -> None:
        """Add event to context log without queuing it for narration."""
        self._event_log.append(event)
        if len(self._event_log) > _MAX_EVENT_LOG:
            self._event_log = self._event_log[-_MAX_EVENT_LOG:]

    def record_turn(self, turn: NarrationTurn) -> None:
        """Update rolling context after each narration turn."""
        if isinstance(turn.trigger, ClaudeCodeEvent):
            self._event_log.append(turn.trigger)
            if len(self._event_log) > _MAX_EVENT_LOG:
                self._event_log = self._event_log[-_MAX_EVENT_LOG:]

        self._conversation.append(turn)
        if len(self._conversation) > _MAX_CONVERSATION:
            self._conversation = self._conversation[-_MAX_CONVERSATION:]

    def get_context_snapshot(self) -> dict:
        """Return current context for Narration LLM prompt assembly."""
        return {
            "task_prompt": self._task_prompt,
            "event_log": [e.summary for e in self._event_log],
            "conversation": [
                {"mode": t.mode, "response": t.response_text}
                for t in self._conversation
            ],
        }

    @property
    def event_log(self) -> list[ClaudeCodeEvent]:
        return self._event_log

    @property
    def conversation(self) -> list[NarrationTurn]:
        return self._conversation

    def drain_nowait(self) -> list[ContextEvent]:
        """Non-blocking drain: return all currently queued events."""
        events: list[ContextEvent] = []
        while not self._queue.empty():
            try:
                _, seq = self._queue.get_nowait()
                events.append(self._pending.pop(seq))
            except asyncio.QueueEmpty:
                break
        return events

    def empty(self) -> bool:
        return self._queue.empty()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
