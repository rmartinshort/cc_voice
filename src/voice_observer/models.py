"""Pydantic data models shared across all components."""

import time
from enum import IntEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class EventPriority(IntEnum):
    USER_VOICE = 0
    CLAUDE_CODE = 1


class UserVoiceEvent(BaseModel):
    priority: Literal[0] = 0  # EventPriority.USER_VOICE
    transcript: str
    timestamp: float = Field(default_factory=time.monotonic)


class ClaudeCodeEvent(BaseModel):
    priority: Literal[1] = 1  # EventPriority.CLAUDE_CODE
    # Vocabulary: "tool_use" | "thinking" | "tool_result" | "text" | "error" | "complete" | "batch"
    event_type: str
    summary: str
    raw_data: dict
    timestamp: float = Field(default_factory=time.monotonic)


# Union type used throughout; order matters for isinstance() checks
ContextEvent = Union[UserVoiceEvent, ClaudeCodeEvent]


class NarrationTurn(BaseModel):
    mode: Literal["proactive", "reactive"]
    trigger: Annotated[ContextEvent, Field(discriminator="priority")]
    response_text: str
    timestamp: float = Field(default_factory=time.monotonic)
