"""Read user prompts and assistant responses from Claude Code transcript JSONL files.

CC writes a transcript at the path provided in hook payloads (transcript_path).
The format is one JSON object per line. This module is the single place that
knows how to parse that format — keeping the socket server in hook_receiver.py
free of transcript logic.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_from_transcript(transcript_path: str) -> tuple[str, str]:
    """Read user prompt and last assistant response from a CC transcript JSONL file.

    Returns (user_prompt, assistant_response). Both are empty strings on failure
    or if the file doesn't exist. Truncates to 300 / 2000 chars respectively.
    """
    try:
        path = Path(transcript_path)
        if not path.exists():
            logger.debug("Transcript file not found: %s", transcript_path)
            return "", ""

        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        logger.debug("Transcript: %d lines from %s", len(lines), transcript_path)

        entries = []
        entry_types = []
        for line in lines:
            try:
                entry = json.loads(line)
                entries.append(entry)
                entry_types.append(entry.get("type", entry.get("role", "unknown")))
            except json.JSONDecodeError:
                continue

        logger.debug("Transcript entry types: %s", entry_types)
        if entries:
            logger.debug("First entry keys: %s", list(entries[0].keys()))

        user_prompt = _extract_user_prompt(entries)
        assistant_response = _extract_last_assistant_response(entries)

        return user_prompt[:300], assistant_response[:2000]

    except Exception as exc:
        logger.warning("Could not read transcript: %s", exc)
        return "", ""


def _extract_user_prompt(entries: list[dict]) -> str:
    """Return the first user/human message from the transcript entries."""
    for entry in entries:
        etype = entry.get("type", entry.get("role", ""))
        if etype in ("human", "user"):
            # Format 1: {"type": "human", "message": {"content": ...}}
            msg = entry.get("message", {})
            content = msg.get("content", entry.get("content", ""))
            text = _extract_text_from_content(content)
            if text:
                return text
    return ""


def _extract_last_assistant_response(entries: list[dict]) -> str:
    """Return the last assistant message from the transcript entries."""
    for entry in reversed(entries):
        etype = entry.get("type", entry.get("role", ""))
        if etype == "assistant":
            msg = entry.get("message", {})
            content = msg.get("content", entry.get("content", []))
            text = _extract_text_from_content(content)
            if text:
                return text
    return ""


def _extract_text_from_content(content: object) -> str:
    """Extract plain text from various CC content formats (str, list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts).strip()
    return ""
