# Voice Observer — Claude Code Developer Guide

## Project Overview

Voice Observer is a local developer tool that adds a real-time, bidirectional voice narration layer on top of Claude Code. While Claude Code silently writes code, a separate "Narration LLM" (Claude Haiku) observes the event stream and speaks a running commentary — like a senior engineer pair-programming next to you. The user can ask questions at any time via microphone; user voice always preempts proactive narration. The system uses asyncio-based priority queueing so voice and code events are never in a race condition.

**Full specification**: `voice-observer-prd.md` at the repo root.

---

## Session Start Checklist

At the start of every session, before doing anything else:
1. Read `STATUS.md` — current phase, what's done, open questions.
2. Confirm with the user whether to continue from the last session or pivot.

---

## Architecture

```
Microphone → Silero VAD → faster-whisper (STT)
                                │  Priority 0 (user voice)
                                ▼
Claude Code SDK ─── Event Filter ──► Context Mux (asyncio.PriorityQueue)
(async stream)       (drop noise)     Priority 1 (Claude events)
                                          │
                                          ▼
                                  Narration LLM (Haiku 3.5)
                                  Mode A: PROACTIVE (Claude event)
                                  Mode B: REACTIVE  (user voice)
                                          │
                                          ▼
                              Pipecat Pipeline → ElevenLabs TTS → Speaker
```

### Component Modules

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic types: `UserVoiceEvent`, `ClaudeCodeEvent`, `NarrationTurn` |
| `claude_stream.py` | Async wrapper around `claude-code-sdk` `query()` |
| `event_filter.py` | Rules engine; drops noise, emits structured summaries |
| `context_mux.py` | `asyncio.PriorityQueue` + rolling context state |
| `narration_llm.py` | Haiku 3.5 wrapper; mode selection; prompt assembly |
| `voice_pipeline.py` | Pipecat: VAD + STT input, TTS + barge-in output (Phase 3+) |
| `orchestrator.py` | asyncio task supervisor; wires all components |
| `main.py` | CLI entry point (argparse) |

---

## Phase Tracker

| Phase | Goal | Status |
|---|---|---|
| 0 | PRD + project scaffold | Done |
| 1 | Working skeleton — filtered events printed to terminal | In progress |
| 2 | Narration LLM replaces verbatim summaries | Not started |
| 3 | Always-on STT + barge-in | Not started |
| 4 | Polish, rolling context, rich terminal UI | Not started |

**Current phase: Phase 1**

---

## Environment Setup

### Required Tools
- Python 3.11+
- `uv` (recommended) or `pip`
- Claude Code CLI must be installed and authenticated (`claude` in PATH)

### Installation
```bash
cd cc_voice
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
# Fill in ANTHROPIC_API_KEY in .env
```

### Required Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Phase 1+ | For both Claude Code SDK and Narration LLM |
| `NARRATION_MODEL` | Phase 1+ | Defaults to `claude-haiku-4-5-20251001` |
| `ELEVENLABS_API_KEY` | Phase 3+ | For TTS |
| `ELEVENLABS_VOICE_ID` | Phase 3+ | Choose a voice |
| `WHISPER_MODEL` | Phase 3+ | `base.en` (fast) or `small.en` (accurate) |

Phase 1 only requires `ANTHROPIC_API_KEY`.

---

## Key Development Commands

```bash
# Run Phase 1 (terminal narration, no audio)
voice-observer "write a Python CSV parser"

# Run with verbose event logging
voice-observer "write a Python CSV parser" --debug

# Run in a specific directory
voice-observer "add tests for the auth module" --cwd /path/to/project

# Run tests
pytest tests/ -v

# Lint
ruff check src/
mypy src/
```

---

## Code Conventions

### Async patterns
- All I/O is async. Never call blocking functions without `asyncio.to_thread()` or async variants.
- The orchestrator owns the event loop. All components expose coroutines or async generators.
- `asyncio.PriorityQueue` entries are `(priority: int, seq: int)` tuples; events are stored separately in a dict keyed by `seq` to avoid Pydantic model comparison errors.

### Pydantic models
- All inter-component data uses the Pydantic models in `models.py`. Never pass raw dicts between modules.
- Use `model.model_dump()` for serialization (not `.dict()`, which is the Pydantic v1 shim).

### Claude Code SDK
- Package: `claude-code-sdk` (PyPI) → import `claude_code_sdk`
- Entry point: `query(prompt, options=ClaudeCodeOptions(...)) -> AsyncIterator[Message]`
- Message union: `AssistantMessage | UserMessage | ResultMessage | SystemMessage`
- `AssistantMessage.content` is a list of `ContentBlock` — can be `TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock`
- `ResultMessage` signals task completion

### Logging and output
- Module-level logger: `logger = logging.getLogger(__name__)`
- Rich `Console` object lives in `orchestrator.py` — use it for all user-visible output
- In Phase 1, narration is printed to terminal only. No audio.

### Error handling
- Wrap the orchestrator run in try/except; log cleanly and exit with code 1
- SDK/API errors are caught at the component level and emitted as synthetic `ClaudeCodeEvent(event_type="error", ...)` so the narration loop can speak them naturally

---

## Reference

- PRD: `voice-observer-prd.md`
- Claude Code SDK: `pip show claude-code-sdk` for installed version
- Pipecat docs: https://github.com/pipecat-ai/pipecat
- ElevenLabs Python SDK: https://github.com/elevenlabs/elevenlabs-python
