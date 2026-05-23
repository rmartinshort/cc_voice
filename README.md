# Voice Observer

A real-time voice narration layer for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). While Claude Code works on coding tasks, a separate narrator LLM watches the event stream and speaks a running commentary — like a senior engineer pair-programming next to you. You can ask questions by speaking at any time.

## How it works

Voice Observer runs as a **sidecar process** alongside Claude Code. It uses Claude Code's [hooks system](https://docs.anthropic.com/en/docs/claude-code/hooks) to receive events (tool calls, command outputs, task completion) without modifying CC or replacing its interactive terminal experience.

```
Claude Code (interactive terminal)
    │
    ├── PreToolUse hook ──→  Voice Observer sidecar
    ├── PostToolUse hook ──→      │
    └── Stop hook ──→             ▼
                            Event Filter → Context Mux → Narration LLM (Haiku) → TTS → Speaker
                                                ▲
                            Microphone → VAD → STT ──→ (reactive questions)
```

The narrator uses "we" language ("we're running the tests now") rather than third-person commentary, making it feel like a pair-programming partner.

## Setup

### Requirements

- Python 3.11+
- Claude Code CLI installed and authenticated (`claude` on PATH)
- Microphone + speakers (headphones recommended)

### Install

```bash
git clone <repo-url>
cd cc_voice
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | For the narration LLM (Haiku) |
| `ELEVENLABS_API_KEY` | No | For high-quality TTS. Falls back to macOS `say` |
| `ELEVENLABS_VOICE_ID` | No | ElevenLabs voice to use |
| `WHISPER_MODEL` | No | `base.en` (default, fast) or `small.en` (accurate) |
| `NARRATION_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |

## Usage

### Sidecar mode (recommended)

Run Voice Observer alongside Claude Code — you keep the full CC interactive experience.

```bash
# One-time: install hooks into your project
voice-observer setup

# Start the sidecar (in a separate terminal)
voice-observer start

# Use Claude Code normally in your main terminal
claude

# When done
voice-observer stop

# Remove hooks
voice-observer teardown
```

While CC works, the sidecar terminal shows activity and narrates what's happening:

```
  ▶ Reading src/main.py
  ▶ $ python -m pytest tests/
    PASSED (3 tests)
  ✓ Task complete.
PROACTIVE We're running the tests — looks like everything passes.
```

### SDK-driven mode (legacy)

Drives Claude Code directly via the SDK. Simpler but replaces CC's interactive terminal.

```bash
# With a text prompt
voice-observer run "write a Python CSV parser" --cwd /path/to/project

# Voice-first (speak your task)
voice-observer run --cwd /path/to/project
```

### Options

| Flag | Description |
|---|---|
| `--debug` | Verbose logging |
| `--cwd PATH` | Working directory for Claude Code |
| `--barge-in` | Enable voice interruption of TTS (requires headphones) |

### Voice interaction

- **Ask questions**: Speak naturally — "what just happened?" or "why did it do that?" The narrator answers from its accumulated context.
- **Keyboard barge-in**: Press `b` to interrupt the narrator mid-sentence.
- **Stop a task** (SDK mode): Say "stop", "cancel", or "abort".

## Architecture

| Module | Role |
|---|---|
| `hook_receiver.py` | Unix socket server; parses CC hook JSON into events |
| `hook_forwarder.py` | Tiny CLI called by CC hooks; forwards stdin to socket |
| `sidecar.py` | Daemon: socket server + voice pipeline + narration consumer |
| `event_filter.py` | Drops noise, batches rapid events, enriches summaries |
| `context_mux.py` | Priority queue + rolling context (last 50 events, 20 turns) |
| `narration_llm.py` | Calls Haiku with event + context; proactive/reactive modes |
| `voice_pipeline.py` | Mic input (faster-whisper STT) + TTS output (ElevenLabs / macOS say) |
| `orchestrator.py` | SDK-driven mode: drives CC via `claude-code-sdk` |
| `main.py` | CLI entry point with subcommands |

## License

MIT
