"""Minimal hook forwarder: reads JSON from stdin, sends to voice-observer socket.

Registered as `voice-observer-hook` console script. Called by Claude Code hooks.
Must be fast (<50ms) to avoid blocking CC execution.

Set VOICE_OBSERVER_DEBUG=1 to write received hook data to /tmp/voice-observer-debug.log
"""

import json
import os
import socket
import sys
from pathlib import Path

SOCKET_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
SOCKET_PATH = SOCKET_DIR / "voice-observer.sock"
DEBUG = os.environ.get("VOICE_OBSERVER_DEBUG", "") == "1"
DEBUG_LOG = Path("/tmp/voice-observer-debug.log")


def main() -> None:
    try:
        raw = sys.stdin.buffer.read()
        if not raw:
            return

        if DEBUG:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"--- RAW ({len(raw)} bytes) ---\n")
                f.write(raw.decode("utf-8", errors="replace")[:2000])
                f.write("\n\n")

        # The JSON stdin from CC includes a `hook_event_name` field
        # identifying the hook type (PreToolUse, PostToolUse, Stop, etc.)
        # We rename it to `hook_type` for our receiver.
        try:
            payload = json.loads(raw)
            if "hook_event_name" in payload and "hook_type" not in payload:
                payload["hook_type"] = payload["hook_event_name"]
            data = json.dumps(payload).encode("utf-8")
        except json.JSONDecodeError:
            data = raw

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(str(SOCKET_PATH))
        sock.sendall(data)
        sock.close()
    except (ConnectionRefusedError, FileNotFoundError):
        # Sidecar not running — silently ignore
        pass
    except Exception:
        # Never block CC execution
        pass
