"""CLI entry point for Voice Observer."""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import anyio
from dotenv import load_dotenv

_HOOK_EVENTS = ["PreToolUse", "PostToolUse", "PostToolUseFailure", "Stop"]


def _make_hook_entry() -> dict:
    """Build a hook entry with the full path to voice-observer-hook."""
    # First try PATH
    hook_cmd = shutil.which("voice-observer-hook")
    if not hook_cmd:
        # Fall back to sibling of the current Python executable
        # (works when voice-observer is installed in a venv)
        bin_dir = Path(sys.executable).parent
        candidate = bin_dir / "voice-observer-hook"
        if candidate.exists():
            hook_cmd = str(candidate)
    if not hook_cmd:
        print(
            "Error: voice-observer-hook not found.\n"
            "Make sure the voice-observer package is installed.",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": hook_cmd,
                "timeout": 2000,
            }
        ],
    }


def cli() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Voice Observer — voice narration layer for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  setup       Install Claude Code hooks (one-time)
  start       Launch the voice observer sidecar
  stop        Stop the running sidecar
  teardown    Remove Claude Code hooks
  run         Legacy SDK-driven mode (drives CC directly)

Examples:
  voice-observer setup
  voice-observer start
  voice-observer start --debug
  voice-observer stop
  voice-observer teardown
  voice-observer run "write a Python CSV parser" --cwd /path/to/project
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # setup
    setup_p = sub.add_parser("setup", help="Install Claude Code hooks")
    setup_p.add_argument(
        "--cwd",
        default=".",
        help="Project directory where .claude/settings.local.json lives",
    )

    # start
    start_p = sub.add_parser("start", help="Launch the sidecar")
    start_p.add_argument("--debug", action="store_true", help="Verbose logging")

    # stop
    sub.add_parser("stop", help="Stop the running sidecar")

    # teardown
    teardown_p = sub.add_parser("teardown", help="Remove Claude Code hooks")
    teardown_p.add_argument(
        "--cwd",
        default=".",
        help="Project directory where .claude/settings.local.json lives",
    )

    # run (legacy SDK-driven mode)
    run_p = sub.add_parser("run", help="Legacy SDK-driven mode")
    run_p.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Initial task for Claude Code (optional — speak it if omitted)",
    )
    run_p.add_argument("--debug", action="store_true", help="Print raw SDK events")
    run_p.add_argument(
        "--barge-in",
        action="store_true",
        help="Enable barge-in (requires headphones)",
    )
    run_p.add_argument(
        "--cwd",
        default=None,
        help="Working directory for Claude Code (default: current directory)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "setup":
        _cmd_setup(args.cwd)
    elif args.command == "start":
        _cmd_start(args.debug)
    elif args.command == "stop":
        _cmd_stop()
    elif args.command == "teardown":
        _cmd_teardown(args.cwd)
    elif args.command == "run":
        _cmd_run(args)


def _cmd_setup(cwd: str) -> None:
    """Install hooks into .claude/settings.local.json."""
    settings_path = Path(cwd) / ".claude" / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    else:
        settings = {}

    hook_entry = _make_hook_entry()
    hooks = settings.setdefault("hooks", {})
    for event_name in _HOOK_EVENTS:
        existing = hooks.get(event_name, [])
        # Check if we already installed our hook
        already = any(
            any(
                "voice-observer-hook" in h.get("command", "")
                for h in entry.get("hooks", [])
            )
            for entry in existing
        )
        if not already:
            existing.append(hook_entry)
            hooks[event_name] = existing

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Hooks installed in {settings_path}")
    print(
        "Run 'voice-observer start' to launch the sidecar, then use Claude Code normally."
    )


def _cmd_start(debug: bool) -> None:
    """Launch the sidecar process."""
    from voice_observer.sidecar import start_sidecar

    start_sidecar(debug=debug)


def _cmd_stop() -> None:
    """Stop the running sidecar."""
    from voice_observer.sidecar import stop_sidecar

    stop_sidecar()


def _cmd_teardown(cwd: str) -> None:
    """Remove hooks from .claude/settings.local.json."""
    settings_path = Path(cwd) / ".claude" / "settings.local.json"
    if not settings_path.exists():
        print("No settings file found — nothing to remove.")
        return

    settings = json.loads(settings_path.read_text())
    hooks = settings.get("hooks", {})
    changed = False

    for event_name in _HOOK_EVENTS:
        entries = hooks.get(event_name, [])
        filtered = [
            entry
            for entry in entries
            if not any(
                "voice-observer-hook" in h.get("command", "")
                for h in entry.get("hooks", [])
            )
        ]
        if len(filtered) != len(entries):
            changed = True
            if filtered:
                hooks[event_name] = filtered
            else:
                del hooks[event_name]

    if changed:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Hooks removed from {settings_path}")
    else:
        print("No voice-observer hooks found — nothing to remove.")


def _cmd_run(args: argparse.Namespace) -> None:
    """Legacy SDK-driven mode."""
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from voice_observer.orchestrator import Orchestrator

    orchestrator = Orchestrator(
        prompt=args.prompt,
        debug=args.debug,
        cwd=args.cwd,
        barge_in=args.barge_in,
    )
    try:
        anyio.run(orchestrator.run)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).exception("Fatal error")
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
