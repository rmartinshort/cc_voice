"""Sidecar daemon: runs alongside Claude Code, narrating via hooks.

Lifecycle:
  voice-observer start  →  launch sidecar (foreground or background)
  voice-observer stop   →  send SIGTERM to running sidecar

The sidecar runs:
1. HookReceiver (Unix socket server) — receives events from CC hooks
2. VoicePipeline (mic + TTS) — captures user speech, plays narration
3. Event consumer loop — filters events, calls NarrationLLM, speaks

Unlike the SDK-driven orchestrator, this does NOT drive Claude Code.
The user runs `claude` normally in their terminal.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from voice_observer.context_mux import ContextMux
from voice_observer.event_filter import EventFilter
from voice_observer.hook_receiver import HookReceiver
from voice_observer.models import ClaudeCodeEvent
from voice_observer.narration_llm import NarrationLLM
from voice_observer.voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)

PID_FILE = Path("/tmp/voice-observer.pid")


class Sidecar:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.console = Console()
        self.receiver = HookReceiver()
        self.pipeline = VoicePipeline(console=self.console)
        self.narration = NarrationLLM()
        self.mux = ContextMux()
        self.filter = EventFilter()
        self._running = True

    async def run(self) -> None:
        """Main sidecar loop."""
        self.console.print(Rule("[bold cyan]Voice Observer — Sidecar Mode[/bold cyan]"))
        self.console.print(
            "[dim]Listening for Claude Code hook events.[/dim]\n"
            "[dim]Run [bold]claude[/bold] in another terminal to start.[/dim]\n"
            "[dim]Speak to ask questions about what CC is doing.[/dim]\n"
            "[dim]Type commands directly in the Claude Code terminal.[/dim]\n"
            "[dim]Press Ctrl+C to stop.[/dim]\n"
        )

        # Install signal handler for clean shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal)

        try:
            await self.receiver.start()
            await self.pipeline.start()

            self.console.print("[green]Ready.[/green] Waiting for events...\n")

            # Run hook consumer and voice consumer concurrently
            await asyncio.gather(
                self._consume_hook_events(),
                self._consume_voice_events(),
            )

        except asyncio.CancelledError:
            pass
        finally:
            await self.pipeline.stop()
            await self.receiver.stop()

        self.console.print(Rule())
        self.console.print("[bold cyan]Sidecar stopped.[/bold cyan]")

    def _handle_signal(self) -> None:
        self._running = False
        for task in asyncio.all_tasks():
            task.cancel()

    def _display_event(self, event: ClaudeCodeEvent) -> None:
        """Print event to the sidecar terminal."""
        if event.event_type == "tool_use":
            self.console.print(f"[dim cyan]  ▶ {escape(event.summary)}[/dim cyan]")
        elif event.event_type == "tool_output":
            content = event.raw_data.get("content", "").strip()
            if content:
                lines = content.splitlines()
                preview = "\n".join(f"    {escape(ln)}" for ln in lines[:8])
                if len(lines) > 8:
                    preview += f"\n    [dim]… ({len(lines) - 8} more lines)[/dim]"
                self.console.print(preview)
        elif event.event_type == "tool_result":
            self.console.print(
                f"[red]  ✗ {escape(event.summary[:120])}[/red]"
            )
        elif event.event_type == "complete":
            response = event.raw_data.get("response_text", "").strip()
            if response:
                # Show CC's response text (truncated for terminal)
                lines = response.splitlines()
                preview = "\n".join(f"    {escape(ln)}" for ln in lines[:15])
                if len(lines) > 15:
                    preview += f"\n    [dim]… ({len(lines) - 15} more lines)[/dim]"
                self.console.print(preview)
            self.console.print("[green]  ✓ Task complete.[/green]")
        elif event.event_type == "error":
            self.console.print(f"[red]  ✗ {escape(event.summary)}[/red]")
        elif self.debug:
            self.console.print(
                f"[dim][{event.event_type}] {escape(event.summary[:120])}[/dim]"
            )

    async def _consume_hook_events(self) -> None:
        """Pull events from hook receiver, display + narrate."""
        while self._running:
            try:
                event = await asyncio.wait_for(self.receiver.queue.get(), timeout=0.5)
            except TimeoutError:
                continue

            # Display on terminal
            if self.debug:
                self.console.print(
                    f"[dim][HOOK] type={event.event_type} "
                    f"summary={escape(event.summary[:100])} "
                    f"keys={list(event.raw_data.keys())}[/dim]"
                )
            self._display_event(event)

            # Log to context (all events)
            enriched = self.filter.enrich_summary(event)
            self.mux.log_event(enriched)

            # Track user prompt from any event that has it
            user_prompt = event.raw_data.get("user_prompt", "")
            if user_prompt:
                self.mux.set_task_prompt(user_prompt)

            # Narrate significant events
            if self.filter.should_emit(enriched):
                # Small delay to batch rapid events
                await asyncio.sleep(0.3)
                # Drain any additional events that arrived
                while not self.receiver.queue.empty():
                    try:
                        extra = self.receiver.queue.get_nowait()
                        self._display_event(extra)
                        extra_enriched = self.filter.enrich_summary(extra)
                        self.mux.log_event(extra_enriched)
                        if extra.event_type == "complete":
                            enriched = extra_enriched  # prefer complete as trigger
                    except asyncio.QueueEmpty:
                        break

                context = self.mux.get_context_snapshot()
                turn = await self.narration.narrate(enriched, context)
                self.mux.record_turn(turn)

                mode_tag = (
                    "[bold green]PROACTIVE[/bold green]"
                    if turn.mode == "proactive"
                    else "[bold yellow]REACTIVE[/bold yellow]"
                )
                self.console.print(f"{mode_tag} {escape(turn.response_text)}\n")
                await self.pipeline.speak(turn.response_text)

    async def _consume_voice_events(self) -> None:
        """Capture user speech and narrate reactively."""
        async for voice_event in self.pipeline.voice_events():
            if not self._running:
                break

            # Put in mux for context
            await self.mux.put_user_voice(voice_event)

            context = self.mux.get_context_snapshot()
            turn = await self.narration.narrate(voice_event, context)
            self.mux.record_turn(turn)

            self.console.print(
                f"[bold yellow]REACTIVE[/bold yellow] {escape(turn.response_text)}\n"
            )
            await self.pipeline.speak(turn.response_text)


def write_pid() -> None:
    PID_FILE.write_text(str(os.getpid()))


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def remove_pid() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def start_sidecar(debug: bool = False, foreground: bool = True) -> None:
    """Launch the sidecar process."""
    existing = read_pid()
    if existing:
        # Check if process is actually running
        try:
            os.kill(existing, 0)
            print(
                f"Sidecar already running (PID {existing}). Use 'voice-observer stop' first."
            )
            sys.exit(1)
        except OSError:
            remove_pid()  # Stale PID file

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    write_pid()
    sidecar = Sidecar(debug=debug)
    try:
        asyncio.run(sidecar.run())
    finally:
        remove_pid()


def stop_sidecar() -> None:
    """Stop a running sidecar process."""
    pid = read_pid()
    if not pid:
        print("No sidecar running.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to sidecar (PID {pid}).")
    except OSError as exc:
        print(f"Could not stop sidecar (PID {pid}): {exc}")
    finally:
        remove_pid()
