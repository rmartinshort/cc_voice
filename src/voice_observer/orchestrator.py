"""Top-level async orchestration loop.

Flow:
1. Start voice pipeline (loads Whisper, opens mic + TTS)
2. Loop:
   a. If no prompt given (voice-first mode): listen for spoken task
   b. Run Claude Code task with that prompt (3 concurrent tasks: CC producer,
      voice producer, consumer/narrator)
   c. On completion, speak "Task complete. What would you like to do next?"
   d. Go back to (a)
3. Ctrl+C exits cleanly

Session continuity: session_id from ResultMessage is passed as `resume` to the
next ClaudeCodeOptions so Claude Code retains context across spoken tasks.
"""

import logging
import os
import re

import anyio
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from voice_observer.claude_stream import stream_claude_code_events
from voice_observer.context_mux import ContextMux
from voice_observer.event_filter import EventFilter, summarize_tool_use
from voice_observer.models import ClaudeCodeEvent, UserVoiceEvent
from voice_observer.narration_llm import NarrationLLM
from voice_observer.voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)

_STOP_PATTERN = re.compile(
    r"^\s*(stop|cancel|abort|quit|halt|stop that|stop it|cancel that|never\s*mind)\s*[.!]?\s*$",
    re.IGNORECASE,
)


class Orchestrator:
    def __init__(
        self,
        prompt: str | None = None,
        debug: bool = False,
        cwd: str | None = None,
        barge_in: bool = False,
    ) -> None:
        self._initial_prompt = prompt
        self.debug = debug
        self.cwd = os.path.abspath(cwd) if cwd else os.getcwd()

        self.console = Console()
        self.pipeline = VoicePipeline(console=self.console, barge_in=barge_in)
        self.narration = NarrationLLM()

        # Reset per task
        self.mux: ContextMux | None = None
        self.filter: EventFilter | None = None
        self._done: anyio.Event | None = None
        self._tg: anyio.abc.TaskGroup | None = None
        self._session_id: str | None = None  # for Claude Code session continuity

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.console.print(Rule("[bold cyan]Voice Observer[/bold cyan]"))
        self.console.print(f"[dim]Working directory:[/dim] {self.cwd}\n")

        try:
            await self.pipeline.start()
        except Exception as exc:
            self.console.print(f"[red]Failed to start voice pipeline: {exc}[/red]")
            raise

        prompt = self._initial_prompt

        try:
            while True:
                if prompt is None:
                    self.console.print(
                        "\n[bold cyan]Listening for your task…[/bold cyan] "
                        "[dim](speak now, or Ctrl+C to quit)[/dim]\n"
                    )
                    prompt = await self.pipeline.capture_single_utterance()
                    if not prompt:
                        continue

                self.console.print(f"\n[bold]Task:[/bold] {escape(prompt)}")
                self.console.print(Rule())

                await self._run_task(prompt)

                # Between tasks: speak prompt and listen for next task
                await self.pipeline.speak(
                    "Task complete. What would you like to do next?"
                )
                prompt = None

        except (KeyboardInterrupt, anyio.get_cancelled_exc_class()):
            pass
        finally:
            await self.pipeline.stop()

        self.console.print(Rule())
        self.console.print("[bold cyan]Session ended.[/bold cyan]")

    # ------------------------------------------------------------------
    # Single Claude Code task
    # ------------------------------------------------------------------

    async def _run_task(self, prompt: str) -> None:
        """Orchestrate one Claude Code task end-to-end."""
        # Carry over context from previous task so narrator retains history
        prior_log = self.mux.event_log if self.mux else None
        prior_conv = self.mux.conversation if self.mux else None
        self.mux = ContextMux(
            prior_event_log=prior_log,
            prior_conversation=prior_conv,
        )
        self.mux.set_task_prompt(prompt)
        self.filter = EventFilter()
        self._done = anyio.Event()
        self._tg = None

        async with anyio.create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._produce_claude_events, prompt)
            tg.start_soon(self._produce_voice_events)
            tg.start_soon(self._consume_events)

    # ------------------------------------------------------------------
    # Task A — Claude Code event producer
    # ------------------------------------------------------------------

    async def _produce_claude_events(self, prompt: str) -> None:
        raw_stream = stream_claude_code_events(
            prompt, cwd=self.cwd, session_id=self._session_id
        )
        async for event in raw_stream:
            # Always display on terminal (raw activity)
            self._display_raw_event(event)

            # Capture session_id for continuity across tasks
            if event.event_type == "complete":
                sid = event.raw_data.get("session_id")
                if sid:
                    self._session_id = sid

            # Enrich summary for all events
            enriched = self.filter.enrich_summary(event)

            # ALL events go into context log so narrator knows what happened
            self.mux.log_event(enriched)

            # Only narratively significant events get queued for narration
            if self.filter.should_emit(enriched):
                await self.mux.put_claude_event(enriched)

        self._done.set()

    def _display_raw_event(self, event: ClaudeCodeEvent) -> None:
        """Print the raw event to terminal so user always sees what Claude Code is doing."""
        if event.event_type == "tool_use":
            name = event.raw_data.get("name", "")
            inp = event.raw_data.get("input", {})
            label = summarize_tool_use(name, inp)
            self.console.print(f"[dim cyan]  ▶ {escape(label)}[/dim cyan]")

        elif event.event_type == "tool_output":
            content = event.raw_data.get("content", "").strip()
            if content:
                # Show first few lines of output, truncated
                lines = content.splitlines()
                preview_lines = lines[:8]
                preview = "\n".join(f"    {escape(ln)}" for ln in preview_lines)
                if len(lines) > 8:
                    preview += f"\n    [dim]… ({len(lines) - 8} more lines)[/dim]"
                self.console.print(preview)

        elif event.event_type == "tool_result":
            # Error result
            self.console.print(
                f"[red]  ✗ Tool error: {escape(event.summary[:120])}[/red]"
            )

        elif event.event_type == "text":
            text = event.raw_data.get("text", "").strip()
            if text:
                # Show full assistant text (code explanations, etc.)
                self.console.print(f"  {escape(text)}")

        elif event.event_type == "complete":
            self.console.print(f"[green]  ✓ {escape(event.summary)}[/green]")

        elif event.event_type == "error":
            self.console.print(f"[red]  ✗ {escape(event.summary)}[/red]")

        elif self.debug:
            self.console.print(
                f"[dim][{event.event_type}] {escape(event.summary[:120])}[/dim]"
            )

    # ------------------------------------------------------------------
    # Task B — User voice producer
    # ------------------------------------------------------------------

    async def _produce_voice_events(self) -> None:
        async for event in self.pipeline.voice_events():
            await self.mux.put_user_voice(event)

    # ------------------------------------------------------------------
    # Task C — Consumer / narrator
    # ------------------------------------------------------------------

    async def _consume_events(self) -> None:
        while True:
            if self._done.is_set() and self.mux.empty():
                break

            # Wait for at least one event
            try:
                with anyio.fail_after(0.5):
                    first = await self.mux.get()
            except TimeoutError:
                continue

            # Small delay to let more events accumulate, then drain
            await anyio.sleep(0.3)
            extra = self.mux.drain_nowait()
            events = [first] + extra

            # Separate voice events (priority 0) and claude events
            voice_events = [e for e in events if isinstance(e, UserVoiceEvent)]
            claude_events = [e for e in events if isinstance(e, ClaudeCodeEvent)]

            # Check for stop commands first
            for ve in voice_events:
                if _STOP_PATTERN.match(ve.transcript):
                    self.console.print("[bold red]  ■ Stopping task…[/bold red]")
                    await self.pipeline.speak("Stopping.")
                    if self._tg is not None:
                        self._tg.cancel_scope.cancel()
                    return

            # Handle voice events (reactive) — each gets its own narration
            for ve in voice_events:
                context = self.mux.get_context_snapshot()
                turn = await self.narration.narrate(ve, context)
                self.mux.record_turn(turn)
                self.console.print(
                    f"[bold yellow]REACTIVE[/bold yellow] {escape(turn.response_text)}\n"
                )
                await self._speak_interruptible(turn.response_text)

            # Handle claude events as a batch — narrate the most recent one
            # (context snapshot already includes all events via log_event)
            saw_complete = any(e.event_type == "complete" for e in claude_events)

            if claude_events:
                # Pick the most interesting event to narrate (complete > tool_use > thinking)
                trigger = claude_events[-1]  # default to last
                for e in claude_events:
                    if e.event_type == "complete":
                        trigger = e
                        break

                context = self.mux.get_context_snapshot()
                turn = await self.narration.narrate(trigger, context)
                self.mux.record_turn(turn)
                self.console.print(
                    f"[bold green]PROACTIVE[/bold green] {escape(turn.response_text)}\n"
                )
                await self._speak_interruptible(turn.response_text)

            if saw_complete:
                break

        if self._tg is not None:
            self._tg.cancel_scope.cancel()

    async def _speak_interruptible(self, text: str) -> None:
        """Speak text, but stop if keyboard barge-in ('b') is pressed."""
        self.pipeline.check_keyboard_barge()  # clear any stale flag
        await self.pipeline.speak(text)
        if self.pipeline.check_keyboard_barge():
            self.console.print("[dim](interrupted — listening)[/dim]")
