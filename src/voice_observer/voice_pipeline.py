"""Voice pipeline: ElevenLabs TTS output + faster-whisper STT input + barge-in.

Architecture:
- Background thread continuously records mic audio
- Energy-based VAD detects speech onset/offset
- Typed sentinel tuples are enqueued so the async side gets real-time feedback:
    ("start",)              — VAD detected speech onset
    ("partial", bytes)      — intermediate audio every ~2s for streaming display
    ("final", bytes)        — utterance complete
- Transcription runs in asyncio thread pool (faster-whisper)
- voice_events() uses Rich Live for streaming STT display
- capture_single_utterance() waits for one complete utterance (for task prompts)
- Barge-in: if speech detected while TTS is playing, TTS is cancelled immediately
- Echo cancellation: mic is gated briefly after TTS ends

Headphones strongly recommended to avoid TTS audio triggering barge-in.
"""

import asyncio
import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pyaudio
from rich.console import Console
from rich.live import Live
from rich.markup import escape

from voice_observer.models import UserVoiceEvent

logger = logging.getLogger(__name__)

# Audio capture constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1024  # ~64ms per chunk at 16kHz
BYTES_PER_SAMPLE = 2

# Energy-based VAD thresholds
SPEECH_RMS_THRESHOLD = 800
SILENCE_CHUNKS_END = 20  # ~1.3s silence to close an utterance
MIN_SPEECH_CHUNKS = 6  # ~0.4s minimum utterance length
PRE_ROLL_CHUNKS = 3

# Streaming display: emit a partial every N speech chunks (~2s)
PARTIAL_EVERY_CHUNKS = 30

# Echo cancellation: gate mic this many seconds after TTS ends
ECHO_GATE_SECONDS = 1.0

# Sentinel tuple types
_QItem = tuple[Any, ...]


class VoicePipeline:
    """Real voice pipeline: ElevenLabs TTS + faster-whisper STT + barge-in."""

    def __init__(self, console: Console | None = None, barge_in: bool = False) -> None:
        self._console = console or Console()
        self._barge_in = barge_in
        self._pa: pyaudio.PyAudio | None = None
        self._whisper = None
        self._el_client = None
        self._voice_id: str = ""

        # Thread → asyncio communication
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: asyncio.Queue[_QItem] = asyncio.Queue()

        # TTS state
        self._tts_playing = threading.Event()
        self._tts_cancel = threading.Event()
        self._echo_gate_until: float = 0.0

        # Keyboard barge-in
        self._keyboard_barge = threading.Event()
        self._keyboard_thread: threading.Thread | None = None

        # Recording thread lifecycle
        self._stop_recording = threading.Event()
        self._recording_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        api_key = os.environ.get("ELEVENLABS_API_KEY", "")
        self._voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "")
        if api_key and self._voice_id:
            from elevenlabs import ElevenLabs

            self._el_client = ElevenLabs(api_key=api_key)
            logger.info("ElevenLabs TTS configured (voice=%s)", self._voice_id)
        else:
            logger.info("ElevenLabs not configured — using macOS say for TTS")

        model_size = os.environ.get("WHISPER_MODEL", "base.en")
        logger.info("Loading Whisper model '%s' …", model_size)
        from faster_whisper import WhisperModel

        self._whisper = await asyncio.to_thread(
            WhisperModel, model_size, device="cpu", compute_type="int8"
        )
        logger.info("Whisper model loaded")

        self._pa = pyaudio.PyAudio()
        self._stop_recording.clear()
        self._recording_thread = threading.Thread(
            target=self._recording_loop, daemon=True, name="voice-recorder"
        )
        self._recording_thread.start()

        # Start keyboard listener for barge-in (press 'b')
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="keyboard-listener"
        )
        self._keyboard_thread.start()
        logger.info("Voice pipeline ready (press 'b' to interrupt narrator)")

    async def stop(self) -> None:
        self._tts_cancel.set()
        self._stop_recording.set()
        if self._recording_thread:
            self._recording_thread.join(timeout=3.0)
        if self._pa:
            self._pa.terminate()
        logger.info("Voice pipeline stopped")

    # ------------------------------------------------------------------
    # TTS output
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Convert text to speech and play. Blocks until done or barge-in."""
        self._tts_cancel.clear()
        self._tts_playing.set()
        try:
            await asyncio.to_thread(self._speak_blocking, text)
        finally:
            self._tts_playing.clear()
            self._echo_gate_until = time.monotonic() + ECHO_GATE_SECONDS

    def _speak_blocking(self, text: str) -> None:
        if self._el_client and self._voice_id:
            if self._speak_elevenlabs(text):
                return
        self._speak_macos_say(text)

    def _speak_elevenlabs(self, text: str) -> bool:
        try:
            audio_iter = self._el_client.text_to_speech.convert(
                voice_id=self._voice_id,
                text=text,
                model_id="eleven_flash_v2_5",
                output_format="pcm_22050",
            )
        except Exception as exc:
            logger.warning("ElevenLabs TTS unavailable (%s), falling back to say", exc)
            return False

        # Play at ~1.15x speed by using a higher sample rate than the source
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=25350,
            output=True,
            frames_per_buffer=2048,
        )
        try:
            for chunk in audio_iter:
                if self._tts_cancel.is_set():
                    break
                if chunk:
                    stream.write(chunk)
        except Exception as exc:
            logger.warning("ElevenLabs playback error: %s", exc)
            return False
        finally:
            stream.stop_stream()
            stream.close()
        return True

    def _speak_macos_say(self, text: str) -> None:
        import subprocess

        clean = text.replace("`", "").replace("*", "").replace("#", "")
        try:
            proc = subprocess.Popen(
                ["say", "-v", "Samantha", "-r", "210", clean],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while proc.poll() is None:
                if self._tts_cancel.is_set():
                    proc.terminate()
                    return
                time.sleep(0.05)
        except Exception as exc:
            logger.error("macOS say error: %s", exc)

    # ------------------------------------------------------------------
    # STT input — streaming display via voice_events()
    # ------------------------------------------------------------------

    async def voice_events(self) -> AsyncIterator[UserVoiceEvent]:
        """Continuously yield UserVoiceEvent, updating terminal with streaming STT."""
        with Live(
            "", console=self._console, refresh_per_second=8, transient=False
        ) as live:
            while True:
                item: _QItem = await self._audio_queue.get()
                kind = item[0]

                if kind == "start":
                    live.update("[dim]🎤 Listening...[/dim]")

                elif kind == "partial":
                    partial = await asyncio.to_thread(self._transcribe, item[1])
                    if partial:
                        live.update(f"[dim]🎤 {escape(partial[:80])}…[/dim]")

                elif kind == "final":
                    transcript = (
                        await asyncio.to_thread(self._transcribe, item[1])
                    ).strip()
                    if transcript:
                        live.update(
                            f"[bold magenta][YOU][/bold magenta] {escape(transcript)}"
                        )
                        yield UserVoiceEvent(transcript=transcript)
                        live.update("")  # reset for next utterance

    async def capture_single_utterance(self) -> str | None:
        """Block until user speaks one complete utterance. Shows streaming feedback.

        Used to capture task prompts before starting Claude Code.
        """
        while True:
            item: _QItem = await self._audio_queue.get()
            kind = item[0]

            if kind == "start":
                self._console.print("[dim]🎤 Listening...[/dim]", end="\r")

            elif kind == "partial":
                partial = await asyncio.to_thread(self._transcribe, item[1])
                if partial:
                    # Pad to overwrite previous line
                    line = f"[dim]🎤 {escape(partial[:80])}…[/dim]"
                    self._console.print(line + " " * 10, end="\r")

            elif kind == "final":
                transcript = (
                    await asyncio.to_thread(self._transcribe, item[1])
                ).strip()
                if transcript:
                    self._console.print(
                        f"\n[bold magenta][YOU][/bold magenta] {escape(transcript)}"
                    )
                    return transcript

    # ------------------------------------------------------------------
    # Keyboard barge-in
    # ------------------------------------------------------------------

    def cancel_tts(self) -> None:
        """Cancel current TTS playback (called from orchestrator on keyboard barge)."""
        self._tts_cancel.set()

    def check_keyboard_barge(self) -> bool:
        """Check and clear the keyboard barge-in flag. Returns True if 'b' was pressed."""
        if self._keyboard_barge.is_set():
            self._keyboard_barge.clear()
            return True
        return False

    def _keyboard_loop(self) -> None:
        """Background thread: listen for 'b' keypress to trigger barge-in."""
        if not sys.stdin.isatty():
            return

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_recording.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "b":
                        logger.debug("Keyboard barge-in")
                        self._keyboard_barge.set()
                        self._tts_cancel.set()
        except Exception:
            logger.exception("Keyboard listener error")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # ------------------------------------------------------------------
    # Transcription (thread pool)
    # ------------------------------------------------------------------

    def _transcribe(self, audio_bytes: bytes) -> str:
        audio_f32 = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = self._whisper.transcribe(
            audio_f32, language="en", vad_filter=True
        )
        return " ".join(seg.text for seg in segments)

    # ------------------------------------------------------------------
    # Recording loop (background thread)
    # ------------------------------------------------------------------

    def _recording_loop(self) -> None:
        stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

        pre_roll: list[bytes] = []
        buffer: list[bytes] = []
        silence_count = 0
        in_speech = False
        speech_onset_emitted = False

        try:
            while not self._stop_recording.is_set():
                data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
                rms = _rms(data)

                # While TTS is playing: gate mic entirely (unless barge-in enabled)
                if self._tts_playing.is_set():
                    if self._barge_in and rms > SPEECH_RMS_THRESHOLD:
                        logger.debug("Barge-in (rms=%d)", rms)
                        self._tts_cancel.set()
                    # Either way, don't record during TTS
                    pre_roll = []
                    buffer = []
                    in_speech = False
                    silence_count = 0
                    speech_onset_emitted = False
                    continue

                # Echo gate: ignore mic briefly after TTS ends
                if time.monotonic() < self._echo_gate_until:
                    pre_roll = []
                    buffer = []
                    in_speech = False
                    silence_count = 0
                    speech_onset_emitted = False
                    continue

                if rms > SPEECH_RMS_THRESHOLD:
                    if not in_speech:
                        in_speech = True
                        speech_onset_emitted = False
                        buffer = list(pre_roll)

                    if not speech_onset_emitted:
                        self._enqueue(("start",))
                        speech_onset_emitted = True

                    buffer.append(data)
                    silence_count = 0

                    # Emit partial for streaming display every PARTIAL_EVERY_CHUNKS
                    if len(buffer) % PARTIAL_EVERY_CHUNKS == 0:
                        self._enqueue(("partial", b"".join(buffer)))

                else:
                    if in_speech:
                        buffer.append(data)
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS_END:
                            if len(buffer) >= MIN_SPEECH_CHUNKS:
                                self._enqueue(("final", b"".join(buffer)))
                            buffer = []
                            silence_count = 0
                            in_speech = False
                            speech_onset_emitted = False

                    pre_roll.append(data)
                    if len(pre_roll) > PRE_ROLL_CHUNKS:
                        pre_roll.pop(0)

        except Exception:
            logger.exception("Recording loop error")
        finally:
            stream.stop_stream()
            stream.close()

    def _enqueue(self, item: _QItem) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._audio_queue.put_nowait, item)


def _rms(data: bytes) -> float:
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr**2))) if len(arr) else 0.0
