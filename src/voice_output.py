#!/usr/bin/env python3
"""
voice_output.py — Piper neural TTS output routed to PipeWire Bluetooth sink.

Your Bluetooth sink (from pactl list short sinks):
    <id>  bluez_output.<MAC>.1   PipeWire   s16le 2ch 48000Hz   <headset name>
    <id>  bluez_output.<MAC>.1   PipeWire   s16le 2ch 48000Hz   <speaker name>


Piper outputs raw 22050 Hz S16LE PCM.  We pipe it through `aplay` with PipeWire's
virtual ALSA device, OR use `paplay` which speaks PipeWire natively.

Strategy (in priority order):
  1. `paplay --device=<sink>` — cleanest for PipeWire
  2. `pw-play`                — PipeWire native player
  3. `aplay -D default`       — last resort ALSA fallback

Stop mechanism:
  stop() sets stop_requested=True and kills _player_proc immediately.
  The write loop in speak() also checks stop_requested every 4KB chunk.
"""

import subprocess
import os
import threading
import shutil


# ── PipeWire Bluetooth sink ───────────────────────────────────────────────────
# The sink is auto-detected at runtime (see _detect_bt_sink). To pin a specific
# device, find its name with `pactl list short sinks` and set it here.
BT_SINK_NAME = ""


# Piper TTS sample rate
PIPER_SR    = 22_050
PIPER_FMT   = "S16_LE"
PIPER_CHANS = 1


def _find_player():
    """Return the best available audio player command tuple."""
    if shutil.which("paplay"):
        return "paplay"
    if shutil.which("pw-play"):
        return "pw-play"
    return "aplay"


class VoiceOutput:
    """
    Piper TTS → PipeWire Bluetooth speaker.

    speaking flag is managed at the response level by output_voice() in
    chat_interface, NOT per-sentence here, so the stop-listener stays alive
    across a full multi-sentence response.
    """

    def __init__(self, logger, engine_type="piper", rate=175, volume=0.9):
        self.logger         = logger
        self.engine_type    = engine_type
        self.speaking       = False
        self.stop_requested = False
        self._player_proc   = None
        self._piper_proc    = None
        self._proc_lock     = threading.Lock()
        self._player_cmd    = _find_player()

        # Allow GUI to override output device index (stored but used only if paplay unavailable)
        self._preferred_output_device = None

        if engine_type == "piper":
            self._init_piper()
        else:
            raise ValueError("Only 'piper' engine supported")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_piper(self):
        self.logger.info("Initialising Piper neural TTS…")
        self.model_path = os.path.expanduser(
            "~/piper_models/en_US-ryan-medium.onnx"
        )
        if not os.path.exists(self.model_path):
            self.logger.error("Piper model not found at ~/piper_models/en_US-ryan-medium.onnx")
            raise RuntimeError("Download model: en_US-ryan-medium.onnx")

        # Verify Bluetooth sink is available
        self._bt_sink = self._detect_bt_sink()
        self.logger.info(
            f"✓ Piper TTS ready | player={self._player_cmd} | sink={self._bt_sink}"
        )

    def _detect_bt_sink(self) -> str:
        """
        Auto-detect any connected Bluetooth sink — no hardcoded MAC needed.
        Scans pactl for any sink starting with 'bluez_output.' and returns it.
        Falls back to @DEFAULT_SINK@ if none is connected.
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("bluez_output."):
                    self.logger.info(f"Auto-detected BT sink: {parts[1]}")
                    return parts[1]
            self.logger.warn("No Bluetooth sink found — is the headset connected? Falling back to default.")
        except Exception as e:
            self.logger.warn(f"pactl check failed: {e} — using default sink")
        return "@DEFAULT_SINK@"

    # ── Core speak ────────────────────────────────────────────────────────────

    def speak(self, text: str, blocking=True, stop_check_callback=None) -> bool:
        """
        Synthesise one sentence with Piper and play on the Bluetooth sink.
        Returns True on success, False if stopped or error.
        """
        if not text or not text.strip():
            return False
        if self.stop_requested:
            return False

        try:
            # ── Step 1: Piper → raw PCM ───────────────────────────────────────
            piper = subprocess.Popen(
                ["piper", "--model", self.model_path, "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            with self._proc_lock:
                self._piper_proc = piper

            try:
                audio, _ = piper.communicate(input=text.encode(), timeout=20)
            except subprocess.TimeoutExpired:
                piper.kill()
                return False
            finally:
                with self._proc_lock:
                    self._piper_proc = None

            if self.stop_requested:
                return False

            # ── Step 2: play PCM via PipeWire ────────────────────────────────
            # Bluetooth A2DP sinks need ~200-300 ms to negotiate the A2DP stream.
            # Bytes sent during that window are silently discarded, clipping the
            # first word.  Prepend silence only when this is the first sentence
            # of a new response (flagged by _prepend_silence).
            if getattr(self, '_prepend_silence', True):
                SILENCE_MS  = 280
                silence     = bytes(int(PIPER_SR * SILENCE_MS / 1000) * 2)  # S16LE mono
                audio       = silence + audio
                self._prepend_silence = False   # subsequent sentences: no padding

            player = self._build_player_cmd(len(audio))
            proc = subprocess.Popen(
                player,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            with self._proc_lock:
                self._player_proc = proc

            # Stream in 4 KB chunks so stop() cuts audio within ~0.09 s
            CHUNK  = 4096
            offset = 0
            while offset < len(audio):
                if self.stop_requested:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    with self._proc_lock:
                        self._player_proc = None
                    return False
                end = min(offset + CHUNK, len(audio))
                try:
                    proc.stdin.write(audio[offset:end])
                except BrokenPipeError:
                    break
                offset = end

            try:
                proc.stdin.close()
                proc.wait(timeout=10)
            except Exception:
                pass
            with self._proc_lock:
                self._player_proc = None
            return True

        except Exception as e:
            self.logger.error(f"Piper speak error: {e}")
            return False

    def _build_player_cmd(self, audio_len: int) -> list:
        """Build the audio player command for raw PCM from Piper."""
        sink = self._bt_sink

        if self._player_cmd == "paplay":
            # paplay accepts raw PCM with explicit format flags
            return [
                "paplay",
                f"--device={sink}",
                "--raw",
                f"--rate={PIPER_SR}",
                "--format=s16le",
                f"--channels={PIPER_CHANS}",
            ]

        elif self._player_cmd == "pw-play":
            return [
                "pw-play",
                f"--target={sink}",
                "--format=s16",
                f"--rate={PIPER_SR}",
                f"--channels={PIPER_CHANS}",
                "-",   # read from stdin
            ]

        else:
            # aplay fallback — uses ALSA default (may not hit Bluetooth)
            return [
                "aplay",
                "-D", "default",
                "-r", str(PIPER_SR),
                "-f", PIPER_FMT,
                "-c", str(PIPER_CHANS),
                "-t", "raw",
            ]

    # ── Stop ──────────────────────────────────────────────────────────────────

    def stop(self):
        """Kill playback immediately. Thread-safe."""
        self.stop_requested = True
        with self._proc_lock:
            player = self._player_proc
            piper  = self._piper_proc
        for p in (player, piper):
            try:
                if p and p.poll() is None:
                    p.kill()
            except Exception:
                pass
        # Belt-and-suspenders
        subprocess.run(["pkill", "-f", "paplay"],  stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "pw-play"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "aplay"],   stderr=subprocess.DEVNULL)
        self.speaking = False

    def request_stop(self):
        self.stop()

    # ── Async helper ──────────────────────────────────────────────────────────

    def speak_async(self, text: str):
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()

    def is_speaking(self) -> bool:
        return self.speaking

    def set_rate(self, rate):
        return True

    def set_volume(self, volume):
        return True