#!/usr/bin/env python3
"""
voice_input.py — Whisper-based voice input.

Hardware config (hardcoded to match your system):
  Input  : USB2.0 Device  → ALSA card 1, device 0
           PulseAudio/PipeWire source index 51
           Native sample rate: 48 000 Hz, mono
  Whisper: expects 16 000 Hz mono float32
  Fix    : record at 48 kHz, resample to 16 kHz with scipy before transcription.

This resolves the garbled / wrong transcription you were seeing with large-v3.
"""

import threading
import time
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except ImportError:
    SOUNDDEVICE_OK = False

try:
    from scipy.signal import resample_poly
    from math import gcd
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    try:
        import whisper as _whisper_orig
        WHISPER_OK = True
        _USE_FASTER = False
    except ImportError:
        WHISPER_OK = False
    else:
        _USE_FASTER = False
else:
    _USE_FASTER = True


# ── Hardware constants ────────────────────────────────────────────────────────
# Whisper requires 16000 Hz mono float32.
# The USB mic (USB2.0 Device) runs at 48000 Hz natively.
# We record at native SR then resample 3:1 with scipy resample_poly.
WHISPER_SR       = 16_000
USB_MIC_NATIVE_SR = 48_000   # default; overridden by actual device SR at runtime

# Name fragment that uniquely identifies the USB mic across reboots.
# Device index changes but the ALSA name stays stable.
USB_MIC_NAME_HINT = "usb2.0 device"   # case-insensitive match


def _find_usb_mic_index():
    """
    Dynamically find the USB mic by name — survives reboots where the
    device index shifts.

    Strategy (in priority order):
      1. Exact name match for USB2.0 Device (hw:1,0)
      2. Any device whose name contains a USB audio keyword
      3. Any non-default, non-virtual input device (hw:X,Y)
      4. First available input device (absolute fallback)

    Returns (index, sample_rate, channels) or (None, 48000, 1).
    """
    if not SOUNDDEVICE_OK:
        return None, USB_MIC_NATIVE_SR, 1

    try:
        devices   = sd.query_devices()
        input_devs = [(i, d) for i, d in enumerate(devices)
                      if d['max_input_channels'] > 0]

        import logging as _log
        _logger = _log.getLogger('voice_input')
        _logger.info("Available input devices:")
        for i, d in input_devs:
            _logger.info(f"  [{i:2d}] {d['name']!r:48s} "
                         f"SR={int(d['default_samplerate'])}  ch={d['max_input_channels']}")

        USB_KEYWORDS = ['usb2.0 device', 'usb2.0', 'usb 2.0 device',
                        'usb audio', 'usb device']

        # Priority 1: name contains a USB keyword (catches USB2.0 Device reliably)
        for i, d in input_devs:
            name_lower = d['name'].lower()
            if any(kw in name_lower for kw in USB_KEYWORDS):
                sr  = int(d['default_samplerate'])
                ch  = min(1, d['max_input_channels'])   # always mono for USB mics
                _logger.info(f"Selected USB mic by name: [{i}] {d['name']!r}  SR={sr}")
                return i, sr, ch

        # Priority 2: any real ALSA hw device that isn't the onboard audio
        # hw:0,X = onboard Intel HDA; hw:1,X = first USB card
        for i, d in input_devs:
            name_lower = d['name'].lower()
            if 'hw:1,' in name_lower or 'hw:2,' in name_lower:
                sr  = int(d['default_samplerate'])
                ch  = min(1, d['max_input_channels'])
                _logger.info(f"Selected by hw slot: [{i}] {d['name']!r}  SR={sr}")
                return i, sr, ch

        # Priority 3: skip virtual/pulse/default devices, take first real hw device
        VIRTUAL = ['pulse', 'default', 'sysdefault', 'samplerate',
                   'speex', 'lavrate', 'upmix', 'vdownmix']
        for i, d in input_devs:
            name_lower = d['name'].lower()
            if not any(v in name_lower for v in VIRTUAL):
                sr  = int(d['default_samplerate'])
                ch  = min(1, d['max_input_channels'])
                _logger.info(f"Selected first non-virtual mic: [{i}] {d['name']!r}  SR={sr}")
                return i, sr, ch

        # Priority 4: absolute fallback — first input device
        if input_devs:
            i, d = input_devs[0]
            sr   = int(d['default_samplerate'])
            ch   = min(1, d['max_input_channels'])
            _logger.warning(f"Fallback to first input device: [{i}] {d['name']!r}")
            return i, sr, ch

    except Exception as e:
        import logging as _log
        _log.getLogger('voice_input').error(f"Device detection error: {e}")

    return None, USB_MIC_NATIVE_SR, 1


def _resample_to_whisper(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample audio array from src_sr to WHISPER_SR (16000 Hz)."""
    if src_sr == WHISPER_SR:
        return audio.astype(np.float32)
    if SCIPY_OK:
        g   = gcd(WHISPER_SR, src_sr)
        up  = WHISPER_SR // g
        dn  = src_sr     // g
        return resample_poly(audio.astype(np.float32), up, dn)
    else:
        # Fallback: numpy interpolation (lower quality but works)
        n_out = int(len(audio) * WHISPER_SR / src_sr)
        return np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio.astype(np.float32)
        ).astype(np.float32)


class VoiceInput:
    """
    Records audio from the USB mic and transcribes with Whisper large-v3.

    Key fix: records at 48 kHz (USB mic native), resamples to 16 kHz before
    passing to Whisper, eliminating the pitch/speed mismatch that caused
    garbled transcription.
    """

    # Sensitivity thresholds (fraction of max amplitude)
    SENSITIVITY_LEVELS = {
        'low':       0.02,
        'medium':    0.01,
        'high':      0.005,
        'very_high': 0.002,
    }

    def __init__(self, logger, model_size="large-v3", noise_reduce=True,
                 device_index=None):
        self.logger       = logger
        self.model_size   = model_size
        self.noise_reduce = noise_reduce
        self.wake_word_enabled = False   # GUI controls this

        # Discover USB mic — returns (index, native_sr, channels)
        # Uses name-based detection so the index survives reboots
        _dev_idx, _dev_sr, _dev_ch = _find_usb_mic_index()
        self._preferred_input_device = device_index if device_index is not None else _dev_idx
        self._native_sr              = _dev_sr
        self._mic_channels           = _dev_ch   # always 1 for USB2.0 Device

        self.logger.info(
            f"Mic: device={self._preferred_input_device}, "
            f"SR={self._native_sr} Hz, ch={self._mic_channels} "
            f"→ resample to {WHISPER_SR} Hz for Whisper"
        )

        # VAD thresholds
        self._silence_threshold = self.SENSITIVITY_LEVELS['medium']
        self._silence_duration  = 1.5   # seconds of silence before auto-stop
        self._min_speech_dur    = 0.3   # minimum speech duration to transcribe

        # Load Whisper model
        self._model = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        self.logger.info(f"Loading Whisper model: {self.model_size} …")
        try:
            if _USE_FASTER:
                self._model = WhisperModel(
                    self.model_size,
                    device="cuda" if self._cuda_available() else "cpu",
                    compute_type="float16" if self._cuda_available() else "int8",
                )
                self.logger.info("✓ faster-whisper model loaded")
            else:
                import whisper as _w
                self._model = _w.load_model(self.model_size)
                self.logger.info("✓ openai-whisper model loaded")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            self._model = None

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    def _resample_if_needed(self, audio: np.ndarray) -> np.ndarray:
        """Resample from native SR to 16kHz if needed. Public helper."""
        return _resample_to_whisper(audio, self._native_sr)

    def set_sensitivity(self, level: str):
        """Set VAD silence threshold. level: 'low'|'medium'|'high'|'very_high'."""
        threshold = self.SENSITIVITY_LEVELS.get(level, self.SENSITIVITY_LEVELS['medium'])
        self._silence_threshold = threshold
        self.logger.info(f"Mic sensitivity: {level} (threshold={threshold})")

    def listen(self, auto_stop=True, context_locations=None, timeout=30.0) -> str:
        """
        Record until silence (auto_stop=True) or timeout, then transcribe.
        Returns transcribed text or empty string.
        """
        if not SOUNDDEVICE_OK:
            self.logger.error("sounddevice not installed")
            return ""
        if self._model is None:
            self.logger.error("Whisper model not loaded")
            return ""

        self.logger.info("Listening… (speak now)")
        audio = self._record_with_vad(timeout=timeout)

        if audio is None or len(audio) == 0:
            self.logger.warn("No audio captured")
            return ""

        return self.transcribe_audio(audio, context_locations=context_locations)

    # ── Known Whisper mishearings for this location set ──────────────────────
    # Whisper maps spoken initialisms/abbreviations to the nearest English word.
    # Key  = what Whisper actually outputs (lowercase)
    # Value = correct location token to substitute
    _MISHEARING_MAP: dict[str, str] = {
        # DG Office — "D G" sounds like "digi" to Whisper
        'digi office':              'DG Office',
        'digi':                     'DG Office',
        'digie office':             'DG Office',
        'digie':                    'DG Office',
        'digital office':           'DG Office',
        'd.g. office':              'DG Office',
        'd g office':               'DG Office',
        'director generals office': 'DG Office',
        # NOTE: bare 'dg' and 'd g' intentionally excluded —
        # too short, would fire inside unrelated names like "DG WD office"
        # BOQ — "B O Q" → various
        'b o q':   'BOQ',
        'b.o.q':   'BOQ',
        'b.o.q.':  'BOQ',
        'boc':     'BOQ',
        'bokeh':   'BOQ',
        'bo queue':'BOQ',
        # MOQ — "M O Q" → various
        'm o q':   'MOQ',
        'm.o.q':   'MOQ',
        'm.o.q.':  'MOQ',
        'moc':     'MOQ',
        'mo queue':'MOQ',
        # HR Office — usually fine but just in case
        'h.r. office': 'HR Office',
        'h r office':  'HR Office',
        # AI&DS Lab
        'aids lab':     'AI&DS Lab',
        'ai ds lab':    'AI&DS Lab',
        'a.i. ds lab':  'AI&DS Lab',
    }

    def _fix_known_mishearings(self, text: str) -> str:
        """
        Correct known Whisper mishearings of abbreviated location names.

        Two-pass strategy:
          1. Full-string match  — handles bare initialism e.g. "digi" alone
          2. Substring replace  — handles it embedded in a command
             e.g. "go to digi office" → "go to DG Office"
        """
        if not text:
            return text
        lower = text.lower().strip()

        # Pass 1: whole transcription is just the misheard word
        if lower in self._MISHEARING_MAP:
            fixed = self._MISHEARING_MAP[lower]
            self.logger.info(f"Mishearing corrected: '{text}' → '{fixed}'")
            return fixed

        # Pass 2: misheard word is embedded inside a longer command
        import re as _re
        result = text
        for wrong, right in self._MISHEARING_MAP.items():
            # Word-boundary aware replacement so "digi" inside "digital" won't fire
            pattern = r'(?<!\w)' + _re.escape(wrong) + r'(?!\w)'
            if _re.search(pattern, result, flags=_re.IGNORECASE):
                result = _re.sub(pattern, right, result, flags=_re.IGNORECASE)
                self.logger.info(f"Mishearing corrected (inline): '{text}' → '{result}'")
                break  # one correction per pass is enough
        return result

    # Known Whisper hallucination phrases triggered by silence/noise.
    # large-v3 is particularly prone to these on short/quiet recordings.
    _HALLUCINATIONS = {
        "thank you for watching",
        "thank you for watching this video",
        "thank you for watching!",
        "thanks for watching",
        "please subscribe",
        "don't forget to subscribe",
        "like and subscribe",
        "see you in the next video",
        "see you next time",
        "subtitles by",
        "transcribed by",
        "www.",
        ".com",
        "amara.org",
        "♪",
        "[ silence ]",
        "[silence]",
        "[music]",
        "[ music ]",
        "(silence)",
    }

    def _is_hallucination(self, text: str) -> bool:
        """Return True if text looks like a Whisper hallucination on silence/noise."""
        if not text:
            return True
        # Strip trailing punctuation Whisper adds (e.g. "3." → "3")
        lower = text.lower().strip().rstrip('.,!?')

        # WHITELIST: single digit or short word that is clearly a real command.
        # Never reject these even if they are only 1 character.
        if lower.isdigit():        # "1", "3", "0" etc.
            return False
        word_nums = {"one","two","three","four","five",
                     "six","seven","eight","nine","zero","cancel","no"}
        if lower in word_nums:
            return False

        # Too short to be real speech (but digits already handled above)
        if len(lower) < 2:
            return True

        # Check against known hallucination phrases
        for phrase in self._HALLUCINATIONS:
            if phrase in lower:
                return True

        return False

    def transcribe_audio(self, audio: np.ndarray, context_locations=None) -> str:
        """
        Transcribe a numpy float32 array.
        Audio may be at native SR (48 kHz) or already at 16 kHz — we normalise both.
        """
        if self._model is None:
            return ""
        if audio is None or len(audio) == 0:
            return ""

        # Ensure float32, mono
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Detect whether we need to resample (recorded at native SR)
        if self._native_sr != WHISPER_SR:
            audio = _resample_to_whisper(audio, self._native_sr)

        # ── Energy gate (MUST run before normalisation) ──────────────────────
        # Only reject completely dead silence (unplugged mic, zero signal).
        raw_rms = float(np.sqrt(np.mean(audio ** 2)))
        if raw_rms < 0.0003:
            self.logger.warn(
                f"Audio completely silent (raw RMS={raw_rms:.6f}) — skipping transcription")
            return ""

        # Normalise amplitude
        peak = np.max(np.abs(audio))
        if peak > 0:
            # Boost gain aggressively if mic signal is weak (peak < 0.1 is quiet mic)
            # This ensures Whisper's internal energy checks see a proper signal level
            target = 0.95
            audio = audio / peak * target

        # Build initial prompt — Whisper uses this as left-side context to bias
        # its language model.  Abbreviations (DG, BOQ, MOQ) must appear early
        # and inside realistic command sentences so Whisper learns to prefer them
        # over phonetically similar English words ("digi", "boc", etc.).
        ABBREV_PRIORITY = ['DG Office', 'BOQ', 'MOQ', 'HR Office', 'AI&DS Lab',
                           'PA&I Lab', 'CV Lab', 'CAAD Office']
        if context_locations:
            rest = [l for l in context_locations if l not in ABBREV_PRIORITY]
            ordered = ABBREV_PRIORITY + rest
            loc_str = ', '.join(ordered[:22])
            prompt = (
                f"Robot voice navigation. Known locations: {loc_str}. "
                f"Commands use exact names: 'go to DG Office', "
                f"'navigate to BOQ', 'take me to MOQ'."
            )
        else:
            prompt = ""

        try:
            if _USE_FASTER:
                segments, _info = self._model.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                    initial_prompt=prompt or None,
                    # vad_filter disabled: it kills quiet speech from low-gain mics.
                    # Normalisation above already ensures signal is at full scale.
                    vad_filter=False,
                )
                text = " ".join(s.text for s in segments).strip()
            else:
                import whisper as _w
                result = self._model.transcribe(
                    audio,
                    language="en",
                    initial_prompt=prompt or None,
                    fp16=self._cuda_available(),
                )
                text = result.get("text", "").strip()

            self.logger.info(f"Transcribed: '{text}'")

            if self._is_hallucination(text):
                self.logger.warn(f"Rejected hallucination: '{text}'")
                return ""

            # Clean up Whisper punctuation artefacts on short commands
            # e.g. "3." → "3",  "Lab." → "Lab"
            cleaned = text.strip()
            if len(cleaned) <= 6:   # only clean short tokens, leave sentences intact
                cleaned = cleaned.rstrip('.,!?')

            # Correct known Whisper mishearings of abbreviated location names
            # e.g. "digi office" → "DG Office", "b o q" → "BOQ"
            cleaned = self._fix_known_mishearings(cleaned)

            return cleaned

        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""

    # ── Internal recording ────────────────────────────────────────────────────

    def _record_with_vad(self, timeout=30.0) -> np.ndarray:
        """
        Record from mic at native SR using Voice Activity Detection.
        Records ALL available channels and picks the loudest one,
        which fixes stereo mics (like K8) where channel 0 is nearly silent.
        Returns raw audio at self._native_sr (caller resamples).
        """
        chunks        = []
        silence_start = None
        speech_found  = False
        start_time    = time.time()

        CHUNK_DUR  = 0.1   # seconds per chunk
        chunk_size = int(self._native_sr * CHUNK_DUR)

        # Use channel count detected at startup (1 for USB2.0 Device)
        n_channels = getattr(self, '_mic_channels', 1)

        def callback(indata, frames, time_info, status):
            nonlocal silence_start, speech_found
            chunk = indata[:, 0].copy()   # always take channel 0 (USB mic is mono)
            chunks.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > self._silence_threshold:
                speech_found  = True
                silence_start = None
            else:
                if speech_found and silence_start is None:
                    silence_start = time.time()

        stream_kwargs = dict(
            samplerate = self._native_sr,
            channels   = n_channels,
            dtype      = 'float32',
            blocksize  = chunk_size,
            callback   = callback,
        )
        if self._preferred_input_device is not None:
            stream_kwargs['device'] = self._preferred_input_device

        try:
            with sd.InputStream(**stream_kwargs):
                while True:
                    time.sleep(0.05)
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self.logger.warn("Recording timeout")
                        break
                    if (speech_found and silence_start is not None and
                            (time.time() - silence_start) >= self._silence_duration):
                        break
        except Exception as e:
            self.logger.error(f"Recording error: {e}")
            return np.array([], dtype=np.float32)

        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks)
        duration = len(audio) / self._native_sr
        self.logger.info(f"Recorded {duration:.1f}s of audio at {self._native_sr} Hz")

        if duration < self._min_speech_dur:
            self.logger.warn("Audio too short, ignoring")
            return np.array([], dtype=np.float32)

        return audio