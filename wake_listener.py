#!/usr/bin/env python3
"""
wake_listener.py — Robust wake-phrase gate (FINAL-only, exact match, confidence check)

Problem you saw
- With wake phrase like "hey robot", the listener triggered on other speech.
Cause
- The earlier version triggered on *partial* recognition results and used permissive substring matching.
Fix in this version
- ✅ Trigger ONLY on FINAL recognition results (AcceptWaveform == True)
- ✅ Require EXACT match to one of your wake phrases (after normalization)
- ✅ Optional confidence gating (reject low-confidence "hallucinated" wakes)
- ✅ Keep mic ownership clean: wake listener releases mic to VAD sidecar, then re-acquires it

How it works (high level)
1) Wake listener holds the microphone and runs Vosk ASR in a restricted grammar mode.
2) When it recognizes a final phrase that exactly equals your wake phrase:
   a) Plays a prompt WAV (e.g., "I am listening")
   b) Stops its own mic stream (releases device)
   c) Calls VAD sidecar: POST /listen/resume (so sidecar can capture speech)
   d) Waits until sidecar auto-pauses again (listening=False)
   e) Reopens mic and returns to wake listening

Dependencies
  pip install vosk simpleaudio sounddevice numpy requests

Vosk model
- Download and unzip a Vosk model directory and set VOSK_MODEL_PATH.
  Example small model good for Raspberry Pi:
    vosk-model-small-en-us-0.15

Run
  source .venv/bin/activate
  VOSK_MODEL_PATH=/home/pi/models/vosk-model-small-en-us-0.15 \
  WAKE_PHRASES="hey robot,hey robert" \
  PROMPT_WAV=./i_am_listening.wav \
  python3 wake_listener.py

Tuning knobs (env vars)
- WAKE_PHRASES: comma-separated phrases (default: "hey robot")
- WAKE_MIN_CONF: min average word confidence (default: 0.70)
- COOLDOWN_SEC: minimum seconds between triggers (default: 2.0)
- SAMPLE_RATE: default 16000
- FRAME_MS: default 20 (10/20/30 ok)
- AUDIO_DEVICE: optional sounddevice input device index or substring of device name
- VAD_CONTROL_URL: base URL to your VAD sidecar (default: http://127.0.0.1:5055)
- RESUME_TIMEOUT_SEC: wait for sidecar to report listening=True (default: 2.0)
- WAIT_FOR_PAUSE_SEC: wait for sidecar to pause again (default: 60.0)

Notes
- For best mic handoff behavior, your VAD sidecar should use:
    AUTO_PAUSE_ON_UTTERANCE_END=true
    STOP_STREAM_WHEN_PAUSED=true
"""

import os
import json
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
import requests
import simpleaudio as sa
from vosk import Model, KaldiRecognizer


# =============================================================================
# Configuration
# =============================================================================

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "/Users/bahram/Projects/voice_models/vosk-model-small-en-us-0.15").strip()
if not VOSK_MODEL_PATH:
    raise RuntimeError(
        "VOSK_MODEL_PATH is not set. Example:\n"
        "  VOSK_MODEL_PATH=/home/pi/models/vosk-model-small-en-us-0.15 python3 wake_listener.py"
    )

WAKE_PHRASES = [p.strip().lower() for p in os.getenv("WAKE_PHRASES", "Wake up Raz").split(",") if p.strip()]
if not WAKE_PHRASES:
    raise RuntimeError("WAKE_PHRASES is empty. Provide at least one phrase, e.g. WAKE_PHRASES='hey robot'.")

WAKE_MIN_CONF = float(os.getenv("WAKE_MIN_CONF", "0.70"))  # average word confidence threshold
COOLDOWN_SEC = float(os.getenv("COOLDOWN_SEC", "2.0"))

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
FRAME_MS = int(os.getenv("FRAME_MS", "20"))
if FRAME_MS not in (10, 20, 30):
    raise ValueError("FRAME_MS must be 10, 20, or 30.")

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

PROMPT_WAV = os.getenv("PROMPT_WAV", "./i_am_listening.wav")

VAD_CONTROL_URL = os.getenv("VAD_CONTROL_URL", "http://127.0.0.1:5055").rstrip("/")
RESUME_TIMEOUT_SEC = float(os.getenv("RESUME_TIMEOUT_SEC", "2.0"))
WAIT_FOR_PAUSE_SEC = float(os.getenv("WAIT_FOR_PAUSE_SEC", "60.0"))

AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "").strip()  # can be index "2" or substring "USB"


# =============================================================================
# Small helpers
# =============================================================================

def normalize_text(s: str) -> str:
    """Lowercase + collapse whitespace."""
    return " ".join(s.lower().strip().split())


def play_wav_blocking(path: str) -> None:
    """Play a WAV file synchronously. If missing/unplayable, just log and continue."""
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"[PROMPT] Could not play WAV '{path}': {e}")


def http_post(url: str, timeout: float = 2.0) -> bool:
    """POST helper: returns True/False, never raises."""
    try:
        r = requests.post(url, timeout=timeout)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[HTTP] POST {url} failed: {e}")
        return False


def http_get_json(url: str, timeout: float = 2.0) -> Optional[dict]:
    """GET helper: returns dict or None, never raises."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[HTTP] GET {url} failed: {e}")
        return None


def select_input_device() -> Optional[int]:
    """
    If AUDIO_DEVICE is set:
      - if it parses as int => use as device index
      - else find a device whose name contains that substring (case-insensitive)
    Returns None to use the default device.
    """
    if not AUDIO_DEVICE:
        return None

    try:
        return int(AUDIO_DEVICE)
    except ValueError:
        pass

    needle = AUDIO_DEVICE.lower()
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        name = str(d.get("name", "")).lower()
        if needle in name and d.get("max_input_channels", 0) > 0:
            return i

    print(f"[AUDIO] Could not find input device matching '{AUDIO_DEVICE}'. Using default.")
    return None


# =============================================================================
# Wake listener
# =============================================================================

@dataclass
class WakeState:
    last_trigger_ts: float = 0.0


class WakeWordListener:
    """
    Owns the microphone while listening for a wake phrase.
    Releases microphone to VAD sidecar when wake triggers.
    """

    def __init__(self):
        self.device = select_input_device()

        # Build Vosk recognizer with grammar restriction:
        # This strongly reduces false positives and improves speed.
        phrases_norm = [normalize_text(p) for p in WAKE_PHRASES]
        grammar_json = json.dumps(phrases_norm)

        self.model = Model(VOSK_MODEL_PATH)
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE, grammar_json)

        # Enable word-level info so we can use confidence gating
        self.rec.SetWords(True)

        self.allowed_phrases = set(phrases_norm)

        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=300)
        self.stream: Optional[sd.InputStream] = None

        self.lock = threading.Lock()
        self.state = WakeState()

        self.sidecar_resume_url = f"{VAD_CONTROL_URL}/listen/resume"
        self.sidecar_status_url = f"{VAD_CONTROL_URL}/listen/status"

    # ---------------- Mic management ----------------

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        mono = indata[:, 0].copy()  # float32
        try:
            self.audio_q.put_nowait(mono)
        except queue.Full:
            pass

    def start_stream(self):
        if self.stream is not None:
            return

        kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )
        if self.device is not None:
            kwargs["device"] = self.device

        self.stream = sd.InputStream(**kwargs)
        self.stream.start()
        print("[WAKE] Mic stream started (wake listening)")

    def stop_stream(self):
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None

        self._drain_queue()
        print("[WAKE] Mic stream stopped (handoff to sidecar)")

    def _drain_queue(self):
        try:
            while True:
                self.audio_q.get_nowait()
        except queue.Empty:
            pass

    # ---------------- Wake detection helpers ----------------

    @staticmethod
    def _float_to_pcm16(frame: np.ndarray) -> bytes:
        pcm = np.clip(frame, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def _cooldown_ok(self) -> bool:
        with self.lock:
            return (time.time() - self.state.last_trigger_ts) >= COOLDOWN_SEC

    def _mark_trigger(self):
        with self.lock:
            self.state.last_trigger_ts = time.time()

    @staticmethod
    def _avg_confidence(vosk_result: dict) -> Optional[float]:
        """
        Vosk final result may contain 'result': [{'word':..., 'conf':...}, ...]
        Returns average confidence or None if unavailable.
        """
        words = vosk_result.get("result")
        if not isinstance(words, list) or not words:
            return None

        confs = []
        for w in words:
            if isinstance(w, dict) and isinstance(w.get("conf"), (int, float)):
                confs.append(float(w["conf"]))

        if not confs:
            return None
        return sum(confs) / len(confs)

    def _wake_match(self, final_text: str, vosk_result: dict) -> bool:
        """
        Robust wake criteria:
        1) EXACT match against allowed phrases (normalized)
        2) Confidence gating if confidence is available
        """
        text_norm = normalize_text(final_text)
        if text_norm not in self.allowed_phrases:
            return False

        avg_conf = self._avg_confidence(vosk_result)
        if avg_conf is None:
            # If confidence isn't present, accept the match (still exact phrase).
            return True

        if avg_conf >= WAKE_MIN_CONF:
            return True

        print(f"[WAKE] Rejected low-confidence match '{text_norm}' (avg_conf={avg_conf:.2f})")
        return False

    # ---------------- Sidecar coordination ----------------

    def _wait_sidecar_listening(self, timeout_s: float) -> bool:
        """Wait until sidecar status shows listening==True."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = http_get_json(self.sidecar_status_url, timeout=1.0)
            if st and st.get("listening") is True:
                return True
            time.sleep(0.1)
        return False

    def _wait_sidecar_paused(self, timeout_s: float) -> bool:
        """Wait until sidecar status shows listening==False."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = http_get_json(self.sidecar_status_url, timeout=1.0)
            if st and st.get("listening") is False:
                return True
            time.sleep(0.2)
        return False

    def _trigger_sequence(self):
        """
        Wake action sequence:
        1) Play prompt WAV
        2) Release mic (stop wake stream)
        3) Resume sidecar
        4) Wait for sidecar to pause again
        5) Re-acquire mic (restart wake stream)
        """
        print("[WAKE] Wake phrase detected (final + exact + confident)")

        # Play prompt (blocking). If you'd prefer to ensure no mic contention
        # with sound output, you can stop_stream() before playing.
        play_wav_blocking(PROMPT_WAV)

        # Release mic so sidecar can open it
        self.stop_stream()

        # Tell sidecar to resume
        if not http_post(self.sidecar_resume_url, timeout=2.0):
            print("[WAKE] Failed to resume sidecar. Returning to wake listening.")
            self.start_stream()
            return

        if not self._wait_sidecar_listening(RESUME_TIMEOUT_SEC):
            print("[WAKE] Sidecar did not become listening=True in time. Returning to wake listening.")
            self.start_stream()
            return

        print("[WAKE] Sidecar is listening. Waiting for it to pause again...")

        if not self._wait_sidecar_paused(WAIT_FOR_PAUSE_SEC):
            print("[WAKE] Sidecar did not pause within timeout. Returning to wake listening anyway.")
        else:
            print("[WAKE] Sidecar paused again. Returning to wake listening.")

        self.start_stream()

    # ---------------- Main loop ----------------

    def run_forever(self):
        print("[WAKE] Starting wake-word listener (final-only exact match)")
        print(f"[WAKE] Allowed wake phrases: {sorted(self.allowed_phrases)}")
        print(f"[WAKE] WAKE_MIN_CONF={WAKE_MIN_CONF:.2f}, COOLDOWN_SEC={COOLDOWN_SEC:.1f}")
        print(f"[WAKE] Vosk model: {VOSK_MODEL_PATH}")
        print(f"[WAKE] Sidecar control: {VAD_CONTROL_URL}")
        print(f"[WAKE] Prompt WAV: {PROMPT_WAV}")

        self.start_stream()

        while True:
            frame = self.audio_q.get()
            pcm = self._float_to_pcm16(frame)

            try:
                # IMPORTANT: we trigger ONLY when AcceptWaveform returns True (final result).
                if self.rec.AcceptWaveform(pcm):
                    res = json.loads(self.rec.Result() or "{}")
                    text = res.get("text", "")

                    text_norm = normalize_text(text)
                    if text_norm:
                        # Uncomment to debug what it hears:
                        # print(f"[WAKE DEBUG] final='{text_norm}' raw={res}")

                        if self._cooldown_ok() and self._wake_match(text_norm, res):
                            self._mark_trigger()
                            self._trigger_sequence()
                else:
                    # Ignore partials to avoid false wakes
                    pass

            except Exception as e:
                print(f"[WAKE] Recognizer error: {e}")
                time.sleep(0.05)


# =============================================================================
# Entry point
# =============================================================================

def main():
    listener = WakeWordListener()
    listener.run_forever()


if __name__ == "__main__":
    main()
