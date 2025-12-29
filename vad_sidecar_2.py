#!/usr/bin/env python3
"""
VAD sidecar for Raspberry Pi 5 (NOISE-ROBUST + SYNC UPLOAD + ACTION PAUSE)

What this script does (high-level):
1) Captures microphone audio continuously (16 kHz mono by default).
2) Splits audio into fixed frames (10/20/30 ms; configurable).
3) Converts float audio -> PCM16 bytes for WebRTC VAD.
4) Uses a robust START detector to avoid false triggers from noise:
   - WebRTC VAD must say "speech"
   - Energy gate must pass (fixed RMS or adaptive noise floor)
   - Start only if enough frames in a short window are "speech-like" (ratio test)
5) Records until:
   - N ms of VAD-silence (plus optional hangover), OR
   - MAX_UTTERANCE_MS safety cap
6) When an utterance ends:
   - Optionally PAUSE listening (and stop mic stream) to avoid listening during action
   - Save WAV locally
   - Upload WAV synchronously (blocking) to backend
   - If upload succeeds: usually stay paused until external /listen/resume
   - If upload fails: optionally auto-resume so you don't get stuck forever
7) Exposes HTTP endpoints:
   POST /listen/pause
   POST /listen/resume
   GET  /listen/status

Key improvements vs your original code:
- ✅ Requires multiple speech-like frames to start (prevents 1-frame false starts)
- ✅ Adds RMS energy gate, optionally adaptive to room noise (noise floor)
- ✅ Default VAD_MODE=3 (fewer false positives)
- ✅ Optional hangover to reduce choppy segmentation
- ✅ Synchronous upload that NEVER crashes the process if backend is down
- ✅ Optional auto-pause around upload/action so you don't listen while acting

Install:
  sudo apt update
  sudo apt install -y python3-pip portaudio19-dev
  pip3 install webrtcvad sounddevice numpy requests flask
"""

import os
import time
import queue
import wave
import threading
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
import requests
from requests.exceptions import RequestException
from flask import Flask, jsonify


# =============================================================================
# Configuration (tune these)
# =============================================================================

# Backend upload
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:18081/robot/woke")
OUT_DIR = os.getenv("OUT_DIR", "./voice_segments")

# Audio format
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))

# Frame size: MUST be 10, 20, or 30 for WebRTC VAD
FRAME_MS = int(os.getenv("FRAME_MS", "20"))  # consider 30 for more stability
if FRAME_MS not in (10, 20, 30):
    raise ValueError("FRAME_MS must be 10, 20, or 30 for WebRTC VAD")

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# WebRTC VAD mode: 0..3, higher = more aggressive (fewer false positives)
VAD_MODE = int(os.getenv("VAD_MODE", "3"))

# Stop condition (silence)
SILENCE_STOP_MS = int(os.getenv("SILENCE_STOP_MS", "1000"))      # stop after this much silence
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "20000"))   # safety cap
PRE_ROLL_MS = int(os.getenv("PRE_ROLL_MS", "200"))               # keep audio before speech start

# Noise-robust START detection (prevents false starts)
START_WINDOW_MS = int(os.getenv("START_WINDOW_MS", "200"))
START_SPEECH_RATIO = float(os.getenv("START_SPEECH_RATIO", "0.6"))

# Energy gate behavior
USE_ADAPTIVE_NOISE_FLOOR = os.getenv("USE_ADAPTIVE_NOISE_FLOOR", "true").lower() == "true"

# Fixed RMS threshold (used only if adaptive noise floor is OFF)
MIN_RMS_FIXED = float(os.getenv("MIN_RMS_FIXED", "0.012"))

# Adaptive noise floor parameters (used only if adaptive noise floor is ON)
NOISE_MULTIPLIER = float(os.getenv("NOISE_MULTIPLIER", "3.0"))
NOISE_ALPHA = float(os.getenv("NOISE_ALPHA", "0.05"))
NOISE_FLOOR_MIN = float(os.getenv("NOISE_FLOOR_MIN", "0.003"))

# Optional "hangover" after speech ends (reduces choppy segmentation)
HANGOVER_MS = int(os.getenv("HANGOVER_MS", "0"))  # e.g. 200; 0 disables

# HTTP control server
HTTP_HOST = os.getenv("HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("HTTP_PORT", "5055"))

# ---------------------------------------------------------------------------
# Action/Upload pause policy (sync upload + do not listen during action)
# ---------------------------------------------------------------------------

# When an utterance ends, pause listening before uploading (recommended for your use case)
AUTO_PAUSE_ON_UTTERANCE_END = os.getenv("AUTO_PAUSE_ON_UTTERANCE_END", "true").lower() == "true"

# After successful upload:
# - If false: remain paused until something calls POST /listen/resume (recommended for "action time")
# - If true: automatically resume immediately after successful upload
AUTO_RESUME_AFTER_SUCCESS = os.getenv("AUTO_RESUME_AFTER_SUCCESS", "false").lower() == "true"

# If upload fails (backend down), auto-resume so you don't get stuck forever
AUTO_RESUME_AFTER_FAILURE = os.getenv("AUTO_RESUME_AFTER_FAILURE", "true").lower() == "true"

# Very important for synchronous upload:
# stopping the mic stream prevents queue growth while you're blocked uploading/acting
STOP_STREAM_WHEN_PAUSED = os.getenv("STOP_STREAM_WHEN_PAUSED", "true").lower() == "true"

# Upload timeout (keep short so failures return quickly)
UPLOAD_TIMEOUT_SEC = float(os.getenv("UPLOAD_TIMEOUT_SEC", "5.0"))


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def float_to_pcm16_bytes(frame: np.ndarray) -> bytes:
    """
    sounddevice provides float32 samples in [-1..1].
    WebRTC VAD expects 16-bit PCM (little-endian) bytes.
    """
    pcm = np.clip(frame, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    return pcm.tobytes()


def frame_rms(frame: np.ndarray) -> float:
    """
    Compute RMS energy of a float frame in [-1..1].
    RMS is a robust "loudness-ish" measure for gating.
    """
    return float(np.sqrt(np.mean(frame * frame) + 1e-12))


def write_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    """
    Writes a mono 16-bit PCM WAV at the given sample rate.
    """
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def post_wav(path: str) -> bool:
    """
    Synchronous upload:
      - Returns True on success, False on failure
      - NEVER raises (so the process stays alive even if backend is down)
    """
    try:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "audio/wav")}
            data = {"source": "pi5", "ts": str(time.time())}

            r = requests.post(
                BACKEND_URL,
                files=files,
                data=data,
                timeout=UPLOAD_TIMEOUT_SEC,
            )
            r.raise_for_status()

        return True

    except RequestException as e:
        print(f"[UPLOAD ERROR] Backend unreachable or request failed: {e}")
        return False

    except Exception as e:
        print(f"[UPLOAD ERROR] Unexpected failure: {e}")
        return False


# =============================================================================
# State
# =============================================================================

@dataclass
class State:
    # Control
    listening_enabled: bool = True

    # Recording state
    listening_enabled: bool = False  # START PAUSED
    recorded: bytearray = field(default_factory=bytearray)

    # Pre-roll ring buffer (stores PCM bytes frames)
    ring: list[bytes] = field(default_factory=list)

    # Counters while recording
    silence_count: int = 0
    utterance_frames: int = 0
    speech_started_at: float | None = None

    # Start detector history: True if (VAD says speech) AND (energy gate passes)
    start_hist: deque[bool] = field(default_factory=lambda: deque(maxlen=1))

    # Noise floor estimator (RMS)
    noise_floor: float = NOISE_FLOOR_MIN

    # Debug last computed metrics
    last_rms: float = 0.0
    last_vad: bool = False
    last_speechlike: bool = False


# =============================================================================
# Main sidecar class
# =============================================================================

class VadSidecar:
    """
    Producer/consumer design:
      - sounddevice callback pushes float frames into a queue
      - main loop consumes frames, runs VAD + gating + segmentation
    """

    def __init__(self):
        ensure_dir(OUT_DIR)

        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)

        self.lock = threading.Lock()
        self.state = State()

        # Convert ms tuning parameters to "frame counts"
        self.pre_roll_frames = max(1, int(PRE_ROLL_MS / FRAME_MS))
        self.silence_stop_frames = max(1, int(SILENCE_STOP_MS / FRAME_MS))
        self.max_frames = max(1, int(MAX_UTTERANCE_MS / FRAME_MS))

        self.start_window_frames = max(1, int(START_WINDOW_MS / FRAME_MS))

        # Optional hangover extends required silence frames
        self.hangover_frames = max(0, int(HANGOVER_MS / FRAME_MS))
        self.stop_required_silence_frames = self.silence_stop_frames + self.hangover_frames

        # Initialize start history deque with correct maxlen
        self.state.start_hist = deque(maxlen=self.start_window_frames)

        self.stream: sd.InputStream | None = None

    # -------------------------------------------------------------------------
    # Pause/Resume/Status
    # -------------------------------------------------------------------------

    def pause(self):
        """
        Pause listening:
          - disables listening
          - clears any in-progress recording state
          - optionally stops the mic stream (recommended for sync upload / action time)
        """
        with self.lock:
            self.state.listening_enabled = False
            self._reset_recording_locked(reset_noise_floor=False)

        if STOP_STREAM_WHEN_PAUSED:
            self._stop_stream()

        # Drain any queued frames so we don't process stale audio after resume
        self._drain_audio_queue()
        print("Listening paused")

    def resume(self):
        """
        Resume listening:
          - optionally restarts mic stream
          - enables listening
          - clears state
        """
        if STOP_STREAM_WHEN_PAUSED:
            self._start_stream()

        with self.lock:
            self.state.listening_enabled = True
            self._reset_recording_locked(reset_noise_floor=False)

        # Drain any queued frames just in case
        self._drain_audio_queue()
        print("Listening resumed")

    def status(self):
        with self.lock:
            return {
                "listening": self.state.listening_enabled,
                "recording": self.state.recording,
                "backend_url": BACKEND_URL,
                "out_dir": OUT_DIR,
                # Tuning info
                "frame_ms": FRAME_MS,
                "vad_mode": VAD_MODE,
                "pre_roll_ms": PRE_ROLL_MS,
                "silence_stop_ms": SILENCE_STOP_MS,
                "hangover_ms": HANGOVER_MS,
                "start_window_ms": START_WINDOW_MS,
                "start_speech_ratio": START_SPEECH_RATIO,
                "use_adaptive_noise_floor": USE_ADAPTIVE_NOISE_FLOOR,
                "noise_floor": self.state.noise_floor,
                "noise_multiplier": NOISE_MULTIPLIER,
                "min_rms_fixed": MIN_RMS_FIXED,
                "upload_timeout_sec": UPLOAD_TIMEOUT_SEC,
                # Action policy
                "auto_pause_on_utterance_end": AUTO_PAUSE_ON_UTTERANCE_END,
                "auto_resume_after_success": AUTO_RESUME_AFTER_SUCCESS,
                "auto_resume_after_failure": AUTO_RESUME_AFTER_FAILURE,
                "stop_stream_when_paused": STOP_STREAM_WHEN_PAUSED,
                # Debug last-frame metrics
                "last_rms": self.state.last_rms,
                "last_vad": self.state.last_vad,
                "last_speechlike": self.state.last_speechlike,
            }

    # -------------------------------------------------------------------------
    # Audio queue + stream management
    # -------------------------------------------------------------------------

    def _drain_audio_queue(self):
        """
        Removes any buffered frames from the queue.
        Useful when pausing/resuming to avoid processing stale audio.
        """
        try:
            while True:
                self.audio_q.get_nowait()
        except queue.Empty:
            pass

    def _callback(self, indata, frames, time_info, status):
        """
        sounddevice callback thread.
        Keep this minimal: copy data, push to queue, drop if overloaded.
        """
        if status:
            # You can log status if you want (XRUNs, etc.)
            pass

        mono = indata[:, 0].copy()  # float32
        try:
            self.audio_q.put_nowait(mono)
        except queue.Full:
            # If main loop can't keep up, drop frames
            pass

    def _start_stream(self):
        if self.stream is not None:
            return
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )
        self.stream.start()
        print("Mic stream started")

    def _stop_stream(self):
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None
        print("Mic stream stopped")

    # -------------------------------------------------------------------------
    # Core signal classification helpers
    # -------------------------------------------------------------------------

    def _is_speech_vad(self, pcm_bytes: bytes) -> bool:
        """WebRTC VAD classification on 10/20/30ms PCM16 frames."""
        return self.vad.is_speech(pcm_bytes, SAMPLE_RATE)

    def _update_noise_floor_locked(self, rms: float, vad_speech: bool):
        """
        Adaptive noise floor estimator.
        Learns from non-speech frames only, and only while NOT recording.
        """
        if not USE_ADAPTIVE_NOISE_FLOOR:
            return
        if self.state.recording:
            return
        if not vad_speech:
            nf = self.state.noise_floor
            nf = (1.0 - NOISE_ALPHA) * nf + NOISE_ALPHA * rms
            self.state.noise_floor = max(NOISE_FLOOR_MIN, nf)

    def _energy_gate_passes_locked(self, rms: float) -> bool:
        """
        Energy gate:
          - Adaptive: rms >= noise_floor * NOISE_MULTIPLIER
          - Fixed:    rms >= MIN_RMS_FIXED
        """
        if USE_ADAPTIVE_NOISE_FLOOR:
            return rms >= (self.state.noise_floor * NOISE_MULTIPLIER)
        return rms >= MIN_RMS_FIXED

    def _speechlike_locked(self, vad_speech: bool, rms: float) -> bool:
        """Speech-like = VAD speech AND energy gate passes."""
        if not vad_speech:
            return False
        return self._energy_gate_passes_locked(rms)

    # -------------------------------------------------------------------------
    # Recording / segmentation
    # -------------------------------------------------------------------------

    def _reset_recording_locked(self, reset_noise_floor: bool = False):
        """Resets utterance recording state (not the queue/stream)."""
        self.state.recording = False
        self.state.recorded = bytearray()
        self.state.ring = []
        self.state.silence_count = 0
        self.state.utterance_frames = 0
        self.state.speech_started_at = None
        self.state.start_hist = deque(maxlen=self.start_window_frames)
        if reset_noise_floor:
            self.state.noise_floor = NOISE_FLOOR_MIN

    def _save_and_upload_sync(self, pcm_bytes: bytes) -> bool:
        """
        Synchronous "finalize" step:
          - save WAV
          - upload WAV (blocking)
          - NEVER raises; returns True/False
        """
        filename = f"utt_{int(time.time() * 1000)}.wav"
        path = os.path.join(OUT_DIR, filename)

        try:
            write_wav(path, pcm_bytes, SAMPLE_RATE)
            print(f"Saved: {path}")
        except Exception as e:
            print(f"[DISK ERROR] Failed to save WAV: {e}")
            return False

        ok = post_wav(path)
        if ok:
            print("Uploaded to backend")
        else:
            print("Upload failed (will not crash; continuing)")
        return ok

    def run_forever(self):
        """
        Main loop:
          - consume frames
          - VAD + energy gating + windowed start
          - record until stop criteria
          - on stop: optionally pause, then save+upload synchronously
        """
        print(f"VAD sidecar started (VAD_MODE={VAD_MODE}, FRAME_MS={FRAME_MS}).")
        print(f"Posting utterances to: {BACKEND_URL}")
        print(f"Output dir: {OUT_DIR}")
        print(f"HTTP control: http://{HTTP_HOST}:{HTTP_PORT}")
        print("Listening...")

        # Start mic stream ONLY if listening is enabled
        if self.state.listening_enabled:
            self._start_stream()
        else:
            print("VAD sidecar started in PAUSED state (waiting for /listen/resume)")


        while True:
            frame = self.audio_q.get()          # float32 mono frame
            rms = frame_rms(frame)              # energy
            pcm = float_to_pcm16_bytes(frame)   # bytes for VAD and WAV

            should_stop = False
            utterance_bytes: bytes | None = None

            with self.lock:
                # Update last metrics for /listen/status debug
                self.state.last_rms = rms

                # If paused: discard audio and keep resetting state
                if not self.state.listening_enabled:
                    self._reset_recording_locked(reset_noise_floor=False)
                    continue

                # Maintain pre-roll ring buffer (PCM bytes)
                self.state.ring.append(pcm)
                if len(self.state.ring) > self.pre_roll_frames:
                    self.state.ring.pop(0)

                # VAD classification
                vad_speech = self._is_speech_vad(pcm)
                self.state.last_vad = vad_speech

                # Update adaptive noise floor (only when appropriate)
                self._update_noise_floor_locked(rms=rms, vad_speech=vad_speech)

                # Combine VAD + energy gate for robust "speechlike" (start detector)
                speechlike = self._speechlike_locked(vad_speech=vad_speech, rms=rms)
                self.state.last_speechlike = speechlike

                # ------------------------------------------------------------
                # START logic (robust)
                # ------------------------------------------------------------
                if not self.state.recording:
                    # Only start if enough of the last window frames are speechlike
                    self.state.start_hist.append(speechlike)

                    if len(self.state.start_hist) == self.start_window_frames:
                        ratio = sum(self.state.start_hist) / len(self.state.start_hist)
                        if ratio >= START_SPEECH_RATIO:
                            self.state.recording = True
                            self.state.recorded = bytearray().join(self.state.ring)
                            self.state.recorded.extend(pcm)
                            self.state.silence_count = 0
                            self.state.utterance_frames = 0
                            self.state.speech_started_at = time.time()
                            print(
                                f"Speech started "
                                f"(ratio={ratio:.2f}, rms={rms:.4f}, nf={self.state.noise_floor:.4f})"
                            )
                    continue

                # ------------------------------------------------------------
                # RECORDING logic
                # ------------------------------------------------------------
                self.state.recorded.extend(pcm)
                self.state.utterance_frames += 1

                # During recording, silence detection uses VAD (not speechlike),
                # so quiet speech won't be cut off by the energy gate.
                if vad_speech:
                    self.state.silence_count = 0
                else:
                    self.state.silence_count += 1

                should_stop = (
                    self.state.silence_count >= self.stop_required_silence_frames
                    or self.state.utterance_frames >= self.max_frames
                )

                if should_stop:
                    start_t = self.state.speech_started_at or time.time()
                    duration = time.time() - start_t
                    reason = "silence" if self.state.silence_count >= self.stop_required_silence_frames else "maxlen"
                    print(f"Speech ended ({duration:.2f}s, reason={reason})")

                    utterance_bytes = bytes(self.state.recorded)
                    self._reset_recording_locked(reset_noise_floor=False)

            # -----------------------------------------------------------------
            # Outside lock: finalize utterance synchronously
            # -----------------------------------------------------------------
            if should_stop and utterance_bytes is not None:
                if AUTO_PAUSE_ON_UTTERANCE_END:
                    # Stop listening immediately (and stop mic stream) so we don't
                    # listen during sync upload/action time.
                    self.pause()

                ok = self._save_and_upload_sync(utterance_bytes)

                # Resume policy
                if ok and AUTO_RESUME_AFTER_SUCCESS:
                    self.resume()
                elif (not ok) and AUTO_RESUME_AFTER_FAILURE:
                    # Backend down -> don't get stuck paused forever
                    self.resume()
                else:
                    # Remain paused until external POST /listen/resume
                    pass


# =============================================================================
# HTTP control server
# =============================================================================

def start_http_server(sidecar: VadSidecar):
    app = Flask(__name__)

    @app.get("/listen/status")
    def status():
        return jsonify(sidecar.status())

    @app.post("/listen/pause")
    def pause():
        sidecar.pause()
        return jsonify(sidecar.status())

    @app.post("/listen/resume")
    def resume():
        sidecar.resume()
        return jsonify(sidecar.status())

    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)


# =============================================================================
# Entry point
# =============================================================================

def main():
    sidecar = VadSidecar()

    http_thread = threading.Thread(target=start_http_server, args=(sidecar,), daemon=True)
    http_thread.start()

    sidecar.run_forever()


if __name__ == "__main__":
    main()
