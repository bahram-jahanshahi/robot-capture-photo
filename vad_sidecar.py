#!/usr/bin/env python3
"""
VAD sidecar for Raspberry Pi 5:
- Continuously listens to microphone (16kHz mono)
- Uses WebRTC VAD to detect speech
- Records an utterance and stops after ~1s of silence
- Saves utterance as WAV
- Uploads WAV to Spring Boot via HTTP multipart POST
- Exposes localhost HTTP endpoints to PAUSE/RESUME listening:
    POST http://127.0.0.1:5055/listen/pause
    POST http://127.0.0.1:5055/listen/resume
    GET  http://127.0.0.1:5055/listen/status

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
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import sounddevice as sd
import webrtcvad
import requests
from flask import Flask, jsonify

# -------------------- Config --------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:18081/robot/woke")
#OUT_DIR = os.getenv("OUT_DIR", "/home/pi/voice_segments")
OUT_DIR = os.getenv("OUT_DIR", "./voice_segments")

SAMPLE_RATE = 16000
CHANNELS = 1

FRAME_MS = 20  # must be 10/20/30 for WebRTC VAD
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples at 16kHz for 20ms

VAD_MODE = int(os.getenv("VAD_MODE", "2"))  # 0..3, higher = more aggressive
SILENCE_STOP_MS = int(os.getenv("SILENCE_STOP_MS", "1000"))  # stop after this much silence
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "20000"))  # safety cap
PRE_ROLL_MS = int(os.getenv("PRE_ROLL_MS", "200"))  # keep this much audio before speech

HTTP_HOST = os.getenv("HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("HTTP_PORT", "5055"))

# Optional: while paused, you can either keep mic stream open (default) or stop the stream.
# Keeping it open avoids device re-init glitches.
STOP_STREAM_WHEN_PAUSED = os.getenv("STOP_STREAM_WHEN_PAUSED", "false").lower() == "true"

# ------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def float_to_pcm16_bytes(frame: np.ndarray) -> bytes:
    """sounddevice gives float32 [-1..1]. Convert to 16-bit PCM bytes."""
    pcm = np.clip(frame, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    return pcm.tobytes()


def write_wav(path: str, pcm_bytes: bytes) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def post_wav(path: str) -> None:
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "audio/wav")}
        data = {"source": "pi5", "ts": str(time.time())}
        r = requests.post(BACKEND_URL, files=files, data=data, timeout=30)
        r.raise_for_status()


@dataclass
class State:
    listening_enabled: bool = True
    recording: bool = False
    recorded: bytearray = field(default_factory=bytearray)
    ring: list[bytes] = field(default_factory=list)
    silence_count: int = 0
    utterance_frames: int = 0
    speech_started_at: float | None = None

class VadSidecar:
    def __init__(self):
        ensure_dir(OUT_DIR)

        self.vad = webrtcvad.Vad(VAD_MODE)
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=100)

        self.lock = threading.Lock()
        self.state = State()

        self.pre_roll_frames = max(1, int(PRE_ROLL_MS / FRAME_MS))
        self.silence_stop_frames = max(1, int(SILENCE_STOP_MS / FRAME_MS))
        self.max_frames = max(1, int(MAX_UTTERANCE_MS / FRAME_MS))

        self.stream: sd.InputStream | None = None

    # ---------- Pause/Resume ----------
    def pause(self):
        with self.lock:
            self.state.listening_enabled = False
            self._reset_recording_locked()
        if STOP_STREAM_WHEN_PAUSED:
            self._stop_stream()
        print("Listening paused")

    def resume(self):
        if STOP_STREAM_WHEN_PAUSED:
            self._start_stream()
        with self.lock:
            self.state.listening_enabled = True
            self._reset_recording_locked()
        print("Listening resumed")

    def status(self):
        with self.lock:
            return {
                "listening": self.state.listening_enabled,
                "recording": self.state.recording,
                "backend_url": BACKEND_URL,
                "out_dir": OUT_DIR,
            }

    # ---------- Stream mgmt ----------
    def _callback(self, indata, frames, time_info, status):
        # Keep callback minimal
        if status:
            # You can log status if you want
            pass
        mono = indata[:, 0].copy()
        try:
            self.audio_q.put_nowait(mono)
        except queue.Full:
            # Drop frames if overloaded
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

    # ---------- Recording logic ----------
    def _reset_recording_locked(self):
        self.state.recording = False
        self.state.recorded = bytearray()
        self.state.ring = []
        self.state.silence_count = 0
        self.state.utterance_frames = 0
        self.state.speech_started_at = None

    def _is_speech(self, pcm_bytes: bytes) -> bool:
        return self.vad.is_speech(pcm_bytes, SAMPLE_RATE)

    def _finish_utterance(self, pcm_bytes: bytes):
        filename = f"utt_{int(time.time() * 1000)}.wav"
        path = os.path.join(OUT_DIR, filename)
        write_wav(path, pcm_bytes)
        print(f"Saved: {path}")

        try:
            post_wav(path)
            print("Uploaded to backend")
        except Exception as e:
            print(f"Upload failed: {e}")

    def run_forever(self):
        print(f"VAD sidecar started (VAD_MODE={VAD_MODE}).")
        print(f"Posting utterances to: {BACKEND_URL}")
        print(f"HTTP control: http://{HTTP_HOST}:{HTTP_PORT}")
        print("Listening...")

        # Start mic stream once by default
        if not STOP_STREAM_WHEN_PAUSED:
            self._start_stream()
        else:
            # even in STOP_STREAM_WHEN_PAUSED mode, start now
            self._start_stream()

        while True:
            frame = self.audio_q.get()
            pcm = float_to_pcm16_bytes(frame)

            with self.lock:
                # If paused: ignore audio and keep state reset.
                if not self.state.listening_enabled:
                    # keep discarding frames; reset state to avoid lingering recording
                    self._reset_recording_locked()
                    continue

                # Maintain pre-roll ring buffer
                self.state.ring.append(pcm)
                if len(self.state.ring) > self.pre_roll_frames:
                    self.state.ring.pop(0)

                speech = self._is_speech(pcm)

                if not self.state.recording:
                    if speech:
                        # Start recording with pre-roll
                        self.state.recording = True
                        self.state.recorded = bytearray().join(self.state.ring)
                        self.state.recorded.extend(pcm)
                        self.state.silence_count = 0
                        self.state.utterance_frames = 0
                        self.state.speech_started_at = time.time()
                        print("Speech started")
                    continue

                # Recording path
                self.state.recorded.extend(pcm)
                self.state.utterance_frames += 1

                if speech:
                    self.state.silence_count = 0
                else:
                    self.state.silence_count += 1

                should_stop = (
                    self.state.silence_count >= self.silence_stop_frames
                    or self.state.utterance_frames >= self.max_frames
                )

                if should_stop:
                    duration = time.time() - (self.state.speech_started_at or time.time())
                    print(f"Speech ended ({duration:.2f}s)")
                    utterance_bytes = bytes(self.state.recorded)
                    self._reset_recording_locked()

            # Finish/upload outside lock
            if should_stop:
                self._finish_utterance(utterance_bytes)


# -------------------- HTTP control server --------------------
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

    # Note: use_reloader=False is important so Flask doesn't start twice
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)


def main():
    sidecar = VadSidecar()

    # Start HTTP server in background thread
    http_thread = threading.Thread(target=start_http_server, args=(sidecar,), daemon=True)
    http_thread.start()

    # Run audio loop
    sidecar.run_forever()


if __name__ == "__main__":
    main()
