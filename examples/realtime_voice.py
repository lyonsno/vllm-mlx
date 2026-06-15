#!/usr/bin/env python3
"""
Realtime voice client with barge-in for vllm-mlx /v1/realtime.

Continuous mic streaming — speak anytime, even during response playback.
When you speak during playback (barge-in), the current response is
truncated and the model responds to your new input.

Usage:
    # Start server first:
    python -m vllm_mlx.server --model mlx-community/gemma-4-12B-it-4bit

    # Then run client:
    python examples/realtime_voice.py
    python examples/realtime_voice.py --vad-threshold 0.02

Press Ctrl+C to exit.
"""

import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE_IN = 16000
SAMPLE_RATE_OUT = 24000
CHUNK_DURATION = 0.1  # 100ms mic chunks


class RealtimeVoiceClient:
    def __init__(self, url, voice, vad_threshold, silence_duration):
        self.url = url
        self.voice = voice
        self.vad_threshold = vad_threshold
        self.silence_duration = silence_duration

        self.ws = None
        self.out_stream = None
        self.mic_stream = None

        # State
        self.is_speaking = False       # user is speaking
        self.is_responding = False     # server is generating
        self.is_playing_audio = False  # speaker is outputting audio
        self.playback_end_time = 0    # when playback stopped (for cooldown)
        self.echo_cooldown = 0.5      # seconds to suppress VAD after playback
        self.speech_start_time = None
        self.last_voice_time = 0
        self.audio_chunks_received = 0
        self.first_audio_time = None
        self.response_text = ""

    async def run(self):
        try:
            import websockets
        except ImportError:
            print("Install websockets: pip install websockets")
            sys.exit(1)

        print()
        print("=" * 60)
        print("  Realtime Voice — speak anytime, barge-in supported")
        print(f"  VAD threshold: {self.vad_threshold}")
        print(f"  Silence to commit: {self.silence_duration}s")
        print("  Press Ctrl+C to exit")
        print("=" * 60)
        print()

        # Open audio output stream
        self.out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE_OUT, channels=1, dtype='float32'
        )
        self.out_stream.start()

        async with websockets.connect(self.url) as ws:
            self.ws = ws

            # Wait for session.created
            msg = json.loads(await ws.recv())
            print(f"[session] {msg.get('session', {}).get('id', '?')}")

            # Configure
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"voice": self.voice},
            }))
            await ws.recv()  # session.updated

            # Run receiver and mic sender concurrently
            receiver_task = asyncio.create_task(self._receive_loop())
            mic_task = asyncio.create_task(self._mic_loop())

            try:
                await asyncio.gather(receiver_task, mic_task)
            except asyncio.CancelledError:
                pass

        if self.out_stream:
            self.out_stream.stop()
            self.out_stream.close()

    async def _receive_loop(self):
        """Receive and handle server events."""
        async for message in self.ws:
            event = json.loads(message)
            etype = event.get("type", "")

            if etype == "response.created":
                self.is_responding = True
                self.audio_chunks_received = 0
                self.first_audio_time = None
                self.response_text = ""

            elif etype == "response.text.delta":
                delta = event.get("delta", "")
                self.response_text += delta
                print(delta, end="", flush=True)

            elif etype == "response.text.done":
                print()

            elif etype == "response.audio.delta":
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    chunk_bytes = base64.b64decode(audio_b64)
                    chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    chunk_float = chunk_int16.astype(np.float32) / 32768.0
                    self.out_stream.write(chunk_float.reshape(-1, 1))
                    self.is_playing_audio = True
                    self.audio_chunks_received += 1
                    if self.first_audio_time is None:
                        self.first_audio_time = time.time()

            elif etype == "response.audio.truncated":
                self.is_playing_audio = False
                self.playback_end_time = time.time()
                audio_end_ms = event.get("audio_end_ms", 0)
                chunks = event.get("chunks_played", 0)
                print(f"\n[barge-in] truncated at {audio_end_ms}ms ({chunks} chunks)")

            elif etype == "response.audio.done":
                self.is_playing_audio = False
                self.playback_end_time = time.time()

            elif etype == "response.done":
                self.is_responding = False
                status = event.get("response", {}).get("status", "completed")
                if status == "completed" and self.audio_chunks_received > 0:
                    duration = self.audio_chunks_received * 3200 / SAMPLE_RATE_OUT
                    print(f"[done] {duration:.1f}s audio")
                print()
                print("Listening...", flush=True)

            elif etype == "error":
                err = event.get("error", {})
                print(f"\n[ERROR] {err.get('type')}: {err.get('message')}")

    async def _mic_loop(self):
        """Continuously capture mic audio, detect speech, commit on silence."""
        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def _callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy())

        self.mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE_IN,
            channels=1,
            dtype='float32',
            blocksize=int(SAMPLE_RATE_IN * CHUNK_DURATION),
            callback=_callback,
        )
        self.mic_stream.start()

        print("Listening...", flush=True)
        pending_audio = b""
        was_speaking = False

        try:
            while True:
                chunk = await audio_queue.get()
                chunk_flat = chunk.flatten()
                energy = float(np.abs(chunk_flat).mean())

                now = time.time()

                # Suppress VAD while speaker is playing (echo cancellation)
                # Use a higher threshold during cooldown period after playback
                in_cooldown = (now - self.playback_end_time) < self.echo_cooldown
                if self.is_playing_audio or in_cooldown:
                    # During playback/cooldown, require much louder input
                    # to trigger barge-in (real speech over speaker output)
                    voice_detected = energy > self.vad_threshold * 5
                else:
                    voice_detected = energy > self.vad_threshold

                if voice_detected:
                    self.last_voice_time = now

                    if not was_speaking:
                        was_speaking = True
                        self.speech_start_time = now
                        pending_audio = b""

                        # If server is responding, this is barge-in
                        if self.is_responding:
                            print("\n[barge-in] interrupting...")

                    # Convert to PCM16 and buffer
                    chunk_int16 = (chunk_flat * 32767).astype(np.int16)
                    pending_audio += chunk_int16.tobytes()

                    # Stream to server
                    audio_b64 = base64.b64encode(chunk_int16.tobytes()).decode("ascii")
                    await self.ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }))

                elif was_speaking:
                    # Silence after speech — check if enough silence to commit
                    silence_elapsed = now - self.last_voice_time
                    if silence_elapsed >= self.silence_duration:
                        was_speaking = False
                        if pending_audio:
                            speech_duration = now - self.speech_start_time
                            print(f"[commit] {speech_duration:.1f}s speech")
                            await self.ws.send(json.dumps({
                                "type": "input_audio_buffer.commit",
                            }))
                            pending_audio = b""
                    else:
                        # Still in silence window — keep buffering
                        chunk_int16 = (chunk_flat * 32767).astype(np.int16)
                        pending_audio += chunk_int16.tobytes()
                        audio_b64 = base64.b64encode(chunk_int16.tobytes()).decode("ascii")
                        await self.ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64,
                        }))

        except asyncio.CancelledError:
            pass
        finally:
            self.mic_stream.stop()
            self.mic_stream.close()


def main():
    parser = argparse.ArgumentParser(description="Realtime voice client with barge-in")
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--voice", default="en-Davis_man")
    parser.add_argument("--vad-threshold", type=float, default=0.015,
                        help="Energy threshold for voice activity detection (default 0.015)")
    parser.add_argument("--silence-duration", type=float, default=0.8,
                        help="Seconds of silence before committing audio (default 0.8)")
    args = parser.parse_args()

    client = RealtimeVoiceClient(args.url, args.voice, args.vad_threshold, args.silence_duration)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n\nExiting.")


if __name__ == "__main__":
    main()
