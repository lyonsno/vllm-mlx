#!/usr/bin/env python3
"""
Realtime voice client for vllm-mlx /v1/realtime WebSocket endpoint.

Records from mic, sends audio to Gemma 4, plays back VibeVoice response.

Usage:
    # Start server first:
    python -m vllm_mlx.server --model mlx-community/gemma-4-12B-it-4bit

    # Then run client:
    python examples/realtime_voice.py
    python examples/realtime_voice.py --seconds 5
    python examples/realtime_voice.py --url ws://localhost:8000/v1/realtime
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


async def run_realtime(url: str, seconds: float, voice: str):
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        sys.exit(1)

    # Record audio
    print(f"Recording {seconds}s from mic...")
    audio = sd.rec(int(seconds * SAMPLE_RATE_IN), samplerate=SAMPLE_RATE_IN,
                   channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    print(f"Recorded {len(audio)} samples ({len(audio)/SAMPLE_RATE_IN:.1f}s)")

    # Convert to PCM16 bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    print(f"Connecting to {url}...")
    async with websockets.connect(url) as ws:
        # Wait for session.created
        msg = json.loads(await ws.recv())
        print(f"Session: {msg.get('session', {}).get('id', 'unknown')}")

        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"voice": voice},
        }))
        await ws.recv()  # session.updated

        # Send audio in chunks (64KB each)
        chunk_size = 64 * 1024
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }))

        # Commit
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("Audio sent, waiting for response...\n")

        # Collect response — play audio as it streams
        full_text = ""
        total_audio_samples = 0
        t0 = time.time()
        t_first_audio = None

        # Open output stream for immediate playback
        out_stream = sd.OutputStream(samplerate=SAMPLE_RATE_OUT, channels=1, dtype='float32')
        out_stream.start()

        async for message in ws:
            event = json.loads(message)
            etype = event.get("type", "")

            if etype == "response.text.delta":
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                full_text += delta

            elif etype == "response.text.done":
                print(f"\n\n[text done in {time.time()-t0:.1f}s]")

            elif etype == "response.audio.delta":
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    chunk_bytes = base64.b64decode(audio_b64)
                    chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    chunk_float = chunk_int16.astype(np.float32) / 32768.0
                    # Play immediately
                    out_stream.write(chunk_float.reshape(-1, 1))
                    total_audio_samples += len(chunk_float)
                    if t_first_audio is None:
                        t_first_audio = time.time()
                        print(f"[first audio chunk at {t_first_audio - t0:.1f}s]")

            elif etype == "response.audio.done":
                total_time = time.time() - t0
                if total_audio_samples > 0:
                    duration = total_audio_samples / SAMPLE_RATE_OUT
                    print(f"[audio done: {duration:.1f}s audio in {total_time:.1f}s, RTF={total_time/duration:.2f}x]")

            elif etype == "response.done":
                break

            elif etype == "error":
                err = event.get("error", {})
                print(f"\n[ERROR] {err.get('type')}: {err.get('message')}")
                break

        # Wait for the sounddevice output buffer to finish playing.
        # OutputStream.write() is non-blocking — audio keeps playing in the
        # OS buffer after the last write. We need to wait for it to drain.
        if total_audio_samples > 0 and t_first_audio is not None:
            # Total audio duration minus time already elapsed since first chunk
            audio_duration = total_audio_samples / SAMPLE_RATE_OUT
            elapsed_since_first = time.time() - t_first_audio
            drain_wait = audio_duration - elapsed_since_first + 0.3
            if drain_wait > 0:
                print(f"[draining {drain_wait:.1f}s of buffered audio...]")
                time.sleep(drain_wait)
        out_stream.stop()
        out_stream.close()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Realtime voice client for vllm-mlx")
    parser.add_argument("--url", default="ws://localhost:8000/v1/realtime")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--voice", default="en-Davis_man")
    args = parser.parse_args()

    asyncio.run(run_realtime(args.url, args.seconds, args.voice))


if __name__ == "__main__":
    main()
