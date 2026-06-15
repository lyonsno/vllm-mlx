# SPDX-License-Identifier: Apache-2.0
"""
Realtime voice endpoint — WebSocket-based bidirectional audio streaming.

OpenAI Realtime API shape with barge-in support:
  - Client streams audio continuously via input_audio_buffer.append
  - Server streams text deltas + audio chunks concurrently
  - Barge-in: new audio commit during response → truncate + restart
  - Truncation tracking: server reports exactly how much audio played

Architecture:
  - Audio input → Gemma 4 (native audio modality via MLLM, no Whisper)
  - Text generation → streamed back as text deltas
  - Text → VibeVoice-Realtime-0.5B (MLX native diffusion TTS)
  - Audio output → streamed back as audio chunks

Concurrency model:
  - Receiver task: reads all client WebSocket messages
  - Response task: runs Gemma 4 + VibeVoice, sends events
  - Barge-in: receiver signals cancellation, response task truncates
"""

import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE_IN = 16000   # Gemma 4 expects 16kHz
SAMPLE_RATE_OUT = 24000  # VibeVoice outputs 24kHz
SAMPLES_PER_CHUNK = 3200  # VibeVoice: one latent frame = 3200 samples
MS_PER_CHUNK = int(SAMPLES_PER_CHUNK / SAMPLE_RATE_OUT * 1000)  # ~133ms


@dataclass
class RealtimeSession:
    """State for one WebSocket realtime session."""
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    conversation: list = field(default_factory=list)
    audio_buffer: bytes = b""
    model_name: str = "mlx-community/gemma-4-12B-it-4bit"
    tts_voice: str = "en-Davis_man"
    cfg_scale: float = 1.5
    diffusion_steps: int = 5
    max_response_tokens: int = 512
    temperature: float = 0.3

    # Concurrency state
    response_task: Optional[asyncio.Task] = field(default=None, repr=False)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    # Truncation tracking
    audio_chunks_sent: int = 0
    response_text_so_far: str = ""


class RealtimeHandler:
    """Handles one WebSocket realtime session with barge-in.

    Concurrent architecture:
      - Receiver loop runs continuously, processing client events
      - Response generation runs as a separate task
      - Barge-in: new audio commit cancels in-flight response
    """

    def __init__(self, mllm_engine=None):
        self.mllm_engine = mllm_engine
        self._tts_model = None
        self._tts_voice_prompt = None

    async def handle_websocket(self, websocket):
        """Main WebSocket handler — runs receiver loop."""
        session = RealtimeSession()

        await self._send_event(websocket, "session.created", {
            "session": {
                "id": session.session_id,
                "model": session.model_name,
                "voice": session.tts_voice,
            }
        })

        try:
            async for message in websocket.iter_text():
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "invalid_json", "Could not parse message")
                    continue

                event_type = event.get("type", "")
                await self._dispatch_event(websocket, session, event_type, event)

        except Exception as e:
            logger.error(f"WebSocket error in session {session.session_id}: {e}")
        finally:
            # Clean up any running response
            await self._cancel_response(session)

    async def _dispatch_event(self, websocket, session, event_type, event):
        """Route client events."""

        if event_type == "session.update":
            config = event.get("session", {})
            for key in ("model", "voice", "temperature"):
                if key in config:
                    mapped = {"model": "model_name", "voice": "tts_voice"}.get(key, key)
                    setattr(session, mapped, config[key])
            if "max_response_output_tokens" in config:
                session.max_response_tokens = config["max_response_output_tokens"]
            await self._send_event(websocket, "session.updated", {
                "session": {"id": session.session_id}
            })

        elif event_type == "input_audio_buffer.append":
            audio_b64 = event.get("audio", "")
            if audio_b64:
                session.audio_buffer += base64.b64decode(audio_b64)

        elif event_type == "input_audio_buffer.commit":
            # BARGE-IN: if a response is running, cancel it first
            if session.response_task and not session.response_task.done():
                await self._cancel_response(session)

            await self._handle_audio_commit(websocket, session)

        elif event_type == "input_audio_buffer.clear":
            session.audio_buffer = b""
            await self._send_event(websocket, "input_audio_buffer.cleared", {})

        elif event_type == "response.cancel":
            await self._cancel_response(session)
            await self._send_event(websocket, "response.cancelled", {})

        elif event_type == "conversation.item.create":
            item = event.get("item", {})
            session.conversation.append(item)
            await self._send_event(websocket, "conversation.item.created", {"item": item})

        elif event_type == "response.create":
            if session.response_task and not session.response_task.done():
                await self._cancel_response(session)
            session.response_task = asyncio.create_task(
                self._generate_response(websocket, session)
            )

    async def _cancel_response(self, session):
        """Cancel in-flight response and report truncation."""
        if session.response_task and not session.response_task.done():
            session.cancel_event.set()
            try:
                await asyncio.wait_for(session.response_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                session.response_task.cancel()
            session.response_task = None
        session.cancel_event.clear()

    async def _handle_audio_commit(self, websocket, session):
        """Process committed audio buffer — start response generation."""
        if not session.audio_buffer:
            await self._send_error(websocket, "empty_audio", "No audio in buffer")
            return

        audio_int16 = np.frombuffer(session.audio_buffer, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        session.audio_buffer = b""

        item_id = f"item_{uuid.uuid4().hex[:8]}"
        await self._send_event(websocket, "input_audio_buffer.committed", {
            "item_id": item_id,
        })

        session.conversation.append({
            "id": item_id,
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": audio_float}],
        })

        # Launch response as concurrent task — receiver loop keeps running
        session.cancel_event.clear()
        session.response_task = asyncio.create_task(
            self._generate_response(websocket, session, audio_input=audio_float)
        )

    async def _generate_response(self, websocket, session, audio_input=None):
        """Run Gemma 4 + VibeVoice TTS with barge-in awareness."""
        response_id = f"resp_{uuid.uuid4().hex[:8]}"
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        session.audio_chunks_sent = 0
        session.response_text_so_far = ""

        await self._send_event(websocket, "response.created", {
            "response": {"id": response_id}
        })

        try:
            # Generate text
            text_response = await self._run_gemma4(
                websocket, session, response_id, item_id, audio_input
            )

            if text_response and not session.cancel_event.is_set():
                # Stream TTS audio
                await self._run_tts(
                    websocket, session, response_id, item_id, text_response
                )

            # If we were cancelled (barge-in), send truncation info
            if session.cancel_event.is_set():
                audio_end_ms = session.audio_chunks_sent * MS_PER_CHUNK
                await self._send_event(websocket, "response.audio.truncated", {
                    "response_id": response_id,
                    "item_id": item_id,
                    "audio_end_ms": audio_end_ms,
                    "chunks_played": session.audio_chunks_sent,
                })

                # Truncate the assistant message in conversation history
                # to only what was actually heard
                if session.response_text_so_far:
                    truncated_item = {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": session.response_text_so_far}],
                        "truncated": True,
                        "audio_end_ms": audio_end_ms,
                    }
                    # Replace the last assistant message if it exists
                    for i in range(len(session.conversation) - 1, -1, -1):
                        if session.conversation[i].get("id") == item_id:
                            session.conversation[i] = truncated_item
                            break
                    else:
                        session.conversation.append(truncated_item)

            await self._send_event(websocket, "response.done", {
                "response": {
                    "id": response_id,
                    "status": "cancelled" if session.cancel_event.is_set() else "completed",
                }
            })

        except asyncio.CancelledError:
            logger.info(f"Response {response_id} cancelled")
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(websocket, "generation_error", str(e))

    async def _run_gemma4(self, websocket, session, response_id, item_id, audio_input):
        """Run Gemma 4 audio understanding, streaming text deltas. Cancel-aware."""
        await self._send_event(websocket, "response.output_item.added", {
            "response_id": response_id,
            "item": {"id": item_id, "type": "message", "role": "assistant"},
        })
        await self._send_event(websocket, "response.content_part.added", {
            "response_id": response_id,
            "item_id": item_id,
            "part": {"type": "text", "text": ""},
        })

        full_text = ""

        try:
            text_gen = await asyncio.to_thread(
                self._gemma4_generate_sync, session, audio_input
            )

            for delta in text_gen:
                if session.cancel_event.is_set():
                    break

                new_text = delta[len(full_text):]
                if new_text:
                    full_text = delta
                    session.response_text_so_far = full_text
                    await self._send_event(websocket, "response.text.delta", {
                        "response_id": response_id,
                        "item_id": item_id,
                        "delta": new_text,
                    })

        except Exception as e:
            logger.error(f"Gemma 4 generation error: {e}")
            await self._send_error(websocket, "model_error", str(e))

        if full_text and not session.cancel_event.is_set():
            await self._send_event(websocket, "response.text.done", {
                "response_id": response_id,
                "item_id": item_id,
                "text": full_text,
            })

            session.conversation.append({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
            })

        return full_text

    _gemma4_model = None
    _gemma4_processor = None

    def _build_context_prompt(self, session):
        """Build a prompt that includes conversation history for multi-turn."""
        parts = []
        for item in session.conversation:
            role = item.get("role", "")
            content = item.get("content", [])
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                text = " ".join(text_parts)
            else:
                text = ""

            if role == "user" and text:
                parts.append(f"User: {text}")
            elif role == "assistant" and text:
                truncated = " [interrupted]" if item.get("truncated") else ""
                parts.append(f"Assistant: {text}{truncated}")

        if parts:
            history = "\n".join(parts[-6:])  # last 3 turns max
            return (
                f"Previous conversation:\n{history}\n\n"
                f"Listen to the user's new audio message and respond naturally, "
                f"taking the conversation history into account."
            )
        return "Listen to this audio and respond naturally."

    def _gemma4_generate_sync(self, session, audio_input):
        """Synchronous Gemma 4 generation in thread."""
        from mlx_vlm.tools.gemma4_audio.core import load_model
        from mlx_vlm.tools.gemma4_audio.prompt import build_prompt
        from mlx_vlm.tools.gemma4_audio.inference import run_inference

        if self._gemma4_model is None:
            logger.info(f"Loading Gemma 4 audio model: {session.model_name}")
            self._gemma4_model, self._gemma4_processor = load_model(session.model_name)
            logger.info("Gemma 4 audio model loaded")

        model = self._gemma4_model
        processor = self._gemma4_processor
        prompt_fn = lambda text: build_prompt(processor, model.config, text)
        prompt = self._build_context_prompt(session)

        results = []
        for text in run_inference(
            model, processor, audio_input, prompt,
            max_tokens=session.max_response_tokens,
            temperature=session.temperature,
            prompt_builder=prompt_fn,
        ):
            results.append(text)

        return results

    async def _run_tts(self, websocket, session, response_id, item_id, text):
        """Stream VibeVoice TTS audio chunks. Cancel-aware."""
        await self._send_event(websocket, "response.audio.started", {
            "response_id": response_id,
            "item_id": item_id,
        })

        chunk_queue = asyncio.Queue()
        _DONE = object()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                for chunk in self._tts_generate_streaming(session, text):
                    if session.cancel_event.is_set():
                        break
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, _DONE)

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        try:
            while True:
                item = await chunk_queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item

                if session.cancel_event.is_set():
                    break

                chunk_int16 = (item * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(chunk_int16.tobytes()).decode("ascii")

                await self._send_event(websocket, "response.audio.delta", {
                    "response_id": response_id,
                    "item_id": item_id,
                    "delta": audio_b64,
                })
                session.audio_chunks_sent += 1

        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self._send_error(websocket, "tts_error", str(e))

        producer.join(timeout=5)

        if not session.cancel_event.is_set():
            await self._send_event(websocket, "response.audio.done", {
                "response_id": response_id,
                "item_id": item_id,
            })

    def _tts_generate_streaming(self, session, text):
        """VibeVoice generation as a generator — yields audio chunks."""
        import copy
        import mlx.core as mx

        if self._tts_model is None:
            from mlx_vlm.tools.gemma4_audio.vibevoice_mlx import (
                load_vibevoice, convert_voice_prompt,
            )
            self._tts_model, self._tts_config = load_vibevoice()
            import os
            voice_path = None
            candidates = [
                f"/private/tmp/vibevoice-ref/demo/voices/streaming_model/{session.tts_voice}.pt",
                os.path.expanduser(f"~/.cache/vibevoice/voices/{session.tts_voice}.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    voice_path = c
                    break
            if voice_path is None:
                from huggingface_hub import hf_hub_download
                voice_path = hf_hub_download(
                    "microsoft/VibeVoice-Realtime-0.5B",
                    f"demo/voices/streaming_model/{session.tts_voice}.pt",
                )
            self._tts_voice_prompt = convert_voice_prompt(voice_path)

        from mlx_vlm.tools.gemma4_audio.vibevoice_mlx import (
            KVCache, StreamingCache,
            TTS_TEXT_WINDOW_SIZE, TTS_SPEECH_WINDOW_SIZE,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        def _restore_cache(kv_list):
            caches = []
            for k, v in kv_list:
                c = KVCache()
                c.keys = k
                c.values = v
                c.offset = k.shape[2]
                caches.append(c)
            return caches

        vp = copy.deepcopy(self._tts_voice_prompt)
        model = self._tts_model
        state = {
            "lm_cache": _restore_cache(vp["lm"]["kv_cache"]),
            "tts_lm_cache": _restore_cache(vp["tts_lm"]["kv_cache"]),
            "neg_lm_cache": _restore_cache(vp["neg_lm"]["kv_cache"]),
            "neg_tts_lm_cache": _restore_cache(vp["neg_tts_lm"]["kv_cache"]),
            "lm_hidden": vp["lm"]["last_hidden_state"],
            "tts_lm_hidden": vp["tts_lm"]["last_hidden_state"],
            "neg_tts_lm_hidden": vp["neg_tts_lm"]["last_hidden_state"],
            "acoustic_cache": StreamingCache(),
        }

        tokens = tokenizer.encode(text.strip() + "\n", add_special_tokens=False)
        i = 0

        while i < len(tokens):
            if session.cancel_event.is_set():
                return

            window = tokens[i:i + TTS_TEXT_WINDOW_SIZE]
            i += TTS_TEXT_WINDOW_SIZE
            is_last_window = (i >= len(tokens))
            text_ids = mx.array([window])

            cur_embeds = model.language_model.embed_tokens(text_ids)
            state["lm_hidden"] = model.language_model(inputs_embeds=cur_embeds, cache=state["lm_cache"])
            mx.eval(state["lm_hidden"])

            tts_embeds = model.tts_language_model.embed_tokens(text_ids)
            splice_start = tts_embeds.shape[1] - state["lm_hidden"].shape[1]
            if splice_start > 0:
                tts_embeds = mx.concatenate([tts_embeds[:, :splice_start], state["lm_hidden"]], axis=1)
            else:
                tts_embeds = state["lm_hidden"]
            type_embed = model.tts_input_types(mx.ones(tts_embeds.shape[:2], dtype=mx.int32))
            tts_embeds = tts_embeds + type_embed
            state["tts_lm_hidden"] = model.tts_language_model(inputs_embeds=tts_embeds, cache=state["tts_lm_cache"])
            mx.eval(state["tts_lm_hidden"])

            for _ in range(TTS_SPEECH_WINDOW_SIZE):
                if session.cancel_event.is_set():
                    return

                pos_cond = state["tts_lm_hidden"][:, -1:, :].reshape(1, -1)
                neg_cond = state["neg_tts_lm_hidden"][:, -1:, :].reshape(1, -1)

                speech_latent = model.sample_speech_tokens(
                    pos_cond, neg_cond,
                    cfg_scale=session.cfg_scale,
                    num_steps=session.diffusion_steps,
                )

                scaled = speech_latent.reshape(1, 1, -1) / model.speech_scaling_factor - model.speech_bias_factor
                scaled_for_decode = mx.transpose(scaled, (0, 2, 1))
                audio_chunk = model.acoustic_decoder(scaled_for_decode, cache=state["acoustic_cache"])
                mx.eval(audio_chunk)

                yield np.array(audio_chunk.reshape(-1), dtype=np.float32)

                acoustic_embed = model.acoustic_connector(speech_latent.reshape(1, 1, -1))
                type_embed_sp = model.tts_input_types(mx.zeros((1, 1), dtype=mx.int32))
                tts_input = acoustic_embed + type_embed_sp
                state["tts_lm_hidden"] = model.tts_language_model(inputs_embeds=tts_input, cache=state["tts_lm_cache"])
                state["neg_tts_lm_hidden"] = model.tts_language_model(inputs_embeds=tts_input, cache=state["neg_tts_lm_cache"])
                mx.eval(state["tts_lm_hidden"], state["neg_tts_lm_hidden"])

                if is_last_window:
                    eos = model.eos_classifier(state["tts_lm_hidden"][:, -1, :])
                    if mx.sigmoid(eos).item() > 0.5:
                        return

    async def _send_event(self, websocket, event_type, data):
        event = {"type": event_type, **data}
        await websocket.send_text(json.dumps(event))

    async def _send_error(self, websocket, code, message):
        await self._send_event(websocket, "error", {
            "error": {"type": code, "message": message}
        })
