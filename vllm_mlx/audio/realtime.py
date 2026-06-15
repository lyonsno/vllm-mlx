# SPDX-License-Identifier: Apache-2.0
"""
Realtime voice endpoint — WebSocket-based bidirectional audio streaming.

Minimal OpenAI Realtime API shape:
  - Client sends audio chunks (base64 PCM16 @ 16kHz)
  - Server responds with text deltas + audio chunks (base64 PCM16 @ 24kHz)
  - Barge-in: new audio input truncates in-progress response

Architecture:
  - Audio input → Gemma 4 (native audio modality via MLLM, no Whisper)
  - Text generation → streamed back as text deltas
  - Text → VibeVoice-Realtime-0.5B (MLX native diffusion TTS)
  - Audio output → streamed back as audio chunks
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE_IN = 16000   # Gemma 4 expects 16kHz
SAMPLE_RATE_OUT = 24000  # VibeVoice outputs 24kHz


@dataclass
class RealtimeSession:
    """State for one WebSocket realtime session."""
    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    conversation: list = field(default_factory=list)
    audio_buffer: bytes = b""
    is_generating: bool = False
    should_cancel: bool = False
    input_audio_sr: int = SAMPLE_RATE_IN
    output_audio_sr: int = SAMPLE_RATE_OUT
    model_name: str = "mlx-community/gemma-4-12B-it-4bit"
    tts_voice: str = "en-Davis_man"
    cfg_scale: float = 1.5
    diffusion_steps: int = 5
    max_response_tokens: int = 512
    temperature: float = 0.3


class RealtimeHandler:
    """Handles one WebSocket realtime session.

    Integrates:
      - mlx-vlm MLLM for Gemma 4 audio understanding
      - VibeVoice MLX for streaming TTS output
    """

    def __init__(self, mllm_engine=None):
        self.mllm_engine = mllm_engine
        self._tts_model = None
        self._tts_voice_prompt = None

    async def handle_websocket(self, websocket):
        """Main WebSocket handler — dispatches client events."""
        session = RealtimeSession()

        # Send session.created
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
                    await self._send_error(websocket, "invalid_json", "Could not parse message as JSON")
                    continue

                event_type = event.get("type", "")
                await self._dispatch_event(websocket, session, event_type, event)

        except Exception as e:
            logger.error(f"WebSocket error in session {session.session_id}: {e}")

    async def _dispatch_event(self, websocket, session, event_type, event):
        """Route client events to handlers."""

        if event_type == "session.update":
            config = event.get("session", {})
            if "model" in config:
                session.model_name = config["model"]
            if "voice" in config:
                session.tts_voice = config["voice"]
            if "temperature" in config:
                session.temperature = config["temperature"]
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
            await self._handle_audio_commit(websocket, session)

        elif event_type == "input_audio_buffer.clear":
            session.audio_buffer = b""
            await self._send_event(websocket, "input_audio_buffer.cleared", {})

        elif event_type == "conversation.item.create":
            item = event.get("item", {})
            session.conversation.append(item)
            await self._send_event(websocket, "conversation.item.created", {"item": item})

        elif event_type == "response.create":
            # Trigger generation from conversation history (text mode)
            asyncio.create_task(self._generate_response(websocket, session))

        elif event_type == "response.cancel":
            session.should_cancel = True
            await self._send_event(websocket, "response.cancelled", {})

        else:
            logger.debug(f"Unhandled event type: {event_type}")

    async def _handle_audio_commit(self, websocket, session):
        """Process committed audio buffer — run Gemma 4 + VibeVoice response."""
        if not session.audio_buffer:
            await self._send_error(websocket, "empty_audio", "No audio in buffer")
            return

        # Convert PCM16 bytes to float32 numpy
        audio_int16 = np.frombuffer(session.audio_buffer, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        session.audio_buffer = b""

        await self._send_event(websocket, "input_audio_buffer.committed", {
            "item_id": f"item_{uuid.uuid4().hex[:8]}",
        })

        # Add to conversation as user audio
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        session.conversation.append({
            "id": item_id,
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": audio_float}],
        })

        # Generate response
        await self._generate_response(websocket, session, audio_input=audio_float)

    async def _generate_response(self, websocket, session, audio_input=None):
        """Run Gemma 4 generation + VibeVoice TTS streaming."""
        if session.is_generating:
            session.should_cancel = True
            # Wait briefly for cancellation
            for _ in range(10):
                if not session.is_generating:
                    break
                await asyncio.sleep(0.05)

        session.is_generating = True
        session.should_cancel = False

        response_id = f"resp_{uuid.uuid4().hex[:8]}"
        item_id = f"item_{uuid.uuid4().hex[:8]}"

        await self._send_event(websocket, "response.created", {
            "response": {"id": response_id}
        })

        try:
            # Generate text from audio using Gemma 4 via MLLM
            text_response = await self._run_gemma4(
                websocket, session, response_id, item_id, audio_input
            )

            if text_response and not session.should_cancel:
                # Stream TTS audio
                await self._run_tts(
                    websocket, session, response_id, item_id, text_response
                )

            await self._send_event(websocket, "response.done", {
                "response": {"id": response_id}
            })

        except Exception as e:
            logger.error(f"Generation error: {e}")
            await self._send_error(websocket, "generation_error", str(e))

        finally:
            session.is_generating = False

    async def _run_gemma4(self, websocket, session, response_id, item_id, audio_input):
        """Run Gemma 4 audio understanding and stream text deltas."""
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
            # Use mlx_vlm to process audio through Gemma 4
            text_gen = await asyncio.to_thread(
                self._gemma4_generate_sync, session, audio_input
            )

            for delta in text_gen:
                if session.should_cancel:
                    break

                new_text = delta[len(full_text):]
                if new_text:
                    full_text = delta
                    await self._send_event(websocket, "response.text.delta", {
                        "response_id": response_id,
                        "item_id": item_id,
                        "delta": new_text,
                    })

        except Exception as e:
            logger.error(f"Gemma 4 generation error: {e}")
            await self._send_error(websocket, "model_error", str(e))

        if full_text:
            await self._send_event(websocket, "response.text.done", {
                "response_id": response_id,
                "item_id": item_id,
                "text": full_text,
            })

            # Add to conversation history
            session.conversation.append({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
            })

        return full_text

    _gemma4_model = None
    _gemma4_processor = None

    def _gemma4_generate_sync(self, session, audio_input):
        """Synchronous Gemma 4 generation — runs in thread.

        Uses mlx_vlm Gemma 4 audio tools directly for native audio
        understanding (no Whisper, no text-only path).
        """
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
        prompt = "Listen to this audio and respond naturally."

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
        """Run VibeVoice TTS and stream audio chunks."""
        await self._send_event(websocket, "response.audio.started", {
            "response_id": response_id,
            "item_id": item_id,
        })

        try:
            chunks = await asyncio.to_thread(
                self._tts_generate_sync, session, text
            )

            for chunk_float32 in chunks:
                if session.should_cancel:
                    await self._send_event(websocket, "response.audio.truncated", {
                        "response_id": response_id,
                        "item_id": item_id,
                    })
                    break

                # Convert float32 → PCM16 → base64
                chunk_int16 = (chunk_float32 * 32767).astype(np.int16)
                audio_b64 = base64.b64encode(chunk_int16.tobytes()).decode("ascii")

                await self._send_event(websocket, "response.audio.delta", {
                    "response_id": response_id,
                    "item_id": item_id,
                    "delta": audio_b64,
                })

        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self._send_error(websocket, "tts_error", str(e))

        await self._send_event(websocket, "response.audio.done", {
            "response_id": response_id,
            "item_id": item_id,
        })

    def _tts_generate_sync(self, session, text):
        """Synchronous VibeVoice generation — returns list of audio chunks."""
        import copy
        import mlx.core as mx

        if self._tts_model is None:
            from mlx_vlm.tools.gemma4_audio.vibevoice_mlx import (
                load_vibevoice, convert_voice_prompt, KVCache, StreamingCache,
                TTS_TEXT_WINDOW_SIZE, TTS_SPEECH_WINDOW_SIZE,
            )
            self._tts_model, self._tts_config = load_vibevoice()
            # Find and load voice prompt
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
            generate, KVCache, StreamingCache,
            TTS_TEXT_WINDOW_SIZE, TTS_SPEECH_WINDOW_SIZE,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        # Build generation state from voice prompt
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
        chunks = []
        i = 0
        total_windows = (len(tokens) + TTS_TEXT_WINDOW_SIZE - 1) // TTS_TEXT_WINDOW_SIZE
        window_idx = 0

        while i < len(tokens):
            window = tokens[i:i + TTS_TEXT_WINDOW_SIZE]
            i += TTS_TEXT_WINDOW_SIZE
            window_idx += 1
            is_last_window = (i >= len(tokens))
            text_ids = mx.array([window])

            # Base LM
            cur_embeds = model.language_model.embed_tokens(text_ids)
            state["lm_hidden"] = model.language_model(inputs_embeds=cur_embeds, cache=state["lm_cache"])
            mx.eval(state["lm_hidden"])

            # TTS LM
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

                chunks.append(np.array(audio_chunk.reshape(-1), dtype=np.float32))

                # Feed back
                acoustic_embed = model.acoustic_connector(speech_latent.reshape(1, 1, -1))
                type_embed_sp = model.tts_input_types(mx.zeros((1, 1), dtype=mx.int32))
                tts_input = acoustic_embed + type_embed_sp
                state["tts_lm_hidden"] = model.tts_language_model(inputs_embeds=tts_input, cache=state["tts_lm_cache"])
                state["neg_tts_lm_hidden"] = model.tts_language_model(inputs_embeds=tts_input, cache=state["neg_tts_lm_cache"])
                mx.eval(state["tts_lm_hidden"], state["neg_tts_lm_hidden"])

                # Only check EOS after all text has been fed — the model
                # sometimes fires EOS mid-utterance if checked too early
                if is_last_window:
                    eos = model.eos_classifier(state["tts_lm_hidden"][:, -1, :])
                    if mx.sigmoid(eos).item() > 0.5:
                        return chunks

        return chunks

    async def _send_event(self, websocket, event_type, data):
        """Send a server event."""
        event = {"type": event_type, **data}
        await websocket.send_text(json.dumps(event))

    async def _send_error(self, websocket, code, message):
        """Send an error event."""
        await self._send_event(websocket, "error", {
            "error": {"type": code, "message": message}
        })
