import io
import os
import struct
import sys
import threading
import wave
from typing import Generator

import numpy as np
import torch
import torch._inductor.config  # noqa: F401
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel


sys.path.append("third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel


MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
PROMPT_WAV = os.getenv("COSYVOICE_PROMPT_WAV", "raw/merged_prompt.wav")
PROMPT_TEXT = os.getenv(
    "COSYVOICE_PROMPT_TEXT",
    "Use this prompt audio as speaker reference for stable zero-shot synthesis.",
)
SYSTEM_PREFIX = "You are a helpful assistant.<|endofprompt|>"
SPEAKER_ID = os.getenv("COSYVOICE_SPEAKER_ID", "custom_speaker")

PORT = int(os.getenv("PORT", "9880"))
MAX_TEXT_CHARS = int(os.getenv("COSYVOICE_MAX_TEXT_CHARS", "1200"))
TTS_CONCURRENCY = int(os.getenv("COSYVOICE_TTS_CONCURRENCY", "1"))

MODEL_LOAD_TRT = os.getenv("COSYVOICE_LOAD_TRT", "False").lower() == "true"
MODEL_LOAD_VLLM = os.getenv("COSYVOICE_LOAD_VLLM", "True").lower() == "true"
MODEL_FP16 = os.getenv("COSYVOICE_FP16", "False").lower() == "true"
ALLOW_VLLM_FALLBACK = os.getenv("COSYVOICE_ALLOW_VLLM_FALLBACK", "True").lower() == "true"


app = FastAPI(title="CosyVoice ZeroShot FastAPI")
cosyvoice = None
sample_rate = 24000
tts_semaphore = threading.Semaphore(max(1, TTS_CONCURRENCY))
load_vllm_used = MODEL_LOAD_VLLM


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    text_frontend: bool = False
    split_text: bool = True


def wav_header_streaming(sr: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sr,
        byte_rate,
        block_align,
        bits,
        b"data",
        0xFFFFFFFF,
    )


def speech_to_pcm_bytes(speech_tensor: torch.Tensor) -> bytes:
    pcm = (speech_tensor.detach().cpu().numpy() * (2**15)).astype(np.int16)
    return pcm.tobytes()


def speech_to_wav_bytes(speech_tensor: torch.Tensor, sr: int) -> bytes:
    pcm = (speech_tensor.detach().cpu().numpy() * (2**15)).astype(np.int16)
    if pcm.ndim == 2:
        pcm = pcm[0]
    elif pcm.ndim != 1:
        raise RuntimeError(f"Unexpected speech shape: {tuple(speech_tensor.shape)}")
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buff.getvalue()


def ensure_ready():
    if cosyvoice is None:
        raise HTTPException(status_code=503, detail="model is not ready")


def normalize_segments(text: str, split_text: bool, text_frontend: bool):
    if not split_text:
        return [text]
    return cosyvoice.frontend.text_normalize(text, split=True, text_frontend=text_frontend)


def setup_vllm_registry() -> None:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


def run_zero_shot_segment(seg: str, speed: float, stream: bool, text_frontend: bool):
    return cosyvoice.inference_zero_shot(
        seg,
        SYSTEM_PREFIX + PROMPT_TEXT,
        PROMPT_WAV,
        zero_shot_spk_id=SPEAKER_ID,
        stream=stream,
        speed=speed,
        text_frontend=text_frontend,
    )


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    ensure_ready()
    text = (req.text or "").strip()
    if not text:
        return StreamingResponse(iter([]), media_type="audio/wav")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"text too long, max={MAX_TEXT_CHARS}")

    segments = normalize_segments(text, split_text=req.split_text, text_frontend=req.text_frontend)

    def generate() -> Generator[bytes, None, None]:
        with tts_semaphore:
            yield wav_header_streaming(sample_rate)
            for seg in segments:
                for out in run_zero_shot_segment(
                    seg=seg,
                    speed=req.speed,
                    stream=True,
                    text_frontend=req.text_frontend,
                ):
                    yield speech_to_pcm_bytes(out["tts_speech"])

    return StreamingResponse(generate(), media_type="audio/wav")


@app.post("/tts/complete")
async def tts_complete(req: TTSRequest):
    ensure_ready()
    text = (req.text or "").strip()
    if not text:
        return Response(content=b"", media_type="audio/wav")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f"text too long, max={MAX_TEXT_CHARS}")

    segments = normalize_segments(text, split_text=req.split_text, text_frontend=req.text_frontend)
    all_chunks = []
    with tts_semaphore:
        for seg in segments:
            for out in run_zero_shot_segment(
                seg=seg,
                speed=req.speed,
                stream=False,
                text_frontend=req.text_frontend,
            ):
                all_chunks.append(out["tts_speech"])
    if not all_chunks:
        raise HTTPException(status_code=500, detail="no audio generated")

    speech = torch.cat(all_chunks, dim=1)
    wav_bytes = speech_to_wav_bytes(speech, sample_rate)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/sample_rate")
async def get_sample_rate():
    return {"sample_rate": sample_rate}


@app.get("/health")
async def health():
    return {
        "ready": cosyvoice is not None,
        "sample_rate": sample_rate if cosyvoice is not None else None,
        "mode": "zero_shot_only",
        "speaker_id": SPEAKER_ID,
        "load_vllm": load_vllm_used,
    }


@app.on_event("startup")
def startup_load_model():
    global cosyvoice, sample_rate, load_vllm_used
    load_vllm_used = MODEL_LOAD_VLLM
    if load_vllm_used:
        try:
            setup_vllm_registry()
        except Exception as e:
            if not ALLOW_VLLM_FALLBACK:
                raise RuntimeError(f"vLLM setup failed and fallback disabled: {e}") from e
            print(f"[warn] vLLM setup failed, fallback to non-vLLM mode: {e}")
            load_vllm_used = False

    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=MODEL_LOAD_TRT,
        load_vllm=load_vllm_used,
        fp16=MODEL_FP16,
    )
    # Pre-register speaker id.
    cosyvoice.add_zero_shot_spk(SYSTEM_PREFIX + PROMPT_TEXT, PROMPT_WAV, SPEAKER_ID)
    # Warmup using registered speaker id.
    for _ in cosyvoice.inference_zero_shot(
        "warmup",
        SYSTEM_PREFIX + PROMPT_TEXT,
        PROMPT_WAV,
        zero_shot_spk_id=SPEAKER_ID,
        stream=False,
        text_frontend=False,
    ):
        pass
    sample_rate = cosyvoice.sample_rate


if __name__ == "__main__":
    uvicorn.run("fastapi_zero_shot_server:app", host="0.0.0.0", port=PORT, reload=False)
