import sys
from pathlib import Path

import torch
import torch._inductor.config  # noqa: F401
import torchaudio


# -------------------- Quick Config (no CLI args) --------------------
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
OUTPUT_WAV = "output/quick_tts.wav"

# Choose one:
# - "zero_shot"      : inference_zero_shot(tts_text, prompt_text, prompt_wav, ...)
# - "instruct2"      : inference_instruct2(tts_text, instruct_text, prompt_wav, ...)
# - "cross_lingual"  : inference_cross_lingual(tts_text, prompt_wav, ...)
# - "sft"            : inference_sft(tts_text, spk_id, ...)
# - "vc"             : inference_vc(source_wav, prompt_wav, ...)
SYNTH_MODE = "zero_shot"

TTS_TEXT = "This is a quick local TTS test."
PROMPT_WAV = "raw/merged_prompt.wav"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>Hello, nice to meet you."
INSTRUCT_TEXT = "You are a helpful assistant.<|endofprompt|>"
SFT_SPK_ID = "中文女"
SOURCE_WAV = "asset/cross_lingual_prompt.wav"
ZERO_SHOT_SPK_ID = ""

STREAM = False
TEXT_FRONTEND = True
SPEED = 1.0

LOAD_VLLM = True
LOAD_TRT = False
FP16 = False
# --------------------------------------------------------------------


def prepare_import_path(project_root: Path) -> None:
    matcha_dir = project_root / "third_party" / "Matcha-TTS"
    if str(matcha_dir) not in sys.path:
        sys.path.insert(0, str(matcha_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def prepare_vllm_registry() -> None:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)


def resolve_optional(project_root: Path, p: str) -> str:
    if not p:
        return ""
    return str(project_root / p)


def build_iterator(cosyvoice, project_root: Path):
    prompt_wav = resolve_optional(project_root, PROMPT_WAV)
    source_wav = resolve_optional(project_root, SOURCE_WAV)

    if SYNTH_MODE == "zero_shot":
        if not ZERO_SHOT_SPK_ID and (not PROMPT_TEXT or not prompt_wav):
            raise ValueError("zero_shot requires PROMPT_TEXT + PROMPT_WAV, or ZERO_SHOT_SPK_ID.")
        return cosyvoice.inference_zero_shot(
            TTS_TEXT,
            PROMPT_TEXT,
            prompt_wav,
            zero_shot_spk_id=ZERO_SHOT_SPK_ID,
            stream=STREAM,
            speed=SPEED,
            text_frontend=TEXT_FRONTEND,
        )

    if SYNTH_MODE == "instruct2":
        if not INSTRUCT_TEXT or not prompt_wav:
            raise ValueError("instruct2 requires INSTRUCT_TEXT + PROMPT_WAV.")
        return cosyvoice.inference_instruct2(
            TTS_TEXT,
            INSTRUCT_TEXT,
            prompt_wav,
            zero_shot_spk_id=ZERO_SHOT_SPK_ID,
            stream=STREAM,
            speed=SPEED,
            text_frontend=TEXT_FRONTEND,
        )

    if SYNTH_MODE == "cross_lingual":
        if not prompt_wav:
            raise ValueError("cross_lingual requires PROMPT_WAV.")
        return cosyvoice.inference_cross_lingual(
            TTS_TEXT,
            prompt_wav,
            zero_shot_spk_id=ZERO_SHOT_SPK_ID,
            stream=STREAM,
            speed=SPEED,
            text_frontend=TEXT_FRONTEND,
        )

    if SYNTH_MODE == "sft":
        if not SFT_SPK_ID:
            raise ValueError("sft requires SFT_SPK_ID.")
        return cosyvoice.inference_sft(
            TTS_TEXT,
            SFT_SPK_ID,
            stream=STREAM,
            speed=SPEED,
            text_frontend=TEXT_FRONTEND,
        )

    if SYNTH_MODE == "vc":
        if not source_wav or not prompt_wav:
            raise ValueError("vc requires SOURCE_WAV + PROMPT_WAV.")
        return cosyvoice.inference_vc(
            source_wav,
            prompt_wav,
            stream=STREAM,
            speed=SPEED,
        )

    raise ValueError(f"Unsupported SYNTH_MODE: {SYNTH_MODE}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    prepare_import_path(project_root)

    if LOAD_VLLM:
        prepare_vllm_registry()

    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = project_root / MODEL_DIR
    output_wav = project_root / OUTPUT_WAV
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(
        model_dir=str(model_dir),
        load_vllm=LOAD_VLLM,
        load_trt=LOAD_TRT,
        fp16=FP16,
    )

    iterator = build_iterator(cosyvoice, project_root)

    chunks = []
    for item in iterator:
        speech = item.get("tts_speech")
        if speech is not None and speech.numel() > 0:
            chunks.append(speech)

    if not chunks:
        raise RuntimeError("No speech generated.")

    wav = torch.cat(chunks, dim=1)
    torchaudio.save(str(output_wav), wav, cosyvoice.sample_rate)
    print(f"done: {output_wav}")


if __name__ == "__main__":
    main()
