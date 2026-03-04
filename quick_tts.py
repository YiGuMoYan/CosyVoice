import sys
from pathlib import Path

import torch
import torch._inductor.config  # noqa: F401
import torchaudio


# -------------------- Quick Config (no CLI args) --------------------
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = "raw/merged_prompt.wav"
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>Hello, nice to meet you."
TTS_TEXT = "This is a zero-argument local TTS test based on vllm_example."
OUTPUT_WAV = "output/quick_tts.wav"

USE_INSTRUCT2 = False
INSTRUCT_TEXT = "You are a helpful assistant.<|endofprompt|>"
ZERO_SHOT_SPK_ID = ""
STREAM = False
TEXT_FRONTEND = True

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


def main() -> None:
    project_root = Path(__file__).resolve().parent
    prepare_import_path(project_root)

    if LOAD_VLLM:
        prepare_vllm_registry()

    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = project_root / MODEL_DIR
    prompt_wav = project_root / PROMPT_WAV
    output_wav = project_root / OUTPUT_WAV
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(
        model_dir=str(model_dir),
        load_vllm=LOAD_VLLM,
        load_trt=LOAD_TRT,
        fp16=FP16,
    )

    if USE_INSTRUCT2:
        iterator = cosyvoice.inference_instruct2(
            TTS_TEXT,
            INSTRUCT_TEXT,
            str(prompt_wav),
            zero_shot_spk_id=ZERO_SHOT_SPK_ID,
            stream=STREAM,
            text_frontend=TEXT_FRONTEND,
        )
    else:
        iterator = cosyvoice.inference_zero_shot(
            TTS_TEXT,
            PROMPT_TEXT,
            str(prompt_wav),
            zero_shot_spk_id=ZERO_SHOT_SPK_ID,
            stream=STREAM,
            text_frontend=TEXT_FRONTEND,
        )

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
