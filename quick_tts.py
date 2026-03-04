import sys
from pathlib import Path

import torch
import torch._inductor.config  # noqa: F401
import torchaudio


# -------------------- Quick Config (no CLI args) --------------------
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = "raw/merged_prompt.wav"
OUTPUT_DIR = "output"

# Keep user-provided texts unchanged.
ZERO_SHOT_TEXT = "你好，我是洛天依，很高兴认识大家。"
PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "再用小部分白米饭和蓝米饭，做出天钿的耳机。唉，天钿，你去哪里，别跑啊。"
    "嗯，这次的演唱会是一场观众朋友们期待很久，我也期待很久的演唱会。为了这次久别重逢。"
)
INSTRUCT2_TEXT = "你好，我是洛天依，很高兴认识大家。"
INSTRUCT_TEXT = "You are a helpful assistant.<|endofprompt|>"

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


def collect_wav_from_iterator(iterator) -> torch.Tensor:
    chunks = []
    for item in iterator:
        speech = item.get("tts_speech")
        if speech is not None and speech.numel() > 0:
            chunks.append(speech)
    if not chunks:
        raise RuntimeError("No speech generated.")
    wav = torch.cat(chunks, dim=1)
    if wav.dim() != 2:
        raise RuntimeError(f"Unexpected waveform shape: {tuple(wav.shape)}")
    return wav


def main() -> None:
    project_root = Path(__file__).resolve().parent
    prepare_import_path(project_root)

    if LOAD_VLLM:
        prepare_vllm_registry()

    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = project_root / MODEL_DIR
    prompt_wav = str(project_root / PROMPT_WAV)
    out_dir = project_root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cosyvoice = AutoModel(
        model_dir=str(model_dir),
        load_vllm=LOAD_VLLM,
        load_trt=LOAD_TRT,
        fp16=FP16,
    )

    zero_iter = cosyvoice.inference_zero_shot(
        ZERO_SHOT_TEXT,
        PROMPT_TEXT,
        prompt_wav,
        zero_shot_spk_id=ZERO_SHOT_SPK_ID,
        stream=STREAM,
        speed=SPEED,
        text_frontend=TEXT_FRONTEND,
    )
    zero_wav = collect_wav_from_iterator(zero_iter)
    zero_path = out_dir / "quick_zero_shot.wav"
    torchaudio.save(str(zero_path), zero_wav, cosyvoice.sample_rate)
    print(f"done: {zero_path}")

    instruct_iter = cosyvoice.inference_instruct2(
        INSTRUCT2_TEXT,
        INSTRUCT_TEXT,
        prompt_wav,
        zero_shot_spk_id=ZERO_SHOT_SPK_ID,
        stream=STREAM,
        speed=SPEED,
        text_frontend=TEXT_FRONTEND,
    )
    instruct_wav = collect_wav_from_iterator(instruct_iter)
    instruct_path = out_dir / "quick_instruct2.wav"
    torchaudio.save(str(instruct_path), instruct_wav, cosyvoice.sample_rate)
    print(f"done: {instruct_path}")


if __name__ == "__main__":
    main()
