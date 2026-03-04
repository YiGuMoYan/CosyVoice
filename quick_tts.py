import sys
import os
import shutil
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
INSTRUCT_TEXT = "请保持与提示音完全一致的音色与语气，自然朗读，不要读出系统提示。<|endofprompt|>"

ZERO_SHOT_SPK_ID = ""
STREAM = False
TEXT_FRONTEND = True
SPEED = 1.0

LOAD_VLLM = True
LOAD_TRT = False
FP16 = False
INSTRUCT2_KEEP_LLM_PROMPT_SPEECH = True
REFINE_INSTRUCT2_WITH_VC = True
VC_REFINE_PASSES = 2
INSTRUCT2_ANCHOR_MODE = "prompt_only"
MIN_AUDIO_SEC = 1.0
MIN_AUDIO_RMS = 0.003
MIN_AUDIO_PEAK = 0.02
INSTRUCT2_CANDIDATES = [
    {"name": "c1", "keep": True, "anchor": "prompt_only", "vc_passes": 1},
    {"name": "c2", "keep": True, "anchor": "prompt_then_instruct", "vc_passes": 1},
    {"name": "c3", "keep": False, "anchor": "instruct_only", "vc_passes": 1},
    {"name": "c4", "keep": False, "anchor": "instruct_only", "vc_passes": 0},
]
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


def audio_stats(wav: torch.Tensor, sample_rate: int):
    duration = float(wav.shape[1]) / float(sample_rate)
    wav_f = wav.float()
    rms = float(torch.sqrt(torch.mean(wav_f * wav_f)).item())
    peak = float(torch.max(torch.abs(wav_f)).item())
    return duration, rms, peak


def is_bad_audio(wav: torch.Tensor, sample_rate: int) -> bool:
    duration, rms, peak = audio_stats(wav, sample_rate)
    return duration < MIN_AUDIO_SEC or rms < MIN_AUDIO_RMS or peak < MIN_AUDIO_PEAK


def quality_score(wav: torch.Tensor, sample_rate: int) -> float:
    duration, rms, peak = audio_stats(wav, sample_rate)
    return duration * (rms + 1e-4) * (peak + 1e-4)


def run_instruct2_candidate(cosyvoice, prompt_wav: str, out_dir: Path, cfg: dict):
    os.environ["COSYVOICE_INSTRUCT2_KEEP_LLM_PROMPT_SPEECH"] = "True" if cfg["keep"] else "False"
    os.environ["COSYVOICE_INSTRUCT2_ANCHOR_MODE"] = cfg["anchor"]
    os.environ["COSYVOICE_INSTRUCT2_STRIP_SYSTEM_PREFIX"] = "True"

    instruct_iter = cosyvoice.inference_instruct2(
        INSTRUCT2_TEXT,
        INSTRUCT_TEXT,
        prompt_wav,
        zero_shot_spk_id=ZERO_SHOT_SPK_ID,
        stream=STREAM,
        speed=SPEED,
        text_frontend=TEXT_FRONTEND,
        prompt_text=PROMPT_TEXT,
    )
    raw_wav = collect_wav_from_iterator(instruct_iter)
    raw_path = out_dir / f"quick_instruct2_raw_{cfg['name']}.wav"
    torchaudio.save(str(raw_path), raw_wav, cosyvoice.sample_rate)
    print(f"done: {raw_path}")

    final_wav = raw_wav
    final_path = raw_path
    if REFINE_INSTRUCT2_WITH_VC and cfg["vc_passes"] > 0:
        vc_source_path = raw_path
        for i in range(cfg["vc_passes"]):
            vc_iter = cosyvoice.inference_vc(
                str(vc_source_path),
                prompt_wav,
                stream=STREAM,
                speed=SPEED,
            )
            final_wav = collect_wav_from_iterator(vc_iter)
            pass_path = out_dir / f"quick_instruct2_{cfg['name']}_vc_pass{i + 1}.wav"
            torchaudio.save(str(pass_path), final_wav, cosyvoice.sample_rate)
            print(f"done: {pass_path}")
            vc_source_path = pass_path
            final_path = pass_path
    return final_wav, final_path, raw_path


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

    candidates = []
    for cfg in INSTRUCT2_CANDIDATES:
        try:
            wav, final_path, raw_path = run_instruct2_candidate(cosyvoice, prompt_wav, out_dir, cfg)
            bad = is_bad_audio(wav, cosyvoice.sample_rate)
            score = quality_score(wav, cosyvoice.sample_rate)
            duration, rms, peak = audio_stats(wav, cosyvoice.sample_rate)
            print(
                f"candidate={cfg['name']} bad={bad} score={score:.6f} "
                f"dur={duration:.3f}s rms={rms:.6f} peak={peak:.6f}"
            )
            candidates.append(
                {
                    "cfg": cfg,
                    "wav": wav,
                    "final_path": final_path,
                    "raw_path": raw_path,
                    "bad": bad,
                    "score": score,
                }
            )
        except Exception as e:
            print(f"candidate={cfg['name']} failed: {e}")

    if not candidates:
        raise RuntimeError("All instruct2 candidates failed.")

    good = [x for x in candidates if not x["bad"]]
    picked = max(good if good else candidates, key=lambda x: x["score"])

    instruct_raw_path = out_dir / "quick_instruct2_raw.wav"
    shutil.copyfile(str(picked["raw_path"]), str(instruct_raw_path))
    print(f"done: {instruct_raw_path}")

    instruct_path = out_dir / "quick_instruct2.wav"
    shutil.copyfile(str(picked["final_path"]), str(instruct_path))
    print(f"done: {instruct_path}")


if __name__ == "__main__":
    main()
