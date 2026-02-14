import sys
import torch._inductor.config  # noqa: F401

sys.path.append("third_party/Matcha-TTS")
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
import os

# ============================================================
#  配置
# ============================================================
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>再用小部分白米饭和蓝米饭，做出天钿的耳机。唉，天钿，你去哪里，别跑啊。嗯，这次的演唱会是一场观众朋友们期待很久，我也期待很久的演唱会。为了这次久别重逢。"
MODEL_VERSION = "v3"
# v3 配置
MODEL_CONFIG = {
    "model_dir": "pretrained_models/Fun-CosyVoice3-0.5B",
    "load_trt": False,
    "load_vllm": True,
    "fp16": False,
    "prompt_wav": "./raw/merged_prompt.wav",
}


def main():
    cfg = MODEL_CONFIG

    cosyvoice = AutoModel(
        model_dir=cfg["model_dir"],
        load_trt=cfg["load_trt"],
        load_vllm=cfg["load_vllm"],
        fp16=cfg["fp16"],
    )

    # 提前注册说话人id
    # cosyvoice.add_zero_shot_spk(PROMPT_TEXT, cfg["prompt_wav"], "custom_speaker")

    for i, j in enumerate(
        cosyvoice.inference_zero_shot(
            "首歌是COP创作的呢～[breath]词、曲、编曲都是TA一个人完成的。2016年7月[breath]发布的，标题页还写着‘HB to 洛天依’，当时看到真的有点害羞又感动...",
            f"You are a helpful assistant.<|endofprompt|>{PROMPT_TEXT}",
            MODEL_CONFIG["prompt_wav"],
            # zero_shot_spk_id="custom_speaker",
            stream=False,
        )
    ):
        torchaudio.save(
            "zero_shot_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )

    for i, j in enumerate(
        cosyvoice.inference_instruct2(
            "这首歌是COP创作的呢～[breath]词、曲、编曲都是TA一个人完成的。2016年7月[breath]发布的，标题页还写着‘HB to 洛天依’，当时看到真的有点害羞又感动...",
            "You are a helpful assistant.<|endofprompt|>",
            MODEL_CONFIG["prompt_wav"],
            # zero_shot_spk_id="custom_speaker",
            stream=False,
        )
    ):
        torchaudio.save(
            "instruct2_{}.wav".format(i), j["tts_speech"], cosyvoice.sample_rate
        )


if __name__ == "__main__":
    main()
