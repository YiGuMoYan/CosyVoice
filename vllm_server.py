"""
CosyVoice TTS Server (vLLM 加速版，支持 v2/v3)

接口：
  POST /tts/complete — 完整文本模式，服务端手动分段 + 逐段流式合成 + chunk 卡顿检测
  GET  /sample_rate  — 返回当前采样率
"""

import sys
import re
import time
import logging
import struct
import queue
import threading
import os

import numpy as np
import torch._inductor.config  # noqa: F401

sys.path.append("third_party/Matcha-TTS")

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ============================================================
#  配置
# ============================================================
MODEL_VERSION = os.getenv("MODEL_VERSION", "v3")

# v3 配置
MODEL_CONFIG = {
    "model_dir": os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B"),
    "load_trt": os.getenv("COSYVOICE_LOAD_TRT", "False").lower() == "true",
    "fp16": os.getenv("COSYVOICE_FP16", "False").lower() == "true",
    "load_vllm": os.getenv("COSYVOICE_LOAD_VLLM", "True").lower() == "true",
}

PROMPT_TEXT = "再用小部分白米饭和蓝米饭，做出天钿的耳机。唉，天钿，你去哪里，别跑啊。嗯，这次的演唱会是一场观众朋友们期待很久，我也期待很久的演唱会。为了这次久别重逢。"
PROMPT_WAV = os.getenv("COSYVOICE_PROMPT_WAV", "raw/merged_prompt.wav")
# v3 系统前缀（在所有文本前添加）
V3_SYS_PREFIX = "You are a helpful assistant.<|endofprompt|>"
PORT = int(os.getenv("PORT", "9880"))
ENABLE_INSTRUCT = os.getenv("COSYVOICE_ENABLE_INSTRUCT", "True").lower() == "true"
TTS_CONCURRENCY = int(os.getenv("COSYVOICE_TTS_CONCURRENCY", "1"))
MAX_TEXT_CHARS = int(os.getenv("COSYVOICE_MAX_TEXT_CHARS", "1200"))
MAX_SEGMENTS = int(os.getenv("COSYVOICE_MAX_SEGMENTS", "24"))
ENABLE_TEXT_CLEAN = os.getenv("COSYVOICE_ENABLE_TEXT_CLEAN", "True").lower() == "true"
ENABLE_ZH_NUM_NORMALIZE = os.getenv("COSYVOICE_ENABLE_ZH_NUM_NORMALIZE", "True").lower() == "true"
ENABLE_AUTO_BREATH = os.getenv("COSYVOICE_ENABLE_AUTO_BREATH", "True").lower() == "true"

# 单个 chunk 最大等待秒数，超过视为 LLM 跑飞，截断当前段并继续下一段
# 由于每次都要处理音频特征，首字延迟可能较高，增加到 30 秒
CHUNK_STALL_TIMEOUT = int(os.getenv("COSYVOICE_CHUNK_STALL_TIMEOUT", "30"))

# ============================================================
#  日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tts-server")

# ============================================================
#  全局状态（在 __main__ 中初始化）
# ============================================================
cosyvoice = None
sample_rate = 24000
tts_stream_semaphore = threading.Semaphore(max(1, TTS_CONCURRENCY))

# ============================================================
#  FastAPI
# ============================================================
app = FastAPI(title="CosyVoice TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    instruct: Optional[str] = None


# ============================================================
#  工具函数
# ============================================================
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0000FE00-\U0000FE0F"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]"
)


def _clean_text(text: str) -> str:
    """清洗 LLM 输出，去除 markdown / emoji / 舞台指示等对 TTS 有害的符号"""
    text = re.sub(r"\*+", "", text)  # markdown 粗体/斜体
    text = re.sub(r"#+\s*", "", text)  # markdown 标题
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # markdown 链接
    text = re.sub(r"[…。]{2,}", "…", text)  # 连续省略号/句号
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"。{2,}", "。", text)
    text = text.replace("～", "，").replace("~", "，")  # 波浪号 → 逗号
    text = _EMOJI_RE.sub("", text)  # emoji
    text = re.sub(r"[（(][^）)]*[）)]", "", text)  # 括号内舞台指示
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _cap_segments(segments, max_segments: int):
    """限制分段数量，避免超长文本引发极端延迟。"""
    if len(segments) <= max_segments:
        return segments
    head = segments[: max_segments - 1]
    tail = "".join(segments[max_segments - 1 :]).strip()
    if tail:
        head.append(tail)
    return head


# ============================================================
#  中文数字规范化（补充 wetext 不足）
# ============================================================
_DIGIT_MAP = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def _year_to_chinese(m: re.Match) -> str:
    """年份逐字读：2016年 → 二零一六年"""
    return "".join(_DIGIT_MAP[d] for d in m.group(1)) + "年"


def _date_month_day(m: re.Match) -> str:
    """月日：1月 → 一月, 12日 → 十二日"""
    num = int(m.group(1))
    suffix = m.group(2)
    return _int_to_chinese(num) + suffix


def _int_to_chinese(n: int) -> str:
    """整数转中文口语读法（0~99999999）"""
    if n == 0:
        return "零"
    if n < 0:
        return "负" + _int_to_chinese(-n)

    units = [
        (100000000, "亿"),
        (10000, "万"),
        (1000, "千"),
        (100, "百"),
        (10, "十"),
    ]
    parts = []
    for base, unit in units:
        if n >= base:
            q = n // base
            if base >= 10000:
                parts.append(_int_to_chinese(q) + unit)
            else:
                parts.append(_DIGIT_MAP[str(q)] + unit)
            n %= base
            if n > 0 and n < base // 10:
                parts.append("零")
    if n > 0:
        parts.append(_DIGIT_MAP[str(n)])
    result = "".join(parts)
    # 口语习惯：一十 → 十（仅在开头），二百/二千/二万/二亿 → 两百/两千/两万/两亿
    if result.startswith("一十"):
        result = result[1:]
    for u in ("百", "千", "万", "亿"):
        if result.startswith("二" + u):
            result = "两" + result[1:]
            break
    return result


def _general_num_to_chinese(m: re.Match) -> str:
    """通用数字转换（含小数）"""
    s = m.group(0)
    if "." in s:
        integer_part, decimal_part = s.split(".", 1)
        result = _int_to_chinese(int(integer_part)) + "点"
        result += "".join(_DIGIT_MAP[d] for d in decimal_part)
        return result
    n = int(s)
    # 电话号码等长数字串逐字读
    if len(s) >= 7:
        return "".join(_DIGIT_MAP[d] for d in s)
    return _int_to_chinese(n)


def _normalize_chinese_num(text: str) -> str:
    """将文本中的阿拉伯数字转为中文读法"""
    if not re.search(r"\d", text):
        return text
    # 年份：4位数字+年
    text = re.sub(r"(\d{4})年", _year_to_chinese, text)
    # 月/日
    text = re.sub(r"(\d{1,2})(月|日|号)", _date_month_day, text)

    # 百分比
    def _pct_repl(m):
        num_str = m.group(1)
        inner = re.match(r"[\d.]+", num_str)
        return "百分之" + _general_num_to_chinese(inner)

    text = re.sub(r"(\d+(?:\.\d+)?)%", _pct_repl, text)
    # 剩余数字（含小数）
    text = re.sub(r"\d+(?:\.\d+)?", _general_num_to_chinese, text)
    return text


# ============================================================
#  气口插入（在长句逗号处自动添加 [breath]）
# ============================================================
BREATH_MIN_CHARS = 15  # 逗号间隔超过此字数才插入气口


def _insert_breath(text: str) -> str:
    """在较长的逗号分句之间插入 [breath] 标记"""
    if "，" not in text and "," not in text:
        return text
    parts = re.split(r"(，|,)", text)
    result = []
    char_count = 0
    for part in parts:
        if part in ("，", ","):
            if char_count >= BREATH_MIN_CHARS:
                result.append(part + "[breath]")
            else:
                result.append(part)
            char_count = 0
        else:
            char_count += len(part)
            result.append(part)
    return "".join(result)


def _wav_header(sr: int, channels: int = 1, bits: int = 16) -> bytes:
    """生成流式 WAV 头（长度字段填 0xFFFFFFFF）"""
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


def _chunk_to_bytes(chunk) -> tuple[bytes, float]:
    """将模型输出 chunk 转为 int16 PCM bytes，返回 (pcm_bytes, duration_sec)"""
    pcm = (chunk["tts_speech"].numpy() * (2**15)).astype(np.int16).tobytes()
    duration = len(pcm) / (sample_rate * 2)
    return pcm, duration


def _ensure_instruct(instruct: str) -> str:
    """确保 instruct 文本格式正确（v3 需要系统前缀）"""
    if MODEL_VERSION == "v3" and not instruct.startswith("You are a helpful assistant"):
        instruct = "You are a helpful assistant. " + instruct
    if "<|endofprompt|>" not in instruct:
        instruct += "<|endofprompt|>"
    return instruct


def _iter_with_timeout(gen, timeout: float):
    """包装生成器，对每个 chunk 施加超时检测。
    产出 (chunk, False) 表示正常数据，(None, True) 表示超时后结束。"""
    q = queue.Queue()
    sentinel = object()
    exc_box = [None]

    def producer():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            exc_box[0] = e
        finally:
            q.put(sentinel)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            yield None, True
            return
        if item is sentinel:
            if exc_box[0]:
                raise exc_box[0]
            return
        yield item, False


# ============================================================
#  接口：POST /tts/complete — 完整文本分段合成（带 chunk 卡顿检测）
# ============================================================
@app.post("/tts/complete")
async def tts_complete(req: TTSRequest):
    t0 = time.perf_counter()
    raw_text = (req.text or "").strip()
    if not raw_text:
        return StreamingResponse(iter([]), media_type="audio/wav")
    if len(raw_text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"text too long ({len(raw_text)} chars), max={MAX_TEXT_CHARS}",
        )

    text = _clean_text(raw_text) if ENABLE_TEXT_CLEAN else raw_text
    if not text:
        return StreamingResponse(iter([]), media_type="audio/wav")

    is_cjk = _contains_cjk(text)
    # 仅对中文文本进行数字规范化，避免英文语句被过度改写
    if is_cjk and ENABLE_ZH_NUM_NORMALIZE:
        text = _normalize_chinese_num(text)
    # 避免重复插入 breath
    if is_cjk and ENABLE_AUTO_BREATH and "[breath]" not in text:
        text = _insert_breath(text)

    use_instruct = bool(req.instruct) and ENABLE_INSTRUCT and hasattr(cosyvoice, "inference_instruct2")
    instruct_text = _ensure_instruct(req.instruct) if use_instruct else ""
    mode = "instruct2" if use_instruct else "zero_shot"
    logger.info("收到请求(complete) | 模式: %s | 文本: %s", mode, text)

    segments = cosyvoice.frontend.text_normalize(text, split=True)
    segments = _cap_segments(segments, max_segments=max(1, MAX_SEGMENTS))
    logger.info("文本分段完成 | 段数: %d", len(segments))

    def generate():
        with tts_stream_semaphore:
            total_sec = 0.0
            yield _wav_header(sample_rate)

            for i, seg in enumerate(segments):
                seg_t0 = time.perf_counter()
                logger.info("开始第 %d/%d 段: %s", i + 1, len(segments), seg)

                try:
                    if use_instruct:
                        seg_out = cosyvoice.inference_instruct2(
                            seg,
                            instruct_text,
                            PROMPT_WAV,
                            zero_shot_spk_id="custom_speaker",
                            stream=True,
                            text_frontend=False,
                        )
                    else:
                        # 默认 zero_shot 模式 + 预注册 id
                        # 注意：prompt_text 必须与注册时完全一致（包含 prefix）
                        seg_out = cosyvoice.inference_zero_shot(
                            seg,
                            V3_SYS_PREFIX + PROMPT_TEXT,
                            PROMPT_WAV,
                            zero_shot_spk_id="custom_speaker",
                            stream=True,
                            text_frontend=False,
                        )

                    chunks = 0
                    stalled = False
                    for chunk, timed_out in _iter_with_timeout(
                        seg_out, CHUNK_STALL_TIMEOUT
                    ):
                        if timed_out:
                            stalled = True
                            break
                        pcm, dur = _chunk_to_bytes(chunk)
                        total_sec += dur
                        chunks += 1
                        yield pcm

                except Exception as e:
                    logger.error("第 %d 段异常: %s", i + 1, e)
                    continue

                seg_elapsed = time.perf_counter() - seg_t0
                if stalled:
                    logger.warning(
                        "第 %d 段截断 | chunks: %d | 耗时: %.2fs (stall > %ds)",
                        i + 1,
                        chunks,
                        seg_elapsed,
                        CHUNK_STALL_TIMEOUT,
                    )
                else:
                    logger.info(
                        "第 %d 段完成 | chunks: %d | 耗时: %.2fs",
                        i + 1,
                        chunks,
                        seg_elapsed,
                    )

            elapsed = time.perf_counter() - t0
            logger.info(
                "生成完成(complete) | 段数: %d | 音频: %.2fs | 总耗时: %.3fs | RTF: %.3f",
                len(segments),
                total_sec,
                elapsed,
                elapsed / total_sec if total_sec > 0 else 0,
            )

    return StreamingResponse(generate(), media_type="audio/wav")


# ============================================================
#  接口：GET /sample_rate
# ============================================================
@app.get("/sample_rate")
async def get_sample_rate():
    return {"sample_rate": sample_rate}

@app.get("/health")
async def health():
    ready = cosyvoice is not None
    return {
        "ready": ready,
        "model_version": MODEL_VERSION,
        "sample_rate": sample_rate if ready else None,
        "enable_instruct": ENABLE_INSTRUCT,
        "tts_concurrency": TTS_CONCURRENCY,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_segments": MAX_SEGMENTS,
        "enable_text_clean": ENABLE_TEXT_CLEAN,
        "enable_zh_num_normalize": ENABLE_ZH_NUM_NORMALIZE,
        "enable_auto_breath": ENABLE_AUTO_BREATH,
        "prompt_trim_silence": os.getenv("COSYVOICE_PROMPT_TRIM_SILENCE", "False"),
        "prompt_normalize_peak": os.getenv("COSYVOICE_PROMPT_NORMALIZE_PEAK", "False"),
    }


# ============================================================
#  启动
# ============================================================
if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    from cosyvoice.cli.cosyvoice import AutoModel

    cfg = MODEL_CONFIG
    logger.info("正在加载 v3 模型 (%s)...", cfg["model_dir"])

    cosyvoice = AutoModel(
        model_dir=cfg["model_dir"],
        load_trt=cfg["load_trt"],
        load_vllm=cfg["load_vllm"],
        fp16=cfg["fp16"],
    )

    # 提前注册说话人id
    # 注意：v3模型需要包含 <|endofprompt|> 的 prompt_text
    cosyvoice.add_zero_shot_spk(V3_SYS_PREFIX + PROMPT_TEXT, PROMPT_WAV, "custom_speaker")

    logger.info("正在预热模型...")
    # 使用 zero_shot_spk_id
    for _ in cosyvoice.inference_zero_shot(
        "你好",
        V3_SYS_PREFIX + PROMPT_TEXT,
        PROMPT_WAV,
        zero_shot_spk_id="custom_speaker",
        stream=False,
    ):
        pass

    sample_rate = cosyvoice.sample_rate
    logger.info("模型准备就绪 (采样率: %d) | 端口: %d", sample_rate, PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
