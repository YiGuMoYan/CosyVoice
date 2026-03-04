import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except Exception:
    aiohttp = None


DEFAULT_TEXTS = [
    "Hello, this is a short test sentence for TTS benchmarking.",
    "This sentence is a bit longer and is used to observe first chunk latency and total synthesis time.",
    "The benchmark repeats multiple rounds so you can compare tuning profiles with stable metrics.",
]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def pack_metric(ok_results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = [float(r[key]) for r in ok_results if key in r]
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    return {
        "total_cases": len(results),
        "ok_cases": len(ok),
        "failed_cases": len(fail),
        "first_byte_ms": pack_metric(ok, "first_byte_ms"),
        "first_audio_ms": pack_metric(ok, "first_audio_ms"),
        "first_chunk_ms": pack_metric(ok, "first_chunk_ms"),
        "total_ms": pack_metric(ok, "total_ms"),
        "audio_sec": pack_metric(ok, "audio_sec"),
        "rtf": pack_metric(ok, "rtf"),
        "errors": fail[:20],
    }


def load_texts(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        path = Path(args.text_file)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [x for x in lines if x]
        if texts:
            return texts
    if args.text:
        return [args.text]
    return DEFAULT_TEXTS


async def run_http_once(
    session: "aiohttp.ClientSession",
    url: str,
    text: str,
    instruct: str,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {"text": text}
    if instruct:
        payload["instruct"] = instruct

    t0 = time.perf_counter()
    first_byte_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    total_bytes = 0
    header_seen = 0
    wav_header_size = 44

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "ok": False,
                    "status": resp.status,
                    "error": body[:200],
                    "text_len": len(text),
                }

            async for chunk in resp.content.iter_chunked(4096):
                if not chunk:
                    continue
                now_ms = (time.perf_counter() - t0) * 1000
                if first_byte_ms is None:
                    first_byte_ms = now_ms

                total_bytes += len(chunk)
                if first_audio_ms is None:
                    if header_seen < wav_header_size:
                        need = wav_header_size - header_seen
                        used = min(need, len(chunk))
                        header_seen += used
                        if len(chunk) > used:
                            first_audio_ms = now_ms
                    else:
                        first_audio_ms = now_ms

    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e), "text_len": len(text)}

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "ok": True,
        "status": 200,
        "text_len": len(text),
        "first_byte_ms": first_byte_ms or 0.0,
        "first_audio_ms": first_audio_ms or (first_byte_ms or 0.0),
        "total_ms": total_ms,
        "total_bytes": total_bytes,
    }


async def http_worker(
    name: str,
    queue: asyncio.Queue,
    sem: asyncio.Semaphore,
    session: "aiohttp.ClientSession",
    url: str,
    instruct: str,
    timeout_s: int,
    results: List[Dict[str, Any]],
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        text = item["text"]
        case_id = item["case_id"]
        async with sem:
            result = await run_http_once(session, url=url, text=text, instruct=instruct, timeout_s=timeout_s)
            result["case_id"] = case_id
            result["worker"] = name
            results.append(result)
        queue.task_done()


async def run_http_benchmark(args: argparse.Namespace, texts: List[str]) -> Dict[str, Any]:
    if aiohttp is None:
        raise RuntimeError("aiohttp is not installed. Install it or run with --mode local.")

    queue: asyncio.Queue = asyncio.Queue()
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, args.concurrency))

    case_id = 0
    for _ in range(max(1, args.rounds)):
        for text in texts:
            queue.put_nowait({"case_id": case_id, "text": text})
            case_id += 1

    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 4))
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(
                http_worker(
                    name=f"w{i}",
                    queue=queue,
                    sem=sem,
                    session=session,
                    url=args.url,
                    instruct=args.instruct,
                    timeout_s=args.timeout,
                    results=results,
                )
            )
            for i in range(max(1, args.concurrency))
        ]
        for _ in workers:
            queue.put_nowait(None)
        await queue.join()
        await asyncio.gather(*workers, return_exceptions=True)

    return summarize(results)


def prepare_local_imports(register_vllm: bool):
    project_root = Path(__file__).resolve().parent.parent
    matcha_dir = project_root / "third_party" / "Matcha-TTS"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if matcha_dir.exists() and str(matcha_dir) not in sys.path:
        sys.path.insert(0, str(matcha_dir))

    if register_vllm:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM

        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel


def run_local_once(model: Any, args: argparse.Namespace, text: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    first_chunk_ms: Optional[float] = None
    audio_sec = 0.0
    chunk_count = 0

    try:
        if args.instruct:
            iterator = model.inference_instruct2(
                text,
                args.instruct,
                args.prompt_wav,
                zero_shot_spk_id=args.zero_shot_spk_id,
                stream=args.stream,
                speed=args.speed,
                text_frontend=not args.disable_text_frontend,
            )
        else:
            iterator = model.inference_zero_shot(
                text,
                args.prompt_text,
                args.prompt_wav,
                zero_shot_spk_id=args.zero_shot_spk_id,
                stream=args.stream,
                speed=args.speed,
                text_frontend=not args.disable_text_frontend,
            )

        for chunk in iterator:
            speech = chunk.get("tts_speech")
            chunk_count += 1
            now_ms = (time.perf_counter() - t0) * 1000
            if first_chunk_ms is None:
                first_chunk_ms = now_ms
            if speech is not None and hasattr(speech, "shape") and len(speech.shape) >= 2:
                audio_sec += float(speech.shape[1]) / float(model.sample_rate)

    except Exception as e:
        return {
            "ok": False,
            "status": 0,
            "error": str(e),
            "text_len": len(text),
        }

    total_ms = (time.perf_counter() - t0) * 1000
    rtf = (total_ms / 1000.0 / audio_sec) if audio_sec > 0 else 0.0
    return {
        "ok": True,
        "status": 200,
        "text_len": len(text),
        "first_chunk_ms": first_chunk_ms or 0.0,
        "total_ms": total_ms,
        "audio_sec": audio_sec,
        "rtf": rtf,
        "chunk_count": chunk_count,
    }


def run_local_benchmark(args: argparse.Namespace, texts: List[str]) -> Dict[str, Any]:
    if args.concurrency > 1:
        print("warning: local mode currently runs sequentially; --concurrency is ignored.")

    AutoModel = prepare_local_imports(register_vllm=args.register_vllm)
    model = AutoModel(
        model_dir=args.model_dir,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )

    results: List[Dict[str, Any]] = []
    case_id = 0

    for _ in range(max(0, args.warmup)):
        for text in texts:
            _ = run_local_once(model, args, text)

    for _ in range(max(1, args.rounds)):
        for text in texts:
            result = run_local_once(model, args, text)
            result["case_id"] = case_id
            case_id += 1
            results.append(result)

    return summarize(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CosyVoice in HTTP or local mode")
    parser.add_argument("--mode", choices=["http", "local"], default="local")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--text", default="")
    parser.add_argument("--text-file", default="")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--output", default="")

    parser.add_argument("--url", default="http://localhost:9880/tts/complete")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)

    parser.add_argument("--model-dir", default="pretrained_models/Fun-CosyVoice3-0.5B")
    parser.add_argument("--prompt-wav", default="./asset/zero_shot_prompt.wav")
    parser.add_argument(
        "--prompt-text",
        default="You are a helpful assistant.<|endofprompt|>This is a prompt for cloning voice tone.",
    )
    parser.add_argument("--zero-shot-spk-id", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--disable-text-frontend", action="store_true")
    parser.add_argument("--load-vllm", action="store_true")
    parser.add_argument("--register-vllm", action="store_true")
    parser.add_argument("--load-trt", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts(args)

    if args.mode == "http":
        summary = asyncio.run(run_http_benchmark(args, texts))
    else:
        summary = run_local_benchmark(args, texts)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
