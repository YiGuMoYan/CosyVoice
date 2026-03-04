import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp


DEFAULT_TEXTS = [
    "你好，今天感觉怎么样？我可以陪你聊聊天。",
    "收到好友从远方寄来的生日礼物，那份意外的惊喜让我很开心。",
    "这段文字稍微长一些，用来测试流式语音合成在长句情况下的首包和总时延表现。",
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


async def run_once(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
    instruct: str,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {"text": text}
    if instruct:
        payload["instruct"] = instruct

    t0 = time.perf_counter()
    first_byte_ms = None
    first_audio_ms = None
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


async def worker(
    name: str,
    queue: asyncio.Queue,
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
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
            r = await run_once(session, url=url, text=text, instruct=instruct, timeout_s=timeout_s)
            r["case_id"] = case_id
            r["worker"] = name
            results.append(r)
        queue.task_done()


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]
    fb = [r["first_byte_ms"] for r in ok]
    fa = [r["first_audio_ms"] for r in ok]
    tt = [r["total_ms"] for r in ok]

    def pack(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }

    return {
        "total_cases": len(results),
        "ok_cases": len(ok),
        "failed_cases": len(fail),
        "first_byte_ms": pack(fb),
        "first_audio_ms": pack(fa),
        "total_ms": pack(tt),
        "errors": fail[:20],
    }


def load_texts(args: argparse.Namespace) -> List[str]:
    if args.text_file:
        path = Path(args.text_file)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [x for x in lines if x]
    if args.text:
        return [args.text]
    return DEFAULT_TEXTS


async def main_async(args: argparse.Namespace) -> Dict[str, Any]:
    texts = load_texts(args)
    queue: asyncio.Queue = asyncio.Queue()
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, args.concurrency))

    case_id = 0
    for _ in range(max(1, args.rounds)):
        for t in texts:
            queue.put_nowait({"case_id": case_id, "text": t})
            case_id += 1

    connector = aiohttp.TCPConnector(limit=max(64, args.concurrency * 4))
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(
                worker(
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark CosyVoice /tts/complete latency")
    p.add_argument("--url", default="http://localhost:9880/tts/complete")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--instruct", default="")
    p.add_argument("--text", default="")
    p.add_argument("--text-file", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(main_async(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
