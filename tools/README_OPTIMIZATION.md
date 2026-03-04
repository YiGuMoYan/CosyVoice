# Optimization Toolkit

## 1) Apply a profile

PowerShell:

```powershell
Get-Content .\tools\profiles\clone_max.env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k, $v = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
}
python .\vllm_server.py
```

## 2) Benchmark latency

```powershell
python .\tools\benchmark_tts.py --url http://localhost:9880/tts/complete --concurrency 2 --rounds 5
```

Use your own testset:

```powershell
python .\tools\benchmark_tts.py --text-file .\your_texts.txt --concurrency 1 --rounds 3 --output .\bench.json
```

## 3) Suggested workflow

1. Start with `clone_max.env` for similarity-first.
2. Run `benchmark_tts.py` and record `first_audio_ms` + `total_ms`.
3. Tune these keys in order:
   - `COSYVOICE_TOKEN_HOP_LEN`
   - `COSYVOICE_STREAM_SCALE_FACTOR`
   - `COSYVOICE_VLLM_TOP_P`
   - `COSYVOICE_VLLM_REPETITION_PENALTY`
4. Keep prompt audio between 6s and 20s, clean and stable.
