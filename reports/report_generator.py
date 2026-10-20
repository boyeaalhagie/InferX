#!/usr/bin/env python3
"""
InferX performance report generator.

Reads JSONL result files from results/, aggregates metrics,
flags regressions vs baseline, and produces formatted reports.

Usage:
    python3 reports/report_generator.py --results-dir results/
    python3 reports/report_generator.py --compare results/A_pytorch_bf16_in512_bs1.jsonl
    python3 reports/report_generator.py --baseline results/baselines/ --compare results/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


REGRESSION_THRESHOLDS = {
    "ttft_ms": 0.05,                 # +5% TTFT = latency regression
    "output_tokens_per_sec": -0.05,  # -5% throughput = regression
    "gpu_memory_peak_gb": 0.20,      # +20% memory = significant increase
}


@dataclass
class RunRecord:
    """One benchmark result record."""
    run_id: str
    model: str
    engine: str
    gpu: str
    dtype: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    requests: int
    ttft_ms: float
    itl_ms: float
    latency_ms: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    requests_per_sec: float
    gpu_memory_peak_gb: float
    gpu_utilization_pct: float
    timestamp: str
    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(
            run_id=d.get("run_id", "?"),
            model=d.get("model", "?"),
            engine=d.get("engine", "?"),
            gpu=d.get("gpu", "?"),
            dtype=d.get("dtype", "?"),
            batch_size=d.get("batch_size", 0),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            requests=d.get("requests", 0),
            ttft_ms=d.get("ttft_ms", 0.0),
            itl_ms=d.get("itl_ms", 0.0),
            latency_ms=d.get("latency_ms", 0.0),
            output_tokens_per_sec=d.get("output_tokens_per_sec", 0.0),
            total_tokens_per_sec=d.get("total_tokens_per_sec", 0.0),
            requests_per_sec=d.get("requests_per_sec", 0.0),
            gpu_memory_peak_gb=d.get("gpu_memory_peak_gb", 0.0),
            gpu_utilization_pct=d.get("gpu_utilization_pct", 0.0),
            timestamp=d.get("timestamp", "?"),
            raw=d,
        )


def load_jsonl(path: Path) -> list[RunRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(RunRecord.from_dict(json.loads(line)))
                except Exception as e:
                    print(f"Warning: skipping malformed line in {path}: {e}")
    return records


def load_dir(directory: Path) -> list[RunRecord]:
    records = []
    for f in sorted(directory.glob("*.jsonl")):
        records.extend(load_jsonl(f))
    return records


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100.0


def flag_regression(metric: str, new_val: float, old_val: float) -> Optional[str]:
    threshold = REGRESSION_THRESHOLDS.get(metric)
    if threshold is None:
        return None
    change = pct_change(new_val, old_val)
    if threshold > 0 and change > threshold * 100:
        return f"REGRESSION: {metric} increased {change:+.1f}%"
    if threshold < 0 and change < threshold * 100:
        return f"REGRESSION: {metric} dropped {change:+.1f}%"
    return None


def print_full_report(record: RunRecord, baseline: Optional[RunRecord] = None):
    sep = "=" * 50
    thin = "-" * 50
    print(sep)
    print("LLM INFERENCE PERFORMANCE REPORT")
    print(sep)
    print()
    print(f"Model:       {Path(record.model).name}")
    print(f"GPU:         {record.gpu}")
    print(f"Engine:      {record.engine}")
    print(f"Precision:   {record.dtype}")
    print(f"Input:       {record.input_tokens} tokens")
    print(f"Output:      {record.output_tokens} tokens")
    print(f"Concurrency: {record.batch_size}")
    print(f"Timestamp:   {record.timestamp}")
    print()
    print(thin)
    print("PERFORMANCE")
    print(thin)

    def delta(metric: str, fmt: str = ".1f") -> str:
        if baseline is None:
            return ""
        old = getattr(baseline, metric)
        new = getattr(record, metric)
        ch = pct_change(new, old)
        sign = "+" if ch >= 0 else ""
        return f"  ({sign}{ch:.1f}% vs baseline)"

    print(f"TTFT:             {record.ttft_ms:.1f} ms{delta('ttft_ms')}")
    print(f"ITL:              {record.itl_ms:.2f} ms{delta('itl_ms')}")
    print(f"E2E Latency:      {record.latency_ms:.1f} ms{delta('latency_ms')}")
    print(f"Output Tok/s:     {record.output_tokens_per_sec:.1f}{delta('output_tokens_per_sec')}")
    print(f"Total Tok/s:      {record.total_tokens_per_sec:.1f}{delta('total_tokens_per_sec')}")
    print(f"Requests/s:       {record.requests_per_sec:.2f}{delta('requests_per_sec')}")
    print(f"GPU Memory Peak:  {record.gpu_memory_peak_gb:.2f} GB{delta('gpu_memory_peak_gb')}")
    print(f"GPU Utilization:  {record.gpu_utilization_pct:.1f}%{delta('gpu_utilization_pct')}")
    print()

    if baseline is not None:
        regressions = []
        for metric in REGRESSION_THRESHOLDS:
            msg = flag_regression(metric, getattr(record, metric), getattr(baseline, metric))
            if msg:
                regressions.append(msg)

        if regressions:
            print(thin)
            print("REGRESSIONS DETECTED")
            print(thin)
            for r in regressions:
                print(f"  !! {r}")
            print()

    print(sep)


def print_comparison_table(records: list[RunRecord]):
    """Print a side-by-side comparison table (all records with same input/output config)."""
    if not records:
        return

    print(f"\n{'Engine':<16} {'dtype':<6} {'BS':<4} {'in':<5} {'out':<5} "
          f"{'TTFT ms':<10} {'ITL ms':<9} {'tok/s':<10} {'mem GB':<8} {'util%':<6}")
    print("-" * 80)
    for r in records:
        print(f"{r.engine:<16} {r.dtype:<6} {r.batch_size:<4} {r.input_tokens:<5} "
              f"{r.output_tokens:<5} {r.ttft_ms:<10.1f} {r.itl_ms:<9.2f} "
              f"{r.output_tokens_per_sec:<10.1f} {r.gpu_memory_peak_gb:<8.2f} "
              f"{r.gpu_utilization_pct:<6.1f}")


def main():
    p = argparse.ArgumentParser(description="InferX report generator")
    p.add_argument("--results-dir", default="results/",
                   help="Directory containing JSONL result files")
    p.add_argument("--baseline-dir", default="results/baselines/",
                   help="Directory containing baseline JSONL files")
    p.add_argument("--compare", nargs="+",
                   help="Specific JSONL files to compare")
    p.add_argument("--table", action="store_true",
                   help="Print comparison table instead of full report")
    args = p.parse_args()

    if args.compare:
        records = []
        for path in args.compare:
            records.extend(load_jsonl(Path(path)))
    else:
        records = load_dir(Path(args.results_dir))

    if not records:
        print("No results found.")
        sys.exit(1)

    # Load baselines for regression comparison
    baseline_map: dict[str, RunRecord] = {}
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        baselines = load_dir(baseline_dir)
        for b in baselines:
            key = f"{b.engine}_{b.dtype}_{b.input_tokens}_{b.batch_size}"
            baseline_map[key] = b

    if args.table:
        print_comparison_table(records)
    else:
        for record in records:
            key = f"{record.engine}_{record.dtype}_{record.input_tokens}_{record.batch_size}"
            baseline = baseline_map.get(key)
            print_full_report(record, baseline)


if __name__ == "__main__":
    main()
