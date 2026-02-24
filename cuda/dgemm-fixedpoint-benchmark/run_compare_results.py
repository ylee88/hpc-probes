#!/usr/bin/env python3
"""Run benchmark in two env modes and print a compact comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass

MODE_A = "Mode A (env unset)"
MODE_B = "Mode B (env enabled)"


@dataclass
class RunMetrics:
    avg_ms: float
    tflops: float
    checksum: float
    retained_mantissa_bits: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dgemm-fixedpoint-benchmark with and without env vars and compare outputs."
    )
    parser.add_argument(
        "--bin",
        default=str(Path(__file__).resolve().with_name("dgemm-fixedpoint-benchmark.x")),
        help="Path to benchmark binary (default: ./dgemm-fixedpoint-benchmark.x next to this script).",
    )
    parser.add_argument(
        "--strategy",
        default="eager",
        help="Value for CUBLAS_EMULATION_STRATEGY in Mode B (default: eager).",
    )
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the benchmark binary after '--'.",
    )
    args = parser.parse_args()
    if args.bench_args and args.bench_args[0] == "--":
        args.bench_args = args.bench_args[1:]
    return args


def mode_header(mode: str) -> str:
    if mode == MODE_A:
        return "== Mode A: env vars unset (baseline) =="
    return "== Mode B: fixed-point emulation enabled via env vars =="


def run_mode(mode: str, bin_path: str, bench_args: list[str], strategy: str) -> str:
    env = os.environ.copy()
    if mode == MODE_A:
        env.pop("CUBLAS_EMULATE_DOUBLE_PRECISION", None)
        env.pop("CUBLAS_EMULATION_STRATEGY", None)
    else:
        env["CUBLAS_EMULATE_DOUBLE_PRECISION"] = "1"
        env["CUBLAS_EMULATION_STRATEGY"] = strategy

    result = subprocess.run(
        [bin_path, *bench_args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    section = [mode_header(mode)]
    if result.stdout:
        section.append(result.stdout.rstrip())
    if result.stderr:
        section.append(result.stderr.rstrip())

    if result.returncode != 0:
        joined = "\n".join(section)
        raise RuntimeError(
            f"Benchmark failed in {mode} with exit code {result.returncode}.\n{joined}"
        )

    return "\n".join(section).strip() + "\n"


def parse_run_line(line: str) -> RunMetrics | None:
    if "[run]" not in line:
        return None

    pat = (
        r"avg_ms=(?P<avg_ms>[-+0-9.eE]+)\s+"
        r"tflops=(?P<tflops>[-+0-9.eE]+)\s+"
        r"checksum=(?P<checksum>[-+0-9.eE]+)\s+"
        r"retained_mantissa_bits=(?P<bits>-?\d+)"
    )
    m = re.search(pat, line)
    if not m:
        return None

    return RunMetrics(
        avg_ms=float(m.group("avg_ms")),
        tflops=float(m.group("tflops")),
        checksum=float(m.group("checksum")),
        retained_mantissa_bits=int(m.group("bits")),
    )


def parse_metrics(lines: list[str]) -> tuple[RunMetrics, RunMetrics]:
    current_mode: str | None = None
    parsed: dict[str, RunMetrics] = {}

    for raw in lines:
        line = raw.strip()
        if line.startswith("== Mode A:"):
            current_mode = MODE_A
            continue
        if line.startswith("== Mode B:"):
            current_mode = MODE_B
            continue

        metrics = parse_run_line(line)
        if metrics is not None and current_mode is not None:
            parsed[current_mode] = metrics

    if MODE_A not in parsed or MODE_B not in parsed:
        raise RuntimeError("Could not parse [run] metrics for both modes.")

    return parsed[MODE_A], parsed[MODE_B]


def ratio_change(new: float, base: float) -> float:
    if base == 0.0:
        return float("nan")
    return new / base


def fmt_float(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


def fmt_ratio(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x:.3f}x"


def print_summary(a: RunMetrics, b: RunMetrics) -> None:
    speedup = (a.avg_ms / b.avg_ms) if b.avg_ms > 0.0 else float("nan")
    checksum_delta = b.checksum - a.checksum

    print("Comparison Summary")
    print("------------------")
    print(f"{'Metric':<24} {'Mode A':>16} {'Mode B':>16} {'B vs A':>12}")
    print(
        f"{'avg_ms (lower better)':<24}"
        f" {fmt_float(a.avg_ms):>16} {fmt_float(b.avg_ms):>16} {fmt_ratio(ratio_change(b.avg_ms, a.avg_ms)):>12}"
    )
    print(
        f"{'tflops (higher better)':<24}"
        f" {fmt_float(a.tflops):>16} {fmt_float(b.tflops):>16} {fmt_ratio(ratio_change(b.tflops, a.tflops)):>12}"
    )
    print(
        f"{'checksum':<24}"
        f" {fmt_float(a.checksum):>16} {fmt_float(b.checksum):>16} {fmt_float(checksum_delta):>12}"
    )
    print(
        f"{'retained_bits':<24}"
        f" {a.retained_mantissa_bits:>16} {b.retained_mantissa_bits:>16} {'n/a':>12}"
    )
    print()
    print(f"Speedup (Mode A / Mode B): {fmt_float(speedup)}x")
    print(
        "Mode B emulation status: "
        + ("engaged" if b.retained_mantissa_bits >= 0 else "not engaged (fallback)")
    )


def main() -> int:
    try:
        args = parse_args()
        mode_a = run_mode(MODE_A, args.bin, args.bench_args, args.strategy)
        mode_b = run_mode(MODE_B, args.bin, args.bench_args, args.strategy)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    text = f"{mode_a}\n{mode_b}"
    print(text, end="")

    try:
        a, b = parse_metrics(text.splitlines())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print()
    print_summary(a, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
