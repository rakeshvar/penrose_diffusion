"""Benchmark exact tile-level OT assignment on CPU and CUDA.

This measures the proposed OTFM training path: build a batched N x N cost
matrix, solve one assignment per sample, return permutation indices, and gather
matched noise. Results are written as CSV and JSON metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
import torch
from scipy.optimize import linear_sum_assignment

try:
    import torch_linear_assignment._backend as tla_backend
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    tla_backend = None
    batch_linear_assignment = None


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.split(","):
        batch, tiles = item.lower().split("x")
        cases.append((int(batch), int(tiles)))
    return cases


def parse_workers(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",")})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=parse_cases("64x96,32x192,32x384,32x768"),
        help="Comma-separated BxN cases",
    )
    parser.add_argument("--workers", type=parse_workers, default=parse_workers("1,2,4,8"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=Path("ot_assignment_benchmark.csv"))
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    function: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[object, dict[str, float]]:
    result = None
    for _ in range(warmup):
        result = function()
    synchronize(device)

    samples_ms = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        result = function()
        synchronize(device)
        samples_ms.append((time.perf_counter() - started) * 1_000)

    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return result, {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
    }


def sample_endpoints(
    batch: int, tiles: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    x0_xy = torch.randn((batch, tiles, 2), device=device, generator=generator)
    x0_a = (
        torch.rand((batch, tiles, 1), device=device, generator=generator) * 2 - 1
    ) * math.sqrt(3)
    noise_xy = torch.randn((batch, tiles, 2), device=device, generator=generator)
    noise_a = (
        torch.rand((batch, tiles, 1), device=device, generator=generator) * 2 - 1
    ) * math.sqrt(3)
    return torch.cat((x0_xy, x0_a), dim=-1), torch.cat((noise_xy, noise_a), dim=-1)


def build_cost(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x0, noise).square()


def solve_one(cost: np.ndarray) -> np.ndarray:
    return linear_sum_assignment(cost)[1].astype(np.int64, copy=False)


def solve_scipy_serial(cost: np.ndarray) -> np.ndarray:
    return np.stack([solve_one(matrix) for matrix in cost])


def solve_scipy_threaded(cost: np.ndarray, pool: ThreadPoolExecutor) -> np.ndarray:
    return np.stack(list(pool.map(solve_one, cost)))


def gather_noise(noise: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return noise.gather(1, permutation.unsqueeze(-1).expand(-1, -1, noise.shape[-1]))


def assignment_cost(cost: torch.Tensor, permutation: torch.Tensor) -> float:
    return float(cost.gather(2, permutation.unsqueeze(-1)).sum().item())


def append_result(
    rows: list[dict[str, object]],
    *,
    device: torch.device,
    batch: int,
    tiles: int,
    method: str,
    stage: str,
    timing: dict[str, float],
    correct: bool = True,
) -> None:
    rows.append(
        {
            "device": str(device),
            "batch": batch,
            "tiles": tiles,
            "method": method,
            "stage": stage,
            **timing,
            "correct": correct,
        }
    )


def benchmark_case(
    batch: int,
    tiles: int,
    device: torch.device,
    workers: list[int],
    warmup: int,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    print(f"\n--- B={batch}, N={tiles}, device={device} ---", flush=True)
    generator = torch.Generator(device=device).manual_seed(seed + batch * 10_000 + tiles)
    x0, noise = sample_endpoints(batch, tiles, device, generator)
    rows: list[dict[str, object]] = []

    cost_result, timing = measure(
        lambda: build_cost(x0, noise), device, warmup, repeats
    )
    cost = cost_result
    assert isinstance(cost, torch.Tensor)
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="torch",
        stage="cost_build",
        timing=timing,
    )

    cpu_result, timing = measure(
        lambda: cost.detach().cpu().numpy(), device, warmup, repeats
    )
    cost_cpu = np.asarray(cpu_result)
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="pageable",
        stage="gpu_to_cpu" if device.type == "cuda" else "tensor_to_numpy",
        timing=timing,
    )

    pinned_cost = None
    if device.type == "cuda":
        pinned_cost = torch.empty(cost.shape, dtype=cost.dtype, device="cpu", pin_memory=True)

        def copy_to_pinned() -> np.ndarray:
            assert pinned_cost is not None
            pinned_cost.copy_(cost, non_blocking=True)
            synchronize(device)
            return pinned_cost.numpy()

        pinned_result, timing = measure(copy_to_pinned, device, warmup, repeats)
        append_result(
            rows,
            device=device,
            batch=batch,
            tiles=tiles,
            method="pinned",
            stage="gpu_to_cpu",
            timing=timing,
        )
        cost_cpu = np.asarray(pinned_result).copy()

    reference_result, timing = measure(
        lambda: solve_scipy_serial(cost_cpu), torch.device("cpu"), warmup, repeats
    )
    reference_np = np.asarray(reference_result)
    reference = torch.from_numpy(reference_np).to(device)
    reference_cost = assignment_cost(cost, reference)
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="scipy_serial",
        stage="solve",
        timing=timing,
    )

    _, timing = measure(
        lambda: torch.from_numpy(reference_np).to(device),
        device,
        warmup,
        repeats,
    )
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="scipy_serial",
        stage="indices_to_device",
        timing=timing,
    )

    _, timing = measure(
        lambda: gather_noise(noise, reference), device, warmup, repeats
    )
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="scipy_serial",
        stage="gather",
        timing=timing,
    )

    for worker_count in workers:
        if worker_count == 1:
            continue
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            threaded_result, timing = measure(
                lambda: solve_scipy_threaded(cost_cpu, pool),
                torch.device("cpu"),
                warmup,
                repeats,
            )
        threaded_np = np.asarray(threaded_result)
        threaded = torch.from_numpy(threaded_np).to(device)
        correct = math.isclose(
            assignment_cost(cost, threaded), reference_cost, rel_tol=1e-5, abs_tol=1e-3
        )
        append_result(
            rows,
            device=device,
            batch=batch,
            tiles=tiles,
            method=f"scipy_threads_{worker_count}",
            stage="solve",
            timing=timing,
            correct=correct,
        )

    cuda_backend = bool(
        batch_linear_assignment is not None
        and tla_backend is not None
        and tla_backend.has_cuda()
        and device.type == "cuda"
    )
    if cuda_backend:
        gpu_result, timing = measure(
            lambda: batch_linear_assignment(cost.detach()),
            device,
            warmup,
            repeats,
        )
        gpu_permutation = gpu_result.long()
        correct = math.isclose(
            assignment_cost(cost, gpu_permutation),
            reference_cost,
            rel_tol=1e-5,
            abs_tol=1e-3,
        )
        append_result(
            rows,
            device=device,
            batch=batch,
            tiles=tiles,
            method="torch_lsa_cuda",
            stage="solve",
            timing=timing,
            correct=correct,
        )

        def cuda_end_to_end() -> torch.Tensor:
            current_cost = build_cost(x0, noise)
            permutation = batch_linear_assignment(current_cost.detach()).long()
            return gather_noise(noise, permutation)

        _, timing = measure(cuda_end_to_end, device, warmup, repeats)
        append_result(
            rows,
            device=device,
            batch=batch,
            tiles=tiles,
            method="torch_lsa_cuda",
            stage="end_to_end",
            timing=timing,
            correct=correct,
        )

    def scipy_end_to_end() -> torch.Tensor:
        current_cost = build_cost(x0, noise)
        current_cpu = current_cost.detach().cpu().numpy()
        permutation_np = solve_scipy_serial(current_cpu)
        permutation = torch.from_numpy(permutation_np).to(device)
        return gather_noise(noise, permutation)

    _, timing = measure(scipy_end_to_end, device, warmup, repeats)
    append_result(
        rows,
        device=device,
        batch=batch,
        tiles=tiles,
        method="scipy_serial",
        stage="end_to_end",
        timing=timing,
    )

    random_permutation = torch.arange(tiles, device=device).expand(batch, -1)
    random_cost = assignment_cost(cost, random_permutation)
    print(
        f"reference_cost={reference_cost:.3f}, "
        f"unmatched_cost={random_cost:.3f}, "
        f"CUDA_backend={cuda_backend}",
        flush=True,
    )
    for row in rows:
        if row["stage"] in {"solve", "end_to_end"}:
            print(
                f"{row['method']:>18} {row['stage']:>10}: "
                f"{row['mean_ms']:10.3f} ms "
                f"(p95 {row['p95_ms']:10.3f}) correct={row['correct']}",
                flush=True,
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    metadata = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_linear_assignment_installed": batch_linear_assignment is not None,
        "torch_linear_assignment_cuda": bool(
            tla_backend is not None and tla_backend.has_cuda()
        ),
        "scipy": scipy.__version__,
        "cpu_count": os.cpu_count(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "cases": args.cases,
        "workers": args.workers,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    print(json.dumps(metadata, indent=2), flush=True)

    all_rows = []
    for batch, tiles in args.cases:
        all_rows.extend(
            benchmark_case(
                batch,
                tiles,
                device,
                args.workers,
                args.warmup,
                args.repeats,
                args.seed,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nWrote {args.output} and {args.output.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
