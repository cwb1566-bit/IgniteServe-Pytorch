#!/usr/bin/env python3
"""Smoke-test process-isolated full-FX compilation on CUDA."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import as_completed

import torch
from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def make_job(variant: int, size: int):
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    if variant % 4 == 0:
        temporary = graph.call_function(torch.ops.aten.sin.default, (value,))
        temporary = graph.call_function(
            torch.ops.aten.mul.Scalar, (temporary, 1.25)
        )
        temporary = graph.call_function(
            torch.ops.aten.add.Tensor, (temporary, value)
        )
        output = graph.call_function(torch.ops.aten.relu.default, (temporary,))
    elif variant % 4 == 1:
        temporary = graph.call_function(torch.ops.aten.cos.default, (value,))
        temporary = graph.call_function(
            torch.ops.aten.mul.Scalar, (temporary, 0.75)
        )
        temporary = graph.call_function(
            torch.ops.aten.sub.Tensor, (temporary, value)
        )
        output = graph.call_function(torch.ops.aten.sigmoid.default, (temporary,))
    elif variant % 4 == 2:
        temporary = graph.call_function(
            torch.ops.aten.mul.Scalar, (value, 0.01)
        )
        temporary = graph.call_function(torch.ops.aten.exp.default, (temporary,))
        temporary = graph.call_function(
            torch.ops.aten.add.Tensor, (temporary, value)
        )
        output = graph.call_function(torch.ops.aten.tanh.default, (temporary,))
    else:
        temporary = graph.call_function(
            torch.ops.aten.mul.Tensor, (value, value)
        )
        sin_value = graph.call_function(torch.ops.aten.sin.default, (value,))
        temporary = graph.call_function(
            torch.ops.aten.sub.Tensor, (temporary, sin_value)
        )
        output = graph.call_function(torch.ops.aten.gelu.default, (temporary,))
    graph.output(output)
    graph_module = torch.fx.GraphModule({}, graph)
    real_input = torch.randn(size, size, device="cuda")
    fake_mode = FakeTensorMode(shape_env=ShapeEnv())
    fake_input = fake_mode.from_tensor(real_input, static_shapes=True)
    return graph_module, [fake_input]


def check_symbolic_fake_tensor_view() -> None:
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    symbolic_size = shape_env.create_unbacked_symint()
    with fake_mode:
        base = torch.empty((symbolic_size, 1))
        view = base.view(symbolic_size)

    restored = GraphPickler.loads(
        GraphPickler.dumps(view),
        FakeTensorMode(shape_env=ShapeEnv()),
    )
    if str(restored.shape[0]) != str(symbolic_size) or not restored._is_view():
        raise RuntimeError("GraphPickler did not preserve the symbolic view")
    print("graph_pickler_symbolic_view=passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--testing-sleep-seconds",
        type=float,
        default=0.0,
        help="Inject an Inductor testing delay to make overlap observable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.jobs < 1:
        raise ValueError("workers and jobs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    check_symbolic_fake_tensor_view()

    os.environ["TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS"] = str(args.workers)
    os.environ.setdefault("TORCHINDUCTOR_FX_COMPILE_INNER_THREADS", "1")

    started = time.perf_counter()
    prewarm = FullFxCompilePool.prewarm_workers()
    for future in prewarm:
        future.result()

    futures = {}
    config = {
        "fx_graph_cache": True,
        "fx_graph_remote_cache": False,
    }
    if args.testing_sleep_seconds > 0:
        config["sleep_sec_TESTING_ONLY"] = args.testing_sleep_seconds

    for variant in range(args.jobs):
        graph, inputs = make_job(variant, args.size)
        future = FullFxCompilePool.submit(graph, inputs, config)
        futures[future] = variant

    worker_pids: set[int] = set()
    artifact_count = 0
    for future in as_completed(futures):
        result = future.result()
        worker_pids.add(result.worker_pid)
        artifact_count += len(result.artifacts)
        print(
            f"job={futures[future]} pid={result.worker_pid} "
            f"compile_s={result.compile_time_s:.6f} "
            f"artifacts={len(result.artifacts)}",
            flush=True,
        )

    FullFxCompilePool.shutdown_pool()
    elapsed = time.perf_counter() - started
    print(
        f"jobs={args.jobs} workers_seen={len(worker_pids)} "
        f"artifacts={artifact_count} wall_s={elapsed:.6f}",
        flush=True,
    )
    if artifact_count < args.jobs:
        raise RuntimeError("one or more jobs produced no cache artifact")
    if args.jobs >= 2 and args.workers >= 2 and len(worker_pids) < 2:
        raise RuntimeError("jobs did not execute in multiple workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
