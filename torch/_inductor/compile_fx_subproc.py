from __future__ import annotations

import atexit
import functools
import importlib
import os
from typing import Optional, TYPE_CHECKING
from typing_extensions import override

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch.fx
from torch._inductor.compile_worker.subproc_pool import (
    AnyPool,
    SubprocKind,
    SubprocPool,
)
from torch._inductor.utils import clear_inductor_caches

from . import config
from .compile_fx_ext import (
    _OutOfProcessFxCompile,
    _WireProtocolPickledInput,
    _WireProtocolPickledOutput,
)
from .output_code import complex_memory_overlap as complex_memory_overlap  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Mapping


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise RuntimeError(f"{name} must be at least 1, got {parsed}")
    return parsed


def _import_worker_modules(extra_env: Optional[Mapping[str, str]]) -> None:
    value = None if extra_env is None else extra_env.get(
        "TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS"
    )
    if value is None:
        value = os.environ.get("TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS", "")
    for module_name in value.split(","):
        module_name = module_name.strip()
        if module_name:
            importlib.import_module(module_name)


class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @override
    def _send_to_child(
        self, input: _WireProtocolPickledInput
    ) -> _WireProtocolPickledOutput:
        # TODO: Do we need to copy across some kind of logging IDs? (ChromiumEventLogger)

        pool = self.process_pool()

        # TODO: This is probably the wrong thing to do long-term - but for now
        # let's share the cache so we can identify tests broken by this later.
        env_vars = [
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS",
            "TORCHINDUCTOR_FX_COMPILE_INNER_THREADS",
        ]
        extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}

        f = pool.submit(_SubprocessFxCompile._run_in_child_subprocess, input, extra_env)

        # For debugging: If we want to print status updates...
        # last = time.time()
        # while not f.done():
        #     print("tick...")
        #     time.sleep(0.125)
        #     now = time.time()
        #     if now - last > 1:
        #         last = now

        output = f.result()

        return output

    @staticmethod
    @functools.cache
    def process_pool() -> AnyPool:
        pool = SubprocPool(
            _positive_int_env("TORCHINDUCTOR_FX_COMPILE_WORKERS", 1),
            kind=SubprocKind.SPAWN,
        )

        atexit.register(pool.shutdown)

        return pool

    @classmethod
    def _run_in_child_subprocess(
        cls,
        pickled_input: _WireProtocolPickledInput,
        extra_env: Optional[Mapping[str, str]],
    ) -> _WireProtocolPickledOutput:
        # TODO: In subprocess mode we need to clear the inductor caches.
        # The problem:
        #   1. We compile in worker A which fills stuff in tmpdir
        #   2. parent clears inductor caches which deletes tmpdirs and tells
        #      cpp_prefix_path() to clear its LRU cache
        #   3. We compile a second time in subproc A - but since we never told
        #      cpp_prefix_path() in worker A to clear its LRU it thinks the
        #      tmpdir still exists and fails to compile.
        #
        # TODO: We probably should be using a separate tmpdir in the worker
        # anyway... but we should probably still respect clear_inductor_caches()
        # in the parent... maybe?
        #
        # TODO: We could be less aggressive by keeping a clock which gets
        # incremented when we clear the cache, send the clock to the worker and
        # only clear caches if the clock changed since last time.
        #
        _import_worker_modules(extra_env)

        clear_inductor_caches()
        torch._inductor.metrics.reset()

        # Graph-level workers already provide parallelism.  Default the nested
        # kernel compiler to one worker to avoid graph_workers * compile_threads
        # process oversubscription.  Advanced users can opt back in explicitly.
        inner_threads = _positive_int_env(
            "TORCHINDUCTOR_FX_COMPILE_INNER_THREADS", 1
        )
        with config.patch(compile_threads=inner_threads):
            return cls._run_in_child(pickled_input, extra_env)
