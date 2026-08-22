from __future__ import annotations

import atexit
import contextlib
import dataclasses
import functools
import importlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING
from unittest import mock

import torch
from torch._inductor.compile_worker.subproc_pool import SubprocKind, SubprocPool
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import config


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from concurrent.futures import Future

    from torch._inductor.utils import InputType
    from torch.fx import GraphModule


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


def _import_worker_modules(value: str) -> None:
    for module_name in value.split(","):
        module_name = module_name.strip()
        if module_name:
            importlib.import_module(module_name)


@dataclass
class FullFxCompileArtifact:
    cache_key: str
    artifact_path: str


@dataclass
class FullFxCompileResult:
    artifacts: list[FullFxCompileArtifact]
    compile_time_s: float
    worker_pid: int
    serialized_input_nbytes: int
    timings: dict[str, float]


@dataclass
class FullFxPrewarmResult:
    worker_pid: int
    prewarm_time_s: float
    imports_time_s: float
    cuda_init_time_s: float


@dataclass
class _FullFxCompileInput:
    gm: GraphModule
    example_inputs: Sequence[InputType]
    config_patches: dict[str, Any]
    tracing_context: torch._guards.TracingContext
    unsafe_force_cache: bool
    worker_context_factories: tuple[Callable[[], Any], ...]

    def serialize(self) -> bytes:
        from torch.fx._graph_pickler import GraphPickler

        try:
            return GraphPickler.dumps(self)
        except Exception as exc:
            failures = []
            components = (
                ("gm", self.gm),
                ("example_inputs", self.example_inputs),
                ("config_patches", self.config_patches),
                ("tracing_context", self.tracing_context),
                ("worker_context_factories", self.worker_context_factories),
            )
            for name, value in components:
                try:
                    GraphPickler.dumps(value)
                except Exception as component_exc:
                    failures.append(f"{name}: {component_exc!r}")
                    if name == "example_inputs":
                        leaves, _ = torch.utils._pytree.tree_flatten(value)
                        for index, leaf in enumerate(leaves):
                            try:
                                GraphPickler.dumps(leaf)
                            except Exception as leaf_exc:
                                shape = getattr(leaf, "shape", None)
                                failures.append(
                                    f"example_inputs[{index}] "
                                    f"type={type(leaf)!r} shape={shape!r}: "
                                    f"{leaf_exc!r}"
                                )
                                if isinstance(
                                    leaf, torch._subclasses.fake_tensor.FakeTensor
                                ):
                                    from torch._subclasses.meta_utils import (
                                        MetaTensorDescriber,
                                    )
                                    from torch.fx._graph_pickler import (
                                        _ShapeEnvPickleData,
                                        _TensorPickleData,
                                    )

                                    tensor_data = _TensorPickleData(
                                        MetaTensorDescriber(copy_data=False), leaf
                                    ).metadata
                                    for field in dataclasses.fields(tensor_data):
                                        field_value = getattr(tensor_data, field.name)
                                        try:
                                            GraphPickler.dumps(field_value)
                                        except Exception as field_exc:
                                            failures.append(
                                                f"example_inputs[{index}].metadata."
                                                f"{field.name}: {field_exc!r}"
                                            )
                                    shape_env = leaf.fake_mode.shape_env
                                    if shape_env is not None:
                                        shape_env_data = _ShapeEnvPickleData(
                                            shape_env
                                        ).data
                                        for field_name, field_value in (
                                            shape_env_data.items()
                                        ):
                                            try:
                                                GraphPickler.dumps(field_value)
                                            except Exception as field_exc:
                                                if (
                                                    "WeakValueDictionary"
                                                    in repr(field_exc)
                                                    or "weakref" in repr(field_exc)
                                                ):
                                                    failures.append(
                                                        "shape_env."
                                                        f"{field_name}: "
                                                        f"{field_exc!r}"
                                                    )
            detail = "; ".join(failures) or "individual components succeeded"
            raise RuntimeError(
                f"Full FX compile payload serialization failed ({detail})"
            ) from exc

    @staticmethod
    def deserialize(value: bytes) -> _FullFxCompileInput:
        from torch.fx._graph_pickler import GraphPickler

        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        result = GraphPickler.loads(value, fake_mode)
        assert isinstance(result, _FullFxCompileInput)
        return result


def _artifact_path(output_code: Any) -> str:
    current_callable = output_code.current_callable
    path = current_callable.__code__.co_filename
    for cell in current_callable.__closure__ or ():
        value = cell.cell_contents
        if callable(value) and hasattr(value, "__code__"):
            candidate = value.__code__.co_filename
            if candidate:
                path = candidate
    return path


def _run_full_fx_prewarm(extra_env: Mapping[str, str]) -> FullFxPrewarmResult:
    """Import the expensive worker-side compiler/framework modules early."""
    started = time.perf_counter()
    imports_time_s = 0.0
    cuda_init_time_s = 0.0
    with contextlib.ExitStack() as stack:
        if extra_env:
            stack.enter_context(mock.patch.dict(os.environ, extra_env))
        phase_started = time.perf_counter()
        _import_worker_modules(
            extra_env.get("TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS", "")
        )
        import torch._inductor.compile_fx  # noqa: F401
        import torch.fx._graph_pickler  # noqa: F401
        import sympy.combinatorics.polyhedron  # noqa: F401
        import sympy.tensor.tensor  # noqa: F401
        imports_time_s = time.perf_counter() - phase_started

        if extra_env.get("TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            phase_started = time.perf_counter()
            from torch._subclasses.fake_tensor import init_gpu_context

            init_gpu_context(torch.device("cuda"))
            cuda_init_time_s = time.perf_counter() - phase_started

    return FullFxPrewarmResult(
        os.getpid(),
        time.perf_counter() - started,
        imports_time_s,
        cuda_init_time_s,
    )


def _run_full_fx_compile(
    serialized_input: bytes,
    extra_env: Mapping[str, str],
) -> FullFxCompileResult:
    worker_started = time.perf_counter()
    timings: dict[str, float] = {}
    with contextlib.ExitStack() as stack:
        if extra_env:
            stack.enter_context(mock.patch.dict(os.environ, extra_env))

        phase_started = time.perf_counter()
        _import_worker_modules(
            extra_env.get("TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS", "")
        )
        timings["worker_imports_s"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        inner_threads = _positive_int_env(
            "TORCHINDUCTOR_FX_COMPILE_INNER_THREADS", 1
        )
        stack.enter_context(config.patch(compile_threads=inner_threads))

        # Direct compile_fx callers normally inherit this context from
        # TorchDynamo's convert_frame path.  A standalone worker does not, but
        # cache-hit and async-compile branches still publish compilation
        # metrics through CompileEventLogger.
        from torch._dynamo.utils import get_metrics_context

        stack.enter_context(get_metrics_context())
        timings["worker_runtime_setup_s"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        profile_dir = os.environ.get("TORCHINDUCTOR_GRAPH_PICKLER_PROFILE_DIR")
        if profile_dir:
            import cProfile
            import pstats

            os.makedirs(profile_dir, exist_ok=True)
            profile = cProfile.Profile()
            profile.enable()
            compile_input = _FullFxCompileInput.deserialize(serialized_input)
            profile.disable()
            profile_stem = os.path.join(
                profile_dir,
                f"graph_pickler_{os.getpid()}_{time.time_ns()}",
            )
            profile.dump_stats(f"{profile_stem}.pstats")
            with open(f"{profile_stem}.txt", "w", encoding="utf-8") as stream:
                stats = pstats.Stats(profile, stream=stream)
                stats.strip_dirs().sort_stats("cumulative").print_stats(120)
                stats.sort_stats("tottime").print_stats(120)
        else:
            compile_input = _FullFxCompileInput.deserialize(serialized_input)
        timings["worker_deserialize_s"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        stack.enter_context(torch._guards.tracing(compile_input.tracing_context))
        stack.enter_context(torch.inference_mode())
        for context_factory in compile_input.worker_context_factories:
            stack.enter_context(context_factory())

        import torch._inductor.compile_fx as compile_fx_module

        if compile_input.unsafe_force_cache:
            from torch._inductor.codecache import FxGraphCache

            stack.enter_context(
                mock.patch.object(
                    FxGraphCache,
                    "_check_can_cache",
                    staticmethod(lambda gm: None),
                )
            )

        # The entire compile is already isolated in this worker.  Avoid nesting
        # another FX subprocess pool at the lower codegen boundary.
        stack.enter_context(
            mock.patch.object(
                compile_fx_module, "fx_compile_mode", compile_fx_module.FxCompileMode.NORMAL
            )
        )
        timings["worker_compile_context_s"] = time.perf_counter() - phase_started

        artifacts: list[FullFxCompileArtifact] = []

        def capture_inner(gm, example_inputs, **kwargs):
            output_code = compile_fx_module.compile_fx_inner(
                gm, example_inputs, **kwargs
            )
            cache_key = output_code._fx_graph_cache_key
            if not cache_key:
                raise RuntimeError(
                    "Full FX subprocess compilation did not produce an FX cache key"
                )
            artifacts.append(
                FullFxCompileArtifact(cache_key, _artifact_path(output_code))
            )
            return output_code

        started = time.perf_counter()
        compile_fx_module.compile_fx(
            compile_input.gm,
            compile_input.example_inputs,
            inner_compile=capture_inner,
            config_patches=compile_input.config_patches,
        )
        compile_time_s = time.perf_counter() - started
        timings["compile_fx_s"] = compile_time_s
        timings["worker_total_s"] = time.perf_counter() - worker_started
        return FullFxCompileResult(
            artifacts,
            compile_time_s,
            os.getpid(),
            len(serialized_input),
            timings,
        )


class FullFxCompilePool:
    """Persistent process pool for complete, mutually isolated compile_fx jobs."""

    _prewarm_futures: list[Future[FullFxPrewarmResult]] = []
    _shutdown_thread: Optional[threading.Thread] = None
    _shutdown_error: Optional[BaseException] = None
    _shutdown_elapsed_s: float = 0.0

    @staticmethod
    @functools.cache
    def process_pool() -> SubprocPool:
        pool = SubprocPool(
            _positive_int_env("TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS", 1),
            kind=SubprocKind.SPAWN,
        )
        atexit.register(pool.shutdown)
        return pool

    @classmethod
    def begin_shutdown_pool(cls) -> bool:
        """Start releasing compiler workers without blocking the caller."""
        if cls._shutdown_thread is not None:
            return False
        if cls.process_pool.cache_info().currsize == 0:
            return False
        pool = cls.process_pool()
        cls.process_pool.cache_clear()
        cls._prewarm_futures = []
        cls._shutdown_error = None
        cls._shutdown_elapsed_s = 0.0

        def shutdown() -> None:
            started = time.perf_counter()
            try:
                fast = os.environ.get(
                    "TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN", ""
                ).strip().lower() in {"1", "true", "yes", "on"}
                if fast:
                    pool.shutdown_fast()
                else:
                    pool.shutdown()
            except BaseException as exc:
                cls._shutdown_error = exc
            finally:
                cls._shutdown_elapsed_s = time.perf_counter() - started

        cls._shutdown_thread = threading.Thread(
            target=shutdown,
            name="full-fx-compile-pool-shutdown",
            daemon=True,
        )
        cls._shutdown_thread.start()
        return True

    @classmethod
    def wait_for_shutdown_pool(cls) -> float:
        """Wait until compiler workers and their CUDA contexts are gone."""
        thread = cls._shutdown_thread
        if thread is None:
            return 0.0
        thread.join()
        cls._shutdown_thread = None
        error = cls._shutdown_error
        cls._shutdown_error = None
        elapsed_s = cls._shutdown_elapsed_s
        cls._shutdown_elapsed_s = 0.0
        if error is not None:
            raise error
        return elapsed_s

    @classmethod
    def shutdown_pool(cls) -> None:
        """Synchronously release compiler workers and their CUDA contexts."""
        cls.begin_shutdown_pool()
        cls.wait_for_shutdown_pool()

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        env_vars = (
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS",
            "TORCHINDUCTOR_FX_COMPILE_INNER_THREADS",
            "TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA",
            "TORCHINDUCTOR_GRAPH_PICKLER_PROFILE_DIR",
        )
        return {name: os.environ[name] for name in env_vars if name in os.environ}

    @classmethod
    def prewarm_workers(cls) -> list[Future[FullFxPrewarmResult]]:
        if cls._prewarm_futures:
            return cls._prewarm_futures
        worker_count = _positive_int_env(
            "TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS", 1
        )
        extra_env = cls._worker_environment()
        pool = cls.process_pool()
        cls._prewarm_futures = [
            pool.submit(_run_full_fx_prewarm, extra_env)
            for _ in range(worker_count)
        ]
        return cls._prewarm_futures

    @staticmethod
    def _tracing_context(
        example_inputs: Sequence[InputType],
    ) -> torch._guards.TracingContext:
        fake_mode = torch._guards.detect_fake_mode(example_inputs)
        if fake_mode is None:
            raise RuntimeError(
                "FullFxCompilePool requires FakeTensor example inputs"
            )
        # A full compile_fx job starts a new AOTAutograd trace in an isolated
        # process.  Reusing the parent's live TracingContext carries unrelated
        # Dynamo/AOT state (including weak-reference-backed module mappings)
        # into the IPC payload and is neither serializable nor semantically
        # required for a standalone piecewise graph.  Preserve the input
        # FakeTensorMode/ShapeEnv, but start with a clean tracing context.
        return torch._guards.TracingContext(fake_mode)

    @classmethod
    def submit(
        cls,
        gm: GraphModule,
        example_inputs: Sequence[InputType],
        config_patches: Optional[dict[str, Any]] = None,
        *,
        unsafe_force_cache: bool = False,
        worker_context_factories: tuple[Callable[[], Any], ...] = (),
    ) -> Future[FullFxCompileResult]:
        compile_input = _FullFxCompileInput(
            gm,
            example_inputs,
            dict(config_patches or {}),
            cls._tracing_context(example_inputs),
            unsafe_force_cache,
            worker_context_factories,
        )
        serialized_input = compile_input.serialize()
        extra_env = cls._worker_environment()
        return cls.process_pool().submit(
            _run_full_fx_compile, serialized_input, extra_env
        )
