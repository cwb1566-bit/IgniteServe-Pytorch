# IgniteServe-Pytorch Deployment

## Validated Runtime

| Component | Version |
|---|---|
| Base commit | PyTorch `v2.7.0`, `134179474539648ba7dee1317959529fbd0e7f89` |
| Python | 3.10 |
| PyTorch runtime | 2.7.0+cu126 |
| CUDA runtime | 12.6 |
| Triton | 3.3.0 |
| GPU | NVIDIA A100 PCIe 40 GB |

## Option A: Build the Fork From Source

Use PyTorch's standard source-build prerequisites. Clone submodules before building:

```bash
git clone --recursive git@github.com:cwb1566-bit/IgniteServe-Pytorch.git
cd IgniteServe-Pytorch

conda create -n igniteserve-pytorch python=3.10 -y
conda activate igniteserve-pytorch
pip install -r requirements.txt

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-$(dirname $(which conda))/../}"
export USE_CUDA=1
python setup.py develop
```

A full PyTorch source build can be lengthy. Follow the upstream README for compiler, CUDA, cuDNN, MKL, and platform-specific prerequisites.

Verify that the new module is present:

```bash
python -c 'import torch; from torch._inductor.compile_fx_full_subproc import FullFxCompilePool; print(torch.__version__, FullFxCompilePool)'
```

Run the standalone CUDA smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/igniteserve_parallel_compile_smoke.py \
  --workers 2 --jobs 2
```

## Option B: Install the Python Overlay

All retained changes are Python-only. For the exact PyTorch 2.7 runtime, the repository provides a guarded overlay installer:

```bash
python tools/install_igniteserve_overlay.py --python /path/to/python
```

The installer:

- imports the target interpreter's PyTorch;
- requires version `2.7.*` unless `--allow-version-mismatch` is explicit;
- verifies expected destination modules;
- creates a timestamped backup under the target `torch` package;
- copies only the four runtime modules;
- writes a JSON installation manifest;
- never modifies the compiled `_C` extension.

Restore the latest overlay backup:

```bash
python tools/install_igniteserve_overlay.py \
  --python /path/to/python \
  --restore
```

Use a dedicated Conda environment. Overlaying a shared system PyTorch installation is not recommended.

## IgniteServe Integration

Install the matching private IgniteServe repository in the same environment. For an expected cold compile-cache miss:

```bash
export IGNITESERVE_ROOT=/opt/IgniteServe
export TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS=4
export TORCHINDUCTOR_FX_COMPILE_INNER_THREADS=1
export TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS=vllm._custom_ops
export TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA=1

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$IGNITESERVE_ROOT \
VLLM_USE_V1=1 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_ENABLE_SUBPROCESS_COMPILE=1 \
VLLM_ENABLE_PARALLEL_COMPILE_METADATA=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /models/Llama-3-8B \
  --served-model-name Llama-3-8B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9
```

For a known warm compile cache, omit `VLLM_ENABLE_SUBPROCESS_COMPILE=1`. The normal IgniteServe S4 loading path will load canonical artifacts and reuse follower graphs without creating compiler workers.

## Lifecycle Requirement

Compiler workers may initialize CUDA while restoring FakeTensors or compiling kernels. IgniteServe begins pool teardown after materialization and calls `wait_for_shutdown_pool()` before KV-memory profiling. Integrations must preserve that ordering; otherwise worker CUDA contexts can reduce the serving process's measured KV capacity.

## Troubleshooting

### GraphPickler rejects a custom operator

Set `TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS` to the module that registers the operator. The name must resolve under `torch.ops.*` after import in each worker.

### Nested compilation creates too many processes

Keep `TORCHINDUCTOR_FX_COMPILE_INNER_THREADS=1`. Increase complete-FX workers only after measuring CPU, RAM, disk-cache, and GPU-context pressure.

### Warm startup is slower

Do not enable the cold-only vLLM subprocess switch on a known warm cache. Pool creation and teardown add avoidable work when no graph must compile.

### KV memory is lower after compilation

Confirm the integration waits for `FullFxCompilePool.wait_for_shutdown_pool()` before memory profiling and that no worker processes remain.
