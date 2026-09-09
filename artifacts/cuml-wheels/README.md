# cuML SPORF wheel artifacts

This directory contains an internal prebuilt cuML wheel and a compatibility
baseline for collaborators who need to run the benchmarks without building the
SPORF cuML fork locally.

Current artifact set:

```text
cuml_cu12-25.10.0-cp313-cp313-linux_x86_64.whl
cuml_env_baseline.json
```

The wheel is for Linux x86_64 / CPython 3.13 / CUDA 12 RAPIDS packages. WSL is
acceptable as the target runtime as long as NVIDIA GPU support is visible inside
WSL and the conda environment matches the baseline closely enough.

From the repository root:

```bash
conda env create -n cuml_dev -f ./environment.yml
conda activate cuml_dev

python ./src/check_cuml_runtime_env.py \
  --wheel ./artifacts/cuml-wheels/cuml_cu12-25.10.0-cp313-cp313-linux_x86_64.whl \
  --baseline ./artifacts/cuml-wheels/cuml_env_baseline.json

python -m pip install --no-deps \
  ./artifacts/cuml-wheels/cuml_cu12-25.10.0-cp313-cp313-linux_x86_64.whl
```

Use `--no-deps` when installing the wheel. The RAPIDS/CUDA dependency stack is
managed by `environment.yml`; allowing pip to resolve RAPIDS dependencies can
pull incompatible wheels or source stubs.

After installation, a quick import check is:

```bash
python - <<'PY'
from cuml.ensemble import SPORFClassifier
print(SPORFClassifier)
PY
```
