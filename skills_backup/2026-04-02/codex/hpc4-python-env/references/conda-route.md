# Conda Route

Use this route only when the user explicitly asks for Conda or when the current HPC4 image makes the `uv` route unsuitable.

## Guardrails

- Keep the project under `/data/user/user224/proj/<project>`
- Disable proxies before downloads
- Source Ascend first, then source `conda.sh`, then activate the env
- Align `torch` and `torch_npu` to a platform-supported pair

## Path Discovery

```bash
for candidate in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
  /opt/conda/Ascend/ascend-toolkit/set_env.sh
do
  [ -f "$candidate" ] && ASCEND_SET_ENV="$candidate" && break
done

if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_HOME="$(cd "$(dirname "$CONDA_BIN")/.." && pwd)"
fi

CONDA_SH="$CONDA_HOME/etc/profile.d/conda.sh"
```

## Standard Flow

```bash
cd /data/user/user224/proj/<project>
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY no_proxy NO_PROXY

source "$ASCEND_SET_ENV"
source "$CONDA_SH"
conda create -n <env-name> python=3.10 -y
conda activate <env-name>
pip install -U pip setuptools wheel
```

Install a platform-supported `torch` + `torch_npu` pair, then the project dependencies.

## Validation

Always require:

- `torch_npu` import
- `torch.npu.device_count()`
- `torch.tensor(..., device="npu:0")`
- `torch.zeros(..., device="npu:0")`
- `torch.arange(..., device="npu:0")`
