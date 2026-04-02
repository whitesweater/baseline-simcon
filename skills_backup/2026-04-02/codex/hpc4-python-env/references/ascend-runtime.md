# Ascend Runtime Checklist

## Fast Checks

1. Confirm the project lives under `/data/user/user224/proj`
2. `npu-smi info`
3. `python3 --version`
4. `uv --version`
5. `find /usr/local/Ascend /opt/conda/Ascend -name set_env.sh 2>/dev/null`
6. `find / -name libhccl.so 2>/dev/null`

## Healthy State

- `npu-smi` works
- `torch` imports
- `torch_npu` imports
- `torch.npu.is_available()` returns `True`
- `torch.tensor(..., device="npu:0")` works
- `torch.zeros(..., device="npu:0")` works
- `torch.arange(..., device="npu:0")` works
- `libhccl.so` exists somewhere on the container
- The container exposes more than just `/usr/local/Ascend/driver`
- The current shell sourced the correct `set_env.sh` before validation

## Common Failure

Symptom:

```text
ImportError: libhccl.so: cannot open shared object file
```

Meaning:

- The container has driver visibility, but does not have the full CANN user-space runtime available to the Python process.
- This is not fixed by changing application code.

Likely causes:

- The image only mounts `/usr/local/Ascend/driver`
- `ascend-toolkit` is missing
- The image was not built from an Ascend-ready base
- Required runtime libraries are installed elsewhere but not exposed via `LD_LIBRARY_PATH`
- The right `set_env.sh` exists, but the current shell did not source it

Another common failure is partial success:

```text
torch.tensor(...) works
torch.zeros(...) fails
torch.arange(...) fails
```

Treat that as a broken Ascend runtime or OPP stack. Do not mark the environment healthy yet.

## What To Do

1. Confirm the target project is inside `/data/user/user224/proj`.
2. Inspect whether `/usr/local/Ascend/ascend-toolkit` or `/opt/conda/Ascend/ascend-toolkit` exists.
3. Find and source the correct `set_env.sh`.
4. Search for `libhccl.so`.
5. Check whether the platform provides a public working Python env that already imports `torch_npu`.
6. If no working env exists and `libhccl.so` is absent, stop and ask for:
   - a different container image, or
   - permission to install the full Ascend/CANN runtime

## Versioning Guidance

- Match `torch_npu` exactly to `torch`.
- Prefer a Python version for which both wheels exist on the chosen mirror.
- When a repo asks for `torch==2.4.1`, but Ascend only has a compatible `torch_npu` for `torch==2.4.0`, decide explicitly whether to:
  - use the nearest compatible Ascend pair, or
  - stay strict to the repo requirement and give up NPU support
- Prefer the platform-supported pair when the user's goal is "make it run" rather than "match the repo pin exactly"
