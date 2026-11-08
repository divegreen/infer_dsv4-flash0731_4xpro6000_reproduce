"""Verify this installation on physical GPUs 0–3 without downloading a model."""

import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"

# Verify the environment belonging to this copied project.
project = Path(__file__).resolve().parents[1]
configured_python = Path(os.environ.get('VLLM_PYTHON', '.venv-vllm/bin/python')).expanduser()
expected = (configured_python if configured_python.is_absolute() else project / configured_python).absolute().parent.parent
assert Path(sys.prefix).resolve() == expected.resolve(), (sys.prefix, str(expected))

import torch
import vllm

print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}, indent=2), flush=True)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4

query = subprocess.check_output([
    "nvidia-smi", "--id=0,1,2,3", "--query-gpu=index,uuid",
    "--format=csv,noheader,nounits",
], text=True)
expected_uuids = {int(index): uuid.strip() for index, uuid in csv.reader(query.splitlines())}

importlib.import_module("vllm._C_stable_libtorch")
from vllm import _custom_ops as ops

print("PASS: vLLM native CUDA extension imported", flush=True)
for index in range(4):
    properties = torch.cuda.get_device_properties(index)
    actual_uuid = str(properties.uuid)
    assert actual_uuid.removeprefix("GPU-") == expected_uuids[index].removeprefix("GPU-"), (
        index, actual_uuid, expected_uuids[index]
    )
    with torch.cuda.device(index):
        x = torch.arange(512, device=f"cuda:{index}", dtype=torch.float32).reshape(4, 128) / 512
        product = x @ x.T
        torch.testing.assert_close(product.cpu(), x.cpu() @ x.cpu().T)
        x = x.to(torch.float16)
        weight = torch.ones(128, device=x.device, dtype=x.dtype)
        output = torch.empty_like(x)
        ops.rms_norm(output, x, weight, 1e-6)
        reference = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(output.float(), reference, rtol=2e-3, atol=2e-3)
        torch.cuda.synchronize(index)
        print(json.dumps({
            "physical_gpu": index,
            "logical_gpu": index,
            "uuid": actual_uuid,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch_matmul": "PASS",
            "vllm_rms_norm": "PASS",
        }), flush=True)
        del x, product, weight, output, reference
        torch.cuda.empty_cache()

print("PASS: all four GPUs verified; model inference and collective communication NOT VERIFIED", flush=True)
