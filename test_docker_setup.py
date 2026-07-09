"""Quick, offline smoke test for the tf-512-gpu Docker image.

Run inside the container:
    pytest test_docker_setup.py -v

No network access and no model downloads, so it should finish in seconds.
"""
import importlib

import pytest


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"


def test_cuda_tensor_op():
    import torch
    x = torch.zeros(8, 8, dtype=torch.float32, device="cuda")
    assert (x + 1).sum().item() == 64.0


def test_gpu_device_name():
    import torch
    name = torch.cuda.get_device_name(0)
    assert name


@pytest.mark.parametrize(
    "module",
    [
        "flash_attn",
        "transformers",
        "tokenizers",
        "safetensors",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "deepspeed",
        "numpy",
        "scipy",
        "sklearn",
        "einops",
        "sentencepiece",
        "tiktoken",
        "wandb",
        "gpustat",
    ],
)
def test_package_importable(module):
    importlib.import_module(module)
