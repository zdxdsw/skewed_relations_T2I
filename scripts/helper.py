import torch

TORCH_DTYPES = {
    'no': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.float16,
}

def exists(val):
    return val is not None
