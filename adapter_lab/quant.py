"""Simulating 4-bit quantization and estimating memory. Real QLoRA uses bitsandbytes, which needs a
GPU; this reproduces its arithmetic on ordinary tensors so the effect can be seen on a laptop."""

from dataclasses import dataclass

import torch

BLOCK_SIZE = 64  # bitsandbytes scales weights in blocks of 64 values

# The 16 NF4 levels from the QLoRA paper, as used by bitsandbytes.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])
FP4_LEVELS = torch.linspace(-1.0, 1.0, 16)  # evenly spaced, for comparison

GB = 1e9


def quantize(weights: torch.Tensor, levels: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Scale each block by its largest magnitude, snap every value to the nearest level, scale back."""
    flat = weights.flatten()
    out = torch.empty_like(flat)
    for start in range(0, flat.numel(), block_size):
        block = flat[start : start + block_size]
        scale = block.abs().max()
        if scale == 0:
            out[start : start + block_size] = 0
            continue
        nearest = (block / scale).unsqueeze(1).sub(levels).abs().argmin(dim=1)
        out[start : start + block_size] = levels[nearest] * scale
    return out.reshape(weights.shape)


def mean_abs_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    return (original - quantized).abs().mean().item()


def bits_per_weight(double_quant: bool, block_size: int = BLOCK_SIZE) -> float:
    """4 bits per weight, plus each block's scale: 32 bits, or about 8 with double quantization."""
    scale_bits = 8 + 32 / 256 if double_quant else 32
    return 4 + scale_bits / block_size


@dataclass(frozen=True)
class ModelCounts:
    """Parameter counts for a model. Quantization only applies to linear layers other than lm_head."""
    total: int
    quantizable: int
    lora_at_r16: int
    modules_to_save: int


def weights_gb(counts: ModelCounts, *, quantized: bool, double_quant: bool = True,
               lora: bool = False, modules_to_save: bool = False, trainable_bits: int = 16) -> float:
    """Memory for the weights alone, ignoring gradients, optimizer state and the tensors of a batch."""
    if quantized:
        bits = counts.quantizable * bits_per_weight(double_quant)
        bits += (counts.total - counts.quantizable) * trainable_bits
    else:
        bits = counts.total * trainable_bits
    if lora:
        bits += counts.lora_at_r16 * trainable_bits
    if modules_to_save:
        bits += counts.modules_to_save * trainable_bits
    return bits / 8 / GB
