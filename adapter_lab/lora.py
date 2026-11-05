"""The arithmetic of LoRA, without any model: one layer is W + (alpha / r) * B * A."""

from dataclasses import dataclass

import torch


def full_params(d_in: int, d_out: int) -> int:
    return d_in * d_out


def lora_params(d_in: int, d_out: int, r: int) -> int:
    """A has shape r x d_in and B has shape d_out x r."""
    return r * (d_in + d_out)


def scaling(alpha: float, r: int) -> float:
    return alpha / r


@dataclass(frozen=True)
class ToyLayer:
    W: torch.Tensor
    A: torch.Tensor
    B: torch.Tensor
    delta: torch.Tensor
    adapted: torch.Tensor
    scale: float
    delta_rank: int


def toy_layer(size: int, r: int, alpha: float, trained: bool, seed: int = 0) -> ToyLayer:
    """A small layer with LoRA attached. Untrained, B is all zeros, exactly as PEFT initialises it."""
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(size, size, generator=gen)
    A = torch.randn(r, size, generator=gen) / size**0.5
    B = torch.randn(size, r, generator=gen) * 0.5 if trained else torch.zeros(size, r)
    scale = scaling(alpha, r)
    delta = scale * (B @ A)
    return ToyLayer(W, A, B, delta, W + delta, scale, int(torch.linalg.matrix_rank(delta)))


def total_lora_params(linear_shapes: list[tuple[int, int]], r: int) -> int:
    return sum(lora_params(d_in, d_out, r) for d_in, d_out in linear_shapes)
