import pytest
import torch

from adapter_lab.quant import FP4_LEVELS, NF4_LEVELS, ModelCounts, bits_per_weight, mean_abs_error, quantize, weights_gb


def test_there_are_16_levels():
    assert len(NF4_LEVELS) == len(FP4_LEVELS) == 16


def test_quantized_values_come_from_the_levels_scaled_by_the_block_maximum():
    weights = torch.tensor([0.5, -0.25, 0.0, 0.1])
    result = quantize(weights, NF4_LEVELS, block_size=4)
    allowed = NF4_LEVELS * 0.5
    assert all((allowed - v).abs().min() < 1e-6 for v in result)
    assert result[0] == pytest.approx(0.5)


def test_nf4_beats_even_spacing_on_bell_shaped_weights():
    weights = torch.randn(64_000, generator=torch.Generator().manual_seed(0))
    nf4 = mean_abs_error(weights, quantize(weights, NF4_LEVELS))
    fp4 = mean_abs_error(weights, quantize(weights, FP4_LEVELS))
    assert nf4 < fp4


def test_all_zero_block_stays_zero():
    assert quantize(torch.zeros(64), NF4_LEVELS).abs().sum() == 0


def test_double_quantization_saves_about_0_4_bits():
    assert bits_per_weight(False) == 4.5
    assert bits_per_weight(True) == pytest.approx(4.127, abs=0.001)


def test_memory_estimate():
    counts = ModelCounts(total=1_000_000_000, quantizable=800_000_000, lora_at_r16=10_000_000, modules_to_save=100_000_000)
    assert weights_gb(counts, quantized=False) == pytest.approx(2.0)
    assert weights_gb(counts, quantized=True, double_quant=False) == pytest.approx(0.45 + 0.4)
    assert weights_gb(counts, quantized=False, lora=True, modules_to_save=True) == pytest.approx(2.22)
