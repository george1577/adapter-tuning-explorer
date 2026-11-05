import torch

from adapter_lab.lora import full_params, lora_params, scaling, total_lora_params, toy_layer


def test_parameter_counts():
    assert full_params(768, 768) == 589_824
    assert lora_params(768, 768, 16) == 24_576
    assert total_lora_params([(10, 20), (30, 40)], 2) == 200


def test_untrained_lora_changes_nothing():
    layer = toy_layer(8, 4, 16, trained=False)
    assert torch.equal(layer.adapted, layer.W)
    assert layer.delta_rank == 0


def test_rank_of_the_change_is_capped_by_r():
    for r in (1, 2, 4):
        assert toy_layer(8, r, 16, trained=True).delta_rank == r


def test_scaling_shrinks_as_r_grows():
    assert scaling(16, 16) == 1
    assert scaling(16, 32) == 0.5
    assert toy_layer(8, 4, 16, trained=True).scale == 4
