"""Integration tests on the real model. Each takes seconds to a minute on CPU."""

import pytest
import torch
from peft import get_peft_model
from PIL import Image

from adapter_lab.data import Example
from adapter_lab.labels import IGNORE
from adapter_lab.lora import total_lora_params
from adapter_lab.runs import load_run
from adapter_lab.training import TrainConfig, train
from adapter_lab.vlm import CaptionModel, encode, image_token_ids, labels_for, linear_layers, load_processor, make_lora_config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("images")
    examples = []
    for i in range(6):
        path = folder / f"{i}.jpg"
        Image.effect_noise((300, 200), 40 + i).convert("RGB").save(path)
        examples.append(Example(str(i), str(path), f"Axial computed tomography scan number {i}."))
    return examples


def test_caption_tokens_follow_the_prompt_exactly(images):
    processor = load_processor()
    prompt = encode(processor, images[0].path, None)
    full = encode(processor, images[0].path, images[0].caption)
    assert full.input_ids[: prompt.prompt_length] == prompt.input_ids
    caption = processor.tokenizer.decode(full.input_ids[prompt.prompt_length :], skip_special_tokens=True)
    assert caption.strip() == images[0].caption


def test_labels_hide_the_image_and_optionally_the_prompt(images):
    processor = load_processor()
    encoded = encode(processor, images[0].path, images[0].caption)
    notebook = labels_for(processor, encoded, mask_prompt=False)
    assert all(l == IGNORE for l, t in zip(notebook, encoded.input_ids) if t in image_token_ids(processor))
    assert any(l != IGNORE for l in notebook[: encoded.prompt_length])
    masked = labels_for(processor, encoded, mask_prompt=True)
    assert all(l == IGNORE for l in masked[: encoded.prompt_length])
    assert masked[encoded.prompt_length :] == encoded.input_ids[encoded.prompt_length :]


def test_arithmetic_matches_what_peft_actually_adds():
    from transformers import AutoConfig, AutoModelForImageTextToText

    from adapter_lab.vlm import MODEL_ID

    shapes = [(l["d_in"], l["d_out"]) for l in linear_layers()]
    with torch.device("meta"):
        meta = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(MODEL_ID))
    peft_model = get_peft_model(meta, make_lora_config(16, 16, False))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert trainable == total_lora_params(shapes, 16)


@pytest.mark.parametrize("modules_to_save", [False, True])
def test_training_keeps_base_frozen_and_adapter_reloads(images, tmp_path, modules_to_save):
    config = TrainConfig(examples=4, r=4, alpha=4, modules_to_save=modules_to_save)
    path = train(config, images[:4], images[4:], "t", tmp_path)
    run = load_run(path)

    assert run.results["frozen_weights_checked"] > 0
    assert run.results["frozen_weights_changed"] == []
    assert run.results["optimizer_steps"] == 1
    assert len(run.history) == 4
    assert run.history[-1]["adapter_norm"] > 0

    # Strength 0 is exactly the base model, and strength 1 is not.
    tuned = CaptionModel(run.adapter_dir)
    base = CaptionModel()
    batch = tuned.to_device(encode(tuned.processor, images[4].path, images[4].caption))
    with torch.no_grad():
        base_logits = base.model(**batch).logits
        with tuned.strength(0.0):
            off = tuned.model(**batch).logits
        with tuned.strength(1.0):
            on = tuned.model(**batch).logits
    assert torch.allclose(off, base_logits, atol=1e-4)
    assert not torch.allclose(on, base_logits, atol=1e-4)


def test_strength_moves_gradually_between_base_and_tuned(images, tmp_path):
    path = train(TrainConfig(examples=8, r=4, alpha=4, learning_rate=1e-3), images[:4], images[4:], "s", tmp_path)
    model = CaptionModel(load_run(path).adapter_dir)
    batch = model.to_device(encode(model.processor, images[4].path, images[4].caption))

    def logits_at(strength):
        with torch.no_grad(), model.strength(strength):
            return model.model(**batch).logits

    off, half, on = logits_at(0.0), logits_at(0.5), logits_at(1.0)
    assert (half - off).abs().mean() > 0
    assert (half - off).abs().mean() < (on - off).abs().mean()
    assert torch.allclose(logits_at(1.0), on)  # returning to 1 gives the same answer
