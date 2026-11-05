"""The training loop. A small teaching copy of what the notebook's SFTTrainer does, written out so
every step can be recorded, and run in a background thread so the app stays usable."""

import math
import random
import resource
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from adapter_lab.data import Example
from adapter_lab.runs import RUNS_DIR, folder_bytes, save_run
from adapter_lab.vlm import (
    CaptionModel,
    adapter_norm,
    encode,
    frozen_checksums,
    labels_for,
    split_loss,
    trainable_by_region,
)

# Fixed at the notebook's values.
DROPOUT = 0.05
GRADIENT_ACCUMULATION = 4
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
BATCH_SIZE = 1

GALLERY_SIZE = 12
HELD_OUT_SIZE = 30


@dataclass(frozen=True)
class TrainConfig:
    examples: int = 60
    r: int = 16
    alpha: int = 16
    learning_rate: float = 2e-4
    mask_prompt: bool = False
    modules_to_save: bool = False
    seed: int = 0


REFERENCE_CONFIG = TrainConfig(examples=100)


def learning_rate_at(step: int, total_steps: int, peak: float, warmup_ratio: float = WARMUP_RATIO) -> float:
    """Linear warmup, then constant, like the notebook's `lr_scheduler_type="constant"` with warmup."""
    warmup = math.ceil(total_steps * warmup_ratio)
    return peak * min(1.0, (step + 1) / warmup) if warmup else peak


def sample_examples(train: list[Example], count: int, seed: int) -> list[Example]:
    """With replacement, like the notebook's `random.choices`, so an image can appear more than once."""
    return random.Random(seed).choices(train, k=count)


def peak_memory_gb() -> float:
    # ru_maxrss is in bytes on macOS and kilobytes on Linux.
    import sys

    divisor = 1e9 if sys.platform == "darwin" else 1e6
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def evaluate(model: CaptionModel, test: list[Example], on_stage: Callable[[str], None]) -> dict:
    """Held-out loss on caption tokens with the adapter off and on, and a gallery of captions."""
    model.model.eval()
    totals = {"base": [0.0, 0], "tuned": [0.0, 0]}
    for i, example in enumerate(test[:HELD_OUT_SIZE]):
        on_stage(f"Measuring held-out loss ({i + 1}/{min(HELD_OUT_SIZE, len(test))})")
        for key, strength in (("base", 0.0), ("tuned", 1.0)):
            loss_sum, tokens = model.caption_loss(example.path, example.caption, strength)
            totals[key][0] += loss_sum
            totals[key][1] += tokens

    gallery = []
    for i, example in enumerate(test[:GALLERY_SIZE]):
        on_stage(f"Captioning test images ({i + 1}/{min(GALLERY_SIZE, len(test))})")
        gallery.append({
            "image_id": example.image_id,
            "path": example.path,
            "real": example.caption,
            "base": model.caption(example.path, strength=0.0)[0],
            "tuned": model.caption(example.path, strength=1.0)[0],
        })
    return {
        "held_out_base": totals["base"][0] / max(totals["base"][1], 1),
        "held_out_tuned": totals["tuned"][0] / max(totals["tuned"][1], 1),
        "held_out_images": min(HELD_OUT_SIZE, len(test)),
        "gallery": gallery,
    }


def train(
    config: TrainConfig,
    train_examples: list[Example],
    test_examples: list[Example],
    name: str,
    runs_dir: Path = RUNS_DIR,
    on_stage: Callable[[str], None] = lambda stage: None,
    on_step: Callable[[dict], None] = lambda row: None,
    should_stop: Callable[[], bool] = lambda: False,
) -> Path:
    """Train an adapter, evaluate it and save the run. Returns the run's folder."""
    torch.manual_seed(config.seed)

    on_stage("Loading the base model")
    model = CaptionModel()
    before = frozen_checksums(model.model)

    on_stage("Attaching LoRA")
    model.add_lora(config.r, config.alpha, config.modules_to_save)
    peft_model = model.model
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in peft_model.parameters())
    regions = trainable_by_region(peft_model)

    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.config.use_cache = False
    peft_model.train()
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)

    chosen = sample_examples(train_examples, config.examples, config.seed)
    total_steps = math.ceil(config.examples / GRADIENT_ACCUMULATION)
    history: list[dict] = []
    started = time.time()
    optimizer_step = 0
    stopped = False

    for index, example in enumerate(chosen, start=1):
        if should_stop():
            stopped = True
            break
        on_stage(f"Training on example {index}/{config.examples}")
        step_started = time.time()
        encoded = encode(model.processor, example.path, example.caption)
        labels = labels_for(model.processor, encoded, config.mask_prompt)
        batch = model.to_device(encoded)
        logits = peft_model(**batch).logits
        parts = split_loss(logits, batch["input_ids"], labels, encoded.prompt_length)
        (parts.loss / GRADIENT_ACCUMULATION).backward()

        if index % GRADIENT_ACCUMULATION == 0 or index == config.examples:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(optimizer_step, total_steps, config.learning_rate)
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

        row = {
            "example": index,
            "optimizer_step": optimizer_step,
            "image_id": example.image_id,
            "loss": parts.loss.item(),
            "prompt_loss": parts.prompt,
            "caption_loss": parts.caption,
            "adapter_norm": adapter_norm(peft_model),
            "seconds": time.time() - step_started,
            "peak_memory_gb": peak_memory_gb(),
        }
        history.append(row)
        on_step(row)
    training_seconds = time.time() - started

    peft_model.gradient_checkpointing_disable()
    peft_model.config.use_cache = True
    results = evaluate(model, test_examples, on_stage)

    on_stage("Saving the adapter")
    after = frozen_checksums(peft_model)
    changed = [k for k in before if k in after and before[k] != after[k]]
    run_path = runs_dir / name
    peft_model.save_pretrained(run_path / "adapter")
    results.update({
        "status": "stopped early" if stopped else "finished",
        "examples_trained": len(history),
        "unique_images": len({row["image_id"] for row in history}),
        "optimizer_steps": optimizer_step,
        "training_seconds": training_seconds,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "trainable_by_region": regions,
        "frozen_weights_checked": sum(k in after for k in before),
        "frozen_weights_changed": changed,
        "adapter_bytes": folder_bytes(run_path / "adapter"),
        "peak_memory_gb": peak_memory_gb(),
        "device": model.device,
    })
    return save_run(name, asdict(config), history, results, runs_dir)


class TrainingJob:
    """One training run in a background thread. The app polls its fields; only the thread writes them."""

    def __init__(self, config: TrainConfig, train_examples, test_examples, name: str, runs_dir: Path = RUNS_DIR):
        self.config = config
        self.name = name
        self.stage = "Starting"
        self.history: list[dict] = []
        self.error: str | None = None
        self.done = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(train_examples, test_examples, runs_dir), daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def running(self) -> bool:
        return not self.done

    def _run(self, train_examples, test_examples, runs_dir: Path) -> None:
        try:
            train(
                self.config, train_examples, test_examples, self.name, runs_dir,
                on_stage=lambda stage: setattr(self, "stage", stage),
                on_step=self.history.append,
                should_stop=self._stop.is_set,
            )
            self.stage = "Finished"
        except Exception as error:  # shown in the app; a crashed thread would otherwise vanish silently
            self.error = f"{type(error).__name__}: {error}"
            self.stage = "Failed"
        finally:
            self.done = True
