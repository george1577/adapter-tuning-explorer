"""The only module that touches the real model. Everything else works on plain ids and numbers."""

import hashlib
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from adapter_lab.labels import IGNORE, build_labels
from adapter_lab.messages import format_message

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_EDGE = 512  # one 512 x 512 view of each image, which keeps a training step at a few seconds
MAX_NEW_TOKENS = 80


def pick_device() -> str:
    # MPS crashes during LoRA training on macOS 14 (exit 134), so it is opt-in: ATE_DEVICE=mps or cuda.
    return os.environ.get("ATE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def load_processor():
    return AutoProcessor.from_pretrained(MODEL_ID, size={"longest_edge": IMAGE_EDGE}, do_image_splitting=False)


def image_token_ids(processor) -> set[int]:
    """Every token that stands for the image: the slots themselves and the markers around them."""
    tokenizer = processor.tokenizer
    return {tokenizer.convert_tokens_to_ids(t) for t in ("<image>", "<fake_token_around_image>", "<global-img>")}


def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class Encoded:
    """One example ready for the model, plus where its caption starts."""
    batch: dict
    input_ids: list[int]
    prompt_length: int


def encode(processor, path: str, caption: str | None, image: Image.Image | None = None) -> Encoded:
    """Apply the chat template and the image processor. Without a caption, ends ready to generate."""
    image = image or read_image(path)
    prompt_messages = format_message(path)["messages"]
    prompt_text = processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
    prompt = processor(text=[prompt_text], images=[[image]], return_tensors="pt")
    prompt_length = prompt["input_ids"].shape[1]
    if caption is None:
        return Encoded(dict(prompt), prompt["input_ids"][0].tolist(), prompt_length)

    text = processor.apply_chat_template(format_message(path, caption)["messages"]).strip()
    batch = processor(text=[text], images=[[image]], return_tensors="pt")
    return Encoded(dict(batch), batch["input_ids"][0].tolist(), prompt_length)


def labels_for(processor, encoded: Encoded, mask_prompt: bool) -> list[int]:
    return build_labels(
        encoded.input_ids, image_token_ids(processor), processor.tokenizer.pad_token_id,
        encoded.prompt_length, mask_prompt,
    )


def token_losses(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for each token, predicted from the tokens before it. Position 0 has no prediction."""
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1), reduction="none"
    )


@dataclass
class LossParts:
    loss: torch.Tensor  # what training minimises: the mean over all labelled tokens
    prompt: float | None  # mean over labelled tokens in the prompt
    caption: float  # mean over caption tokens


def split_loss(logits: torch.Tensor, input_ids: torch.Tensor, labels: list[int], prompt_length: int) -> LossParts:
    """Separate prompt from caption, so a falling loss can be traced to what the model actually learned."""
    per_token = token_losses(logits, input_ids)
    targets = torch.tensor(labels[1:], device=per_token.device)
    labelled = targets != IGNORE
    in_caption = torch.arange(1, len(labels), device=per_token.device) >= prompt_length
    prompt_part = per_token[labelled & ~in_caption]
    return LossParts(
        loss=per_token[labelled].mean(),
        prompt=prompt_part.mean().item() if prompt_part.numel() else None,
        caption=per_token[in_caption].mean().item(),
    )


def make_lora_config(r: int, alpha: int, modules_to_save: bool) -> LoraConfig:
    """The notebook's LoraConfig, with its modules_to_save made optional."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] if modules_to_save else None,
    )


def _base_name(name: str) -> str:
    """A parameter's name in the original model, however PEFT has wrapped it."""
    return name.replace("base_model.model.", "").replace(".base_layer", "").replace(".original_module", "")


def frozen_checksums(model) -> dict[str, str]:
    """A fingerprint of every original weight. Equal before and after training proves they never changed."""
    sums = {}
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            continue
        data = param.detach().cpu().numpy().tobytes()
        sums[_base_name(name)] = hashlib.blake2b(data, digest_size=16).hexdigest()
    return sums


def region_of_parameter(name: str) -> str:
    if "vision_model" in name:
        return "Vision encoder"
    if "connector" in name:
        return "Image-to-text connector"
    if "lm_head" in name:
        return "Output layer (lm_head)"
    if "embed_tokens" in name:
        return "Word embeddings (embed_tokens)"
    return "Text decoder"


def trainable_by_region(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            region = region_of_parameter(name)
            counts[region] = counts.get(region, 0) + param.numel()
    return counts


def adapter_norm(model) -> float:
    """Size of the B matrices, which start at zero. It grows as the adapter learns."""
    total = 0.0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            total += param.detach().pow(2).sum().item()
    return total**0.5


class CaptionModel:
    """SmolVLM, optionally with an adapter on top."""

    def __init__(self, adapter_dir: Path | None = None):
        self.device = pick_device()
        self.processor = load_processor()
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).to(self.device)
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir)).to(self.device)
        self.model.eval()

    @property
    def has_adapter(self) -> bool:
        return isinstance(self.model, PeftModel)

    def add_lora(self, r: int, alpha: int, modules_to_save: bool) -> None:
        self.model = get_peft_model(self.model, make_lora_config(r, alpha, modules_to_save))
        self.model.enable_input_require_grads()

    def to_device(self, encoded: Encoded) -> dict:
        return {k: v.to(self.device) for k, v in encoded.batch.items()}

    def strength(self, value: float):
        """A context in which the adapter applies at `value` x its trained strength. 0 is the base model."""
        if not self.has_adapter:
            return nullcontext()
        if value == 0:
            return self.model.disable_adapter()
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                for name in module.scaling:
                    module.set_scale(name, value)
        return nullcontext()

    @torch.no_grad()
    def caption(self, path: str, strength: float = 1.0, image: Image.Image | None = None) -> tuple[str, list[int]]:
        """Deterministic (greedy) caption, and the ids of the tokens it wrote."""
        encoded = encode(self.processor, path, None, image)
        batch = self.to_device(encoded)
        with self.strength(strength):
            output = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        ids = output[0, encoded.prompt_length :].tolist()
        return self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip(), ids

    @torch.no_grad()
    def next_token_table(self, path: str, written: list[int], strength: float, top: int = 8) -> list[tuple[str, float]]:
        """The most likely next tokens after the prompt and the tokens already written."""
        encoded = encode(self.processor, path, None)
        batch = self.to_device(encoded)
        prefix = torch.tensor([written], dtype=torch.long, device=self.device)
        batch["input_ids"] = torch.cat([batch["input_ids"], prefix], dim=1)
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        with self.strength(strength):
            logits = self.model(**batch).logits[0, -1]
        probs = logits.softmax(dim=-1)
        values, ids = probs.topk(top)
        return [(self.token_text(i), v) for i, v in zip(ids.tolist(), values.tolist())]

    @torch.no_grad()
    def caption_loss(self, path: str, caption: str, strength: float) -> tuple[float, float]:
        """Sum of caption-token losses and how many tokens were scored, for a held-out example."""
        encoded = encode(self.processor, path, caption)
        batch = self.to_device(encoded)
        with self.strength(strength):
            logits = self.model(**batch).logits
        per_token = token_losses(logits, batch["input_ids"])
        caption_part = per_token[encoded.prompt_length - 1 :]
        return caption_part.sum().item(), caption_part.numel()

    def token_text(self, token_id: int) -> str:
        text = self.processor.tokenizer.decode([token_id])
        return text.replace(" ", "␣").replace("\n", "⏎")


def weight_file() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_ID, "model.safetensors")


def linear_layers() -> list[dict]:
    """Every linear layer in SmolVLM, without loading its weights. lm_head is the one LoRA skips."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    return [
        {"name": name, "d_in": m.in_features, "d_out": m.out_features}
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name
    ]


def read_weights(key_pattern: str = r"\.weight$", min_values: int = 4096) -> dict[str, tuple[int, ...]]:
    """Names and shapes of the 2-D weights in the checkpoint that are big enough to study."""
    shapes = {}
    with safe_open(weight_file(), framework="pt") as f:
        for key in f.keys():
            if re.search(key_pattern, key):
                shape = tuple(f.get_slice(key).get_shape())
                if len(shape) == 2 and shape[0] * shape[1] >= min_values:
                    shapes[key] = shape
    return shapes


def read_weight(key: str) -> torch.Tensor:
    with safe_open(weight_file(), framework="pt") as f:
        return f.get_tensor(key).float()
