"""Which tokens count towards the loss. Works on plain token ids so it can be tested without a model."""

from dataclasses import dataclass

IGNORE = -100  # Hugging Face's convention: a label of -100 is left out of the loss


@dataclass(frozen=True)
class TokenRow:
    position: int
    token_id: int
    token: str
    region: str  # "image", "prompt", "caption" or "padding"
    label: int


def build_labels(
    input_ids: list[int],
    image_ids: set[int],
    pad_id: int,
    prompt_length: int,
    mask_prompt: bool = False,
) -> list[int]:
    """Copy the input ids, then blank out what shouldn't be predicted.

    The notebook's collator only blanks padding and image tokens, so the model is also trained to
    predict the prompt. `mask_prompt=True` blanks the prompt too, leaving only the caption.
    """
    labels = []
    for position, token_id in enumerate(input_ids):
        masked = token_id == pad_id or token_id in image_ids or (mask_prompt and position < prompt_length)
        labels.append(IGNORE if masked else token_id)
    return labels


def region_of(position: int, token_id: int, image_ids: set[int], pad_id: int, prompt_length: int) -> str:
    if token_id == pad_id:
        return "padding"
    if token_id in image_ids:
        return "image"
    return "prompt" if position < prompt_length else "caption"


def token_rows(
    input_ids: list[int],
    tokens: list[str],
    labels: list[int],
    image_ids: set[int],
    pad_id: int,
    prompt_length: int,
) -> list[TokenRow]:
    return [
        TokenRow(i, token_id, tokens[i], region_of(i, token_id, image_ids, pad_id, prompt_length), labels[i])
        for i, token_id in enumerate(input_ids)
    ]
