"""The chat messages used for training and captioning, copied from the reference notebooks."""

SYSTEM_PROMPT = "You are an expert researcher in radiology."
USER_PROMPT = (
    "Write a caption for this image explaining what it depicts, "
    "suitable for use in a research publication."
)


def format_message(image_path: str, caption: str | None = None) -> dict:
    """One example in the notebook's format. Without a caption it is a prompt for inference."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image", "image": str(image_path)},
            ],
        },
    ]
    if caption is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": caption.strip()}]})
    return {"messages": messages}
