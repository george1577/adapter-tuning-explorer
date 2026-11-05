"""The sample of the ROCO radiology dataset: 100 training and 100 test images."""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATASET = "eltorio/ROCO-radiology"
ROWS_URL = "https://datasets-server.huggingface.co/first-rows?dataset={dataset}&config=default&split={split}"
SPLITS = ("train", "test")

# Images the reference notebooks display, so their recorded captions can be shown next to them.
NOTEBOOK_TRAIN_IMAGE = "ROCO_00003"
NOTEBOOK_TEST_IMAGE = "ROCO_00006"


@dataclass(frozen=True)
class Example:
    image_id: str
    path: str
    caption: str


def is_downloaded(data_dir: Path = DATA_DIR) -> bool:
    return all((data_dir / f"{split}.jsonl").exists() for split in SPLITS)


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[Example]:
    lines = (data_dir / f"{split}.jsonl").read_text().splitlines()
    return [Example(**json.loads(line)) for line in lines]


def parse_rows(rows: list[dict], image_dir: Path) -> list[tuple[Example, str]]:
    """Turn dataset-server rows into (example, image URL) pairs, dropping rows with an empty caption."""
    parsed = []
    for row in rows:
        row = row["row"]
        caption = (row.get("caption") or "").strip()
        if caption:
            path = image_dir / f"{row['image_id']}.jpg"
            parsed.append((Example(row["image_id"], str(path), caption), row["image"]["src"]))
    return parsed


def download_sample(data_dir: Path = DATA_DIR, progress: Callable[[str], None] = print) -> None:
    """Download the first rows of each split. Safe to run again: it replaces the sample."""
    for split in SPLITS:
        image_dir = data_dir / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ROWS_URL.format(dataset=DATASET, split=split)) as response:
            rows = json.load(response)["rows"]
        examples = []
        for example, url in parse_rows(rows, image_dir):
            urllib.request.urlretrieve(url, example.path)
            examples.append(example)
        with open(data_dir / f"{split}.jsonl", "w") as f:
            for e in examples:
                f.write(json.dumps({"image_id": e.image_id, "path": e.path, "caption": e.caption}) + "\n")
        progress(f"{split}: {len(examples)} images")
