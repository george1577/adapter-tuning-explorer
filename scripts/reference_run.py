"""Train the reference run, the one the guide's numbers come from. Takes about 8 minutes on a laptop CPU.
Run with: uv run python scripts/reference_run.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter_lab.data import download_sample, is_downloaded, load_split
from adapter_lab.runs import REFERENCE_NAME, RUNS_DIR
from adapter_lab.training import REFERENCE_CONFIG, train

if __name__ == "__main__":
    if not is_downloaded():
        download_sample()
    if len(sys.argv) > 1 and sys.argv[1] == "--replace":
        pass
    elif (RUNS_DIR / REFERENCE_NAME / "results.json").exists():
        sys.exit("A reference run already exists. Pass --replace to overwrite it.")
    train(
        REFERENCE_CONFIG, load_split("train"), load_split("test"), REFERENCE_NAME,
        on_stage=lambda stage: print(stage, flush=True),
        on_step=lambda row: print(f"  loss {row['loss']:.3f}  caption {row['caption_loss']:.3f}", flush=True),
    )
