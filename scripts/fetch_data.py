"""Download the radiology image sample into data/. Run with: uv run python scripts/fetch_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter_lab.data import download_sample

if __name__ == "__main__":
    download_sample()
