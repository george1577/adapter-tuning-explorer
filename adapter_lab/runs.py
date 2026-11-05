"""Finished training runs on disk: one folder per run, holding the adapter and what was measured."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from adapter_lab.data import ROOT

RUNS_DIR = ROOT / "runs"
REFERENCE_NAME = "reference"


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    config: dict
    history: list[dict]
    results: dict

    @property
    def adapter_dir(self) -> Path:
        return self.path / "adapter"

    @property
    def is_reference(self) -> bool:
        return self.name == REFERENCE_NAME


def new_run_name(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("run-%Y%m%d-%H%M%S")


def save_run(name: str, config: dict, history: list[dict], results: dict, runs_dir: Path = RUNS_DIR) -> Path:
    path = runs_dir / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2))
    (path / "history.json").write_text(json.dumps(history))
    (path / "results.json").write_text(json.dumps(results, indent=2))
    return path


def load_run(path: Path) -> Run:
    read = lambda filename: json.loads((path / filename).read_text())
    return Run(path.name, path, read("config.json"), read("history.json"), read("results.json"))


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Run]:
    """Finished runs, the reference run first and then newest first. Unfinished folders are skipped."""
    if not runs_dir.exists():
        return []
    finished = [p for p in runs_dir.iterdir() if (p / "results.json").exists()]
    finished.sort(key=lambda p: (p.name != REFERENCE_NAME, -p.stat().st_mtime))
    return [load_run(p) for p in finished]


def delete_run(name: str, runs_dir: Path = RUNS_DIR) -> None:
    shutil.rmtree(runs_dir / name)


def folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
