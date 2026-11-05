from datetime import datetime

import pytest

from adapter_lab.data import Example, parse_rows
from adapter_lab.messages import format_message
from adapter_lab.runs import list_runs, load_run, new_run_name, save_run
from adapter_lab.training import learning_rate_at, sample_examples


def test_warmup_rises_to_the_peak_then_stays():
    # 125 steps at 3% warmup is 4 steps, as in the notebook.
    assert learning_rate_at(0, 125, 1.0) == pytest.approx(0.25)
    assert learning_rate_at(3, 125, 1.0) == 1.0
    assert learning_rate_at(100, 125, 1.0) == 1.0


def test_sampling_is_repeatable_and_with_replacement():
    pool = [Example(str(i), f"{i}.jpg", "c") for i in range(10)]
    first = sample_examples(pool, 50, seed=0)
    assert first == sample_examples(pool, 50, seed=0)
    assert len({e.image_id for e in first}) < 50


def test_format_message_matches_the_notebook_shape():
    with_caption = format_message("a.jpg", " A caption.\n")["messages"]
    assert [m["role"] for m in with_caption] == ["system", "user", "assistant"]
    assert with_caption[2]["content"][0]["text"] == "A caption."
    assert [m["role"] for m in format_message("a.jpg")["messages"]] == ["system", "user"]


def test_rows_with_empty_captions_are_dropped(tmp_path):
    rows = [
        {"row": {"image_id": "A", "caption": " real\n", "image": {"src": "u"}}},
        {"row": {"image_id": "B", "caption": "  ", "image": {"src": "v"}}},
    ]
    parsed = parse_rows(rows, tmp_path)
    assert [(e.image_id, e.caption, url) for e, url in parsed] == [("A", "real", "u")]


def test_runs_round_trip_and_list_only_finished_ones(tmp_path):
    save_run("reference", {"r": 16}, [{"loss": 1.0}], {"status": "finished"}, tmp_path)
    save_run("run-1", {"r": 8}, [], {"status": "stopped early"}, tmp_path)
    (tmp_path / "unfinished").mkdir()
    runs = list_runs(tmp_path)
    assert [r.name for r in runs][0] == "reference"
    assert {r.name for r in runs} == {"reference", "run-1"}
    assert load_run(tmp_path / "reference").config == {"r": 16}


def test_run_names_sort_by_time():
    assert new_run_name(datetime(2026, 1, 2, 3, 4, 5)) == "run-20260102-030405"
