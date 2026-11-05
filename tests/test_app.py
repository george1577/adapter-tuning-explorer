"""Renders the whole app without a browser, the way Streamlit's own test harness does."""

import pytest
from streamlit.testing.v1 import AppTest

from adapter_lab.data import is_downloaded


@pytest.mark.slow
@pytest.mark.skipif(not is_downloaded(), reason="needs the data sample")
def test_every_tab_renders_without_an_exception():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs][:5] == [
        "① Why tune, and the data", "② Inside LoRA", "③ QLoRA and memory", "④ Train an adapter", "⑤ Test the adapter",
    ]
