"""Headless test of the Streamlit app's manual-primer scoring tab.

Uses Streamlit's AppTest (no browser). Skipped when streamlit is not installed.
Requires the head-A model artifacts to exist (run predictor.pipeline.head_a).
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "copilot" / "app" / "main.py")
MODELS = Path("data/models")

pytestmark = pytest.mark.skipif(
    not (MODELS / "head_a_classifier.joblib").exists(),
    reason="head-A model artifacts not built",
)

# A short template with a known perfect-match primer substring at offset 4.
TEMPLATE = "GGGGATTGACTACGACGCGCTCATTTACGTACGTACGT"
PERFECT = "ATTGACTACGACGCGCTCAT"
MISMATCH = PERFECT[:-4] + "AAAA"


def test_manual_scoring_tab_scores_primers():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    at.text_area(key="template").set_value(TEMPLATE)
    at.text_area(key="primers").set_value(f"{PERFECT}\n{MISMATCH}")
    at.button(key="score_btn").click().run()
    assert not at.exception

    # A results dataframe should have been rendered with two scored primers.
    assert len(at.dataframe) >= 1
    data = at.dataframe[0].value
    primers = [str(x) for x in data["primer"]]
    assert PERFECT in primers and MISMATCH in primers


def test_manual_load_example_button_fills_template():
    from copilot.app.main import _DEMO_MANUAL_PRIMERS, _DEMO_TEMPLATE

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="load_manual_example").click().run()
    assert not at.exception
    assert at.text_area(key="template").value == _DEMO_TEMPLATE
    assert at.text_area(key="primers").value == _DEMO_MANUAL_PRIMERS


@pytest.mark.skipif(
    shutil.which("blastn") is None or not Path("data/blast/ighv_db.nin").exists(),
    reason="BLAST+ or IGHV demo DB not available",
)
def test_specificity_demo_tab_runs():
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.button(key="spec_demo").click().run()
    assert not at.exception
    assert len(at.metric) >= 1  # off-target metrics rendered
