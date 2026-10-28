"""pytest entry point:   pytest -v   (run `python -m src.train` first)"""
import pytest

from tests.cases import CASES
from src import config as C

NEEDS_MODEL = {"T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T19"}


@pytest.mark.parametrize("case", CASES, ids=[c.id + "-" + c.area for c in CASES])
def test_case(case):
    if case.id in NEEDS_MODEL and not C.MODEL_PATH.exists():
        pytest.skip("Trained model missing - run: python -m src.train")
    outcome = case.observe()
    assert case.check(outcome), (
        f"{case.id} {case.description}\nExpected: {case.expected}\n"
        f"Observed: {case.show(outcome)}")
