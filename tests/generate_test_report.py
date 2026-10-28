"""Run every test case and write docs/TEST_REPORT.md with the EXPECTED result
(written in advance) next to the OBSERVED result (measured now).

    python -m tests.generate_test_report
"""
from datetime import datetime, timezone

from src.config import PROJECT_ROOT
from tests.cases import CASES


def main():
    rows, passed = [], 0
    for c in CASES:
        outcome = c.observe()
        ok = c.check(outcome)
        passed += ok
        esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {c.id} | {c.area} | {esc(c.description)} | {esc(c.expected)} "
                    f"| {esc(c.show(outcome))} | {'PASS' if ok else '**FAIL**'} |")
    out = [
        "# Test Report",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        "by `python -m tests.generate_test_report`\n",
        f"**Result: {passed} / {len(CASES)} test cases passed.**\n",
        "| ID | Area | Test | Expected result | Observed result | Status |",
        "|----|------|------|-----------------|-----------------|--------|",
        *rows,
        "\n> Note: for hand-written sample text (T07-T09) the test checks that the "
        "system behaves correctly (valid output, warnings), NOT that the predicted "
        "label is 'right' - there is no ground truth for text I wrote myself. "
        "Correctness of predictions is measured on the held-out dataset split "
        "(see T19 and reports/metrics.json).",
    ]
    path = PROJECT_ROOT / "docs" / "TEST_REPORT.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"{passed}/{len(CASES)} passed -> {path}")


if __name__ == "__main__":
    main()
