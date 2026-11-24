from __future__ import annotations

from las_mbpp.prompts import build_error_correction_prompt
from las_mbpp.workflows import mbpp_baseline


def test_strip_to_code_extracts_and_renames_function() -> None:
    raw = """Here is code:

```python
def wrong_name(x):
    return x + 1
```
"""
    assert mbpp_baseline.strip_to_code(raw, expected_name="plus_one").startswith("def plus_one")


def test_evaluate_code_passes_simple_task() -> None:
    task = {
        "function_name": "plus_one",
        "setup_code": "",
        "tests": ["assert plus_one(1) == 2", "assert plus_one(-1) == 0"],
    }
    result = mbpp_baseline.evaluate_code(task, "def plus_one(x):\n    return x + 1\n", timeout_sec=5)
    assert result["passed_all"] is True
    assert result["pass_count"] == 2


def test_error_correction_prompt_includes_tests() -> None:
    prompt = build_error_correction_prompt(
        task_description="Return x plus one.",
        function_stub="def plus_one(x):\n    pass\n",
        function_name_from_tests="plus_one",
        verified_code="def plus_one(x):\n    return x\n",
        test_feedback="Tests passed: 0/1",
        tests=["assert plus_one(1) == 2"],
    )
    assert "assert plus_one(1) == 2" in prompt
