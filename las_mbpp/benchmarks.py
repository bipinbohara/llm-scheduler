"""Adapters for official function-synthesis benchmarks."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from las_mbpp.telemetry import observe, span

DATASETS = ("mbpp", "mbpp-plus", "humaneval", "humaneval-plus")
PLUS_NAMES = {"mbpp-plus": "mbpp", "humaneval-plus": "humaneval"}


MBPP_RELEASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "3f540f420bfae7e6ce48365a68d0ae003feb2f87/mbpp/sanitized-mbpp.json"
)
MBPP_RELEASE_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
MBPP_SPLITS = {"prompt": (1, 10), "test": (11, 510), "validation": (511, 600), "train": (601, 974)}


def read_mbpp_release(path: Path) -> dict:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != MBPP_RELEASE_SHA256:
        raise ValueError(f"{path} is not the pinned Google Research sanitized MBPP release.")
    rows = json.loads(data)
    return {
        split: [row for row in rows if interval[0] <= row["task_id"] <= interval[1]]
        for split, interval in MBPP_SPLITS.items()
    }


def sanitized_mbpp_records(split, hub_loader):
    local = os.environ.get("MBPP_SANITIZED_JSON")
    if local:
        try:
            return read_mbpp_release(Path(local))[split]
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load sanitized MBPP split {split!r} from MBPP_SANITIZED_JSON. "
                "Prepare the pinned copy with python -m las_mbpp.cli.prepare_data mbpp --mbpp-json PATH."
            ) from exc
    return hub_loader("google-research-datasets/mbpp", "sanitized", split=split)


def add_dataset_argument(parser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="mbpp")


def load_official_problems(dataset):
    try:
        from evalplus import data
        from evalplus.data.humaneval import get_human_eval
    except ImportError as exc:
        raise RuntimeError('Install the benchmark dependency with: pip install -e ".[evalplus]"') from exc

    loaders = {
        "humaneval": (get_human_eval, None),
        "humaneval-plus": (data.get_human_eval_plus, data.get_human_eval_plus_hash),
        "mbpp-plus": (data.get_mbpp_plus, data.get_mbpp_plus_hash),
    }
    loader, fingerprint = loaders[dataset]
    return loader(), fingerprint() if fingerprint else None


def load_tasks(dataset, split, *, limit, task_ids, native_loader):
    if dataset == "mbpp":
        try:
            return native_loader(split=split, limit=limit, task_ids=task_ids)
        except (OSError, ValueError, ConnectionError) as exc:
            raise RuntimeError(f"Cannot load sanitized MBPP split {split!r}. Check --splits or prepare a local copy with python -m las_mbpp.cli.prepare_data mbpp --mbpp-json PATH.") from exc
    if split != "test":
        raise ValueError(f"{dataset} has only an official test split. Use --splits test.")

    try:
        problems, release_hash = load_official_problems(dataset)
    except (OSError, ValueError, ConnectionError) as exc:
        raise RuntimeError(f"Could not prepare {dataset}. Run python -m las_mbpp.cli.prepare_data {dataset} with access to the official release.") from exc

    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    problems = sorted(problems.values(), key=lambda p: int(p["task_id"].split("/")[-1]))
    problems = [p for p in problems if task_ids is None or int(p["task_id"].split("/")[-1]) in task_ids]
    if limit is not None:
        problems = problems[:max(0, limit)]

    tasks = []
    for problem in problems:
        benchmark_id = problem["task_id"]
        entry = problem["entry_point"]
        prompt = problem["prompt"]
        reference = prompt + problem["canonical_solution"]
        task = {
            "split": split,
            "task_id": int(benchmark_id.split("/")[-1]),
            "benchmark_id": benchmark_id,
            "benchmark_name": "MBPP+" if dataset == "mbpp-plus" else "HumanEval",
            "description": prompt.strip(),
            "prompt_stub": prompt,
            "function_name": entry,
            "setup_code": "",
            "tests": [],
            "reference_code": reference,
        }
        if dataset == "humaneval":
            task["evaluation_tests"] = [problem["test"] + f"\ncheck({entry})"]
        else:
            suites = {}
            preparation_start = time.perf_counter()
            for field in ("base", "plus"):
                inputs = problem[f"{field}_input"]
                expected, timings = trusted_exec(
                    reference, deepcopy(inputs), entry, record_time=True,
                    output_not_none=dataset == "mbpp-plus" and entry in MBPP_OUTPUT_NOT_NONE_TASKS,
                )
                suites[field] = (inputs, expected, timings)
            task.update(
                evalplus=PLUS_NAMES[dataset], evalplus_version=version("evalplus"),
                suites=suites, release_hash=release_hash, atol=problem["atol"],
                reference_preparation_seconds=time.perf_counter() - preparation_start,
            )
        tasks.append(task)
    return tasks


@observe("grader.evalplus_suite", "grading")
def evaluate_plus_suite(task: dict, code: str, suite: str) -> dict:
    from evalplus.eval import PASS, untrusted_check

    inputs, expected, timings = task["suites"][suite]
    start = time.perf_counter()
    error = ""
    try:
        ast.parse(code)
    except SyntaxError as exc:
        error = f"Syntax error: {exc}"
    if not code.strip():
        error = "Empty candidate."
    if error:
        status, outcomes = "invalid", []
    else:
        with span("grader.evalplus_execute", "grading", suite=suite, inputs=len(inputs)):
            status, outcomes = untrusted_check(
                task["evalplus"], code, inputs, task["function_name"], expected=expected,
                atol=task["atol"], ref_time=timings, fast_check=False,
            )
    passed_count = sum(bool(value) for value in outcomes)
    feedback = f"Official EvalPlus {suite} suite: {status} ({passed_count}/{len(inputs)})."
    return {
        "code": code,
        "pass_count": passed_count,
        "total_count": len(inputs),
        "passed_all": status == PASS,
        "exec_status": feedback,
        "exec_error": error or (status if status not in {PASS, "fail"} else ""),
        "failures": [],
        "feedback_text": error or feedback,
        "eval_latency_sec": time.perf_counter() - start,
    }


@observe("evaluation.final_scoring", "grading")
def final_benchmark_score(task: dict, code: str, base: dict) -> dict:
    if "evalplus" not in task:
        return {}
    plus = evaluate_plus_suite(task, code, "plus")
    return {
        "evalplus_base_correct": bool(base["passed_all"]),
        "evalplus_plus_correct": bool(base["passed_all"] and plus["passed_all"]),
        "plus_tests_passed": plus["pass_count"],
        "plus_tests_total": plus["total_count"],
        "plus_exec_status": plus["exec_status"],
        "plus_latency_sec": plus["eval_latency_sec"],
    }
