#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from las_mbpp.config import DEFAULT_RESULTS_DIR  # noqa: E402
from las_mbpp.prompts import (  # noqa: E402
    CODING_SYSTEM_PROMPT,
    CONSENSUS_SYSTEM_PROMPT,
    build_consensus_prompt,
    build_debate_initial_prompt,
    build_debate_revision_prompt,
    build_error_correction_prompt,
    build_reflection_prompt,
    build_verification_prompt,
)


DEFAULT_MODEL = "MiniMax-M2.7"
DEFAULT_SPLITS = ("test", "train")
DEFAULT_EVAL_TIMEOUT = 60.0
DEFAULT_TESTS_PREVIEW_LIMIT = 20
DEFAULT_MAX_TOKENS = 500000
DEFAULT_TEMPERATURE = 0.7

ASSERT_CALL_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")
DEF_RE = re.compile(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*:")
FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _slugify_name(name: str) -> str:
    normalized = SAFE_NAME_RE.sub("_", (name or "").strip())
    normalized = normalized.strip("._-")
    return normalized or "run"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _extract_json_dict(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        try:
            data = json.loads(text[first : last + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
    for match in re.finditer(r"\{[\s\S]*?\}", text):
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


def extract_function_name_from_code(reference_code: str) -> Optional[str]:
    if not reference_code:
        return None
    m = DEF_RE.search(reference_code)
    if m:
        return m.group(1)
    return None


def extract_function_name_from_tests(tests: List[str]) -> Optional[str]:
    names: List[str] = []
    for test in tests or []:
        regex_match = ASSERT_CALL_RE.search(test or "")
        if regex_match:
            names.append(regex_match.group(1))
            continue
        try:
            tree = ast.parse(test)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.append(node.func.id)
                break
    if not names:
        return None
    counts = Counter(names)
    return counts.most_common(1)[0][0]


def build_prompt_stub(description: str, reference_code: str, function_name: str) -> str:
    desc = (description or "No description provided.").replace('"""', '\\"""').strip()
    sig_match = DEF_RE.search(reference_code or "")
    if sig_match:
        args = sig_match.group(2)
        sig_line = f"def {function_name}({args}):"
    else:
        sig_line = f"def {function_name}(*args, **kwargs):"
    return f'{sig_line}\n    """{desc}"""\n    pass\n'


def strip_to_code(raw: str, expected_name: Optional[str] = None) -> str:
    text = (raw or "").strip()
    code = text
    for match in FENCE_RE.finditer(text):
        block = (match.group(1) or "").strip()
        if block:
            code = block
            break
    else:
        m = re.search(r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\(", text)
        if m:
            code = text[m.start() :]
    code = code.strip()
    if expected_name:
        m_name = re.search(r"(?m)^(\s*def\s+)([A-Za-z_]\w*)(\s*\()", code)
        if m_name:
            got_name = m_name.group(2)
            if got_name != expected_name:
                code = re.sub(
                    rf"(?m)^(\s*def\s+){re.escape(got_name)}(\s*\()",
                    rf"\1{expected_name}\2",
                    code,
                    count=1,
                )
                code = re.sub(rf"\b{re.escape(got_name)}\b", expected_name, code)
    return code.strip()


def _summarize_failures(failures: List[Dict[str, Any]], max_items: int = 5) -> str:
    if not failures:
        return "All tests passed."
    lines = []
    for item in failures[:max_items]:
        lines.append(
            f"Test #{item.get('index', 0) + 1}: {item.get('error_type', 'Error')} | "
            f"{item.get('error_message', '')} | {item.get('test_case', '')}"
        )
    if len(failures) > max_items:
        lines.append(f"... ({len(failures) - max_items} more failures omitted)")
    return "\n".join(lines)


def _evaluate_code_worker(code: str, setup_code: str, tests: List[str]) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    tests = tests or []
    total_count = len(tests)

    try:
        ast.parse(code)
    except SyntaxError as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "pass_count": 0,
            "total_count": total_count,
            "passed_all": False,
            "exec_status": "Compilation failed.",
            "exec_error": f"Syntax error: {exc}",
            "failures": [],
            "feedback_text": f"Syntax error: {exc}\n{tb}",
        }

    try:
        if setup_code:
            exec(setup_code, namespace)
        exec(code, namespace)
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return {
            "pass_count": 0,
            "total_count": total_count,
            "passed_all": False,
            "exec_status": "Runtime failure before tests.",
            "exec_error": f"Execution failed: {exc}",
            "failures": [],
            "feedback_text": f"Execution failed: {exc}\n{tb}",
        }

    pass_count = 0
    for idx, test in enumerate(tests):
        try:
            exec(test, namespace)
            pass_count += 1
        except AssertionError as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            failures.append(
                {
                    "index": idx,
                    "test_case": test,
                    "error_type": "AssertionError",
                    "error_message": str(exc),
                    "traceback": tb,
                }
            )
        except Exception as exc:  # noqa: BLE001
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            failures.append(
                {
                    "index": idx,
                    "test_case": test,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": tb,
                }
            )

    passed_all = pass_count == total_count and not failures
    if total_count == 0:
        exec_status = "No official tests provided."
        passed_all = True
    elif passed_all:
        exec_status = "All tests passed."
    else:
        exec_status = "Completed with failing tests."

    return {
        "pass_count": pass_count,
        "total_count": total_count,
        "passed_all": passed_all,
        "exec_status": exec_status,
        "exec_error": None,
        "failures": failures,
        "feedback_text": _summarize_failures(failures),
    }


def evaluate_code(task: Dict[str, Any], raw_completion: str, timeout_sec: float = DEFAULT_EVAL_TIMEOUT) -> Dict[str, Any]:
    expected_name = task.get("function_name")
    code = strip_to_code(raw_completion, expected_name=expected_name)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _evaluate_code_worker,
            code,
            task.get("setup_code", "") or "",
            list(task.get("tests", []) or []),
        )
        try:
            result = future.result(timeout=timeout_sec)
        except TimeoutError:
            result = {
                "pass_count": 0,
                "total_count": len(task.get("tests", []) or []),
                "passed_all": False,
                "exec_status": f"Evaluation timeout after {timeout_sec:.1f}s.",
                "exec_error": "Timeout",
                "failures": [],
                "feedback_text": "Evaluation timed out.",
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "pass_count": 0,
                "total_count": len(task.get("tests", []) or []),
                "passed_all": False,
                "exec_status": "Evaluation infrastructure error.",
                "exec_error": str(exc),
                "failures": [],
                "feedback_text": f"Evaluation infrastructure error: {exc}",
            }
    result["code"] = code
    result["eval_latency_sec"] = time.perf_counter() - t0
    return result


def build_test_feedback(eval_result: Dict[str, Any], max_failures: int = 5) -> str:
    if eval_result.get("passed_all"):
        return "All tests passed."
    lines = [f"Execution status: {eval_result.get('exec_status', '')}"]
    exec_error = eval_result.get("exec_error")
    if exec_error:
        lines.append(f"Execution error: {exec_error}")
    pass_count = int(eval_result.get("pass_count", 0) or 0)
    total_count = int(eval_result.get("total_count", 0) or 0)
    lines.append(f"Tests passed: {pass_count}/{total_count}")
    failures = eval_result.get("failures", []) or []
    if failures:
        lines.append("Failure summary (test code omitted):")
        for item in failures[:max_failures]:
            lines.append(
                f"- Test #{item.get('index', 0) + 1}: {item.get('error_type', '')}: "
                f"{item.get('error_message', '')}"
            )
        if len(failures) > max_failures:
            lines.append(f"... ({len(failures) - max_failures} more failures omitted)")
    else:
        lines.append(eval_result.get("feedback_text", "Unknown failure."))
    return "\n".join(lines)


class AnthropicChatClient:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    }
                ],
            )
            latency = time.perf_counter() - t0
            text_chunks: List[str] = []
            for block in getattr(message, "content", []) or []:
                if getattr(block, "type", "") == "text":
                    text_chunks.append(getattr(block, "text", "") or "")
            output_text = "\n".join(chunk for chunk in text_chunks if chunk).strip()

            usage_obj = getattr(message, "usage", None)
            input_tokens = 0
            output_tokens = 0
            if usage_obj is not None:
                if isinstance(usage_obj, dict):
                    input_tokens = int(usage_obj.get("input_tokens", 0) or 0)
                    output_tokens = int(usage_obj.get("output_tokens", 0) or 0)
                else:
                    input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
                    output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            if input_tokens == 0:
                input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
            if output_tokens == 0:
                output_tokens = _estimate_tokens(output_text)
            total_tokens = input_tokens + output_tokens
            return {
                "ok": True,
                "error": "",
                "output_text": output_text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
                "latency_sec": latency,
            }
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - t0
            input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "output_text": "",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "total_tokens": input_tokens,
                },
                "latency_sec": latency,
            }


def _parallel_calls(
    client: AnthropicChatClient,
    requests: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not requests:
        return results
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        future_to_agent = {
            pool.submit(
                client.call,
                system_prompt=req["system_prompt"],
                user_prompt=req["user_prompt"],
                model=req["model"],
            ): agent
            for agent, req in requests.items()
        }
        for fut in as_completed(future_to_agent):
            agent = future_to_agent[fut]
            req = requests[agent]
            try:
                results[agent] = fut.result()
            except Exception as exc:  # noqa: BLE001
                est_input = _estimate_tokens(req["system_prompt"]) + _estimate_tokens(req["user_prompt"])
                results[agent] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_text": "",
                    "usage": {
                        "input_tokens": est_input,
                        "output_tokens": 0,
                        "total_tokens": est_input,
                    },
                    "latency_sec": 0.0,
                }
    return results


def _parse_consensus_choice(raw_text: str, valid_agents: List[str]) -> Tuple[Optional[str], str]:
    data = _extract_json_dict(raw_text)
    reason = ""
    if isinstance(data, dict):
        reason = str(data.get("reason", "") or "").strip()
        candidate = data.get("selected_agent")
        if isinstance(candidate, str):
            m = re.search(r"[A-Za-z]", candidate)
            if m:
                letter = m.group(0).upper()
                if letter in valid_agents:
                    return letter, reason
        if isinstance(candidate, (int, float)):
            idx = int(candidate) - 1
            if 0 <= idx < len(valid_agents):
                return valid_agents[idx], reason
        for key in ("solution_letter", "choice", "selected", "answer"):
            val = data.get(key)
            if isinstance(val, str):
                m = re.search(r"[A-Za-z]", val)
                if m:
                    letter = m.group(0).upper()
                    if letter in valid_agents:
                        return letter, reason
    m = re.search(r"\b([A-C])\b", raw_text or "", flags=re.I)
    if m:
        letter = m.group(1).upper()
        if letter in valid_agents:
            return letter, reason
    return None, reason


def _make_stage_row(
    *,
    run_id: str,
    task: Dict[str, Any],
    stage_index: int,
    stage_name: str,
    agent_name: str,
    model_name: str,
    system_prompt: str,
    input_prompt: str,
    output_text: str,
    output_code: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_sec: float,
    api_error: str,
    eval_result: Dict[str, Any],
    selected_agent: str = "",
    selected_reason: str = "",
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": _utc_now_iso(),
        "split": task["split"],
        "task_id": task["task_id"],
        "function_name_from_tests": task["function_name"],
        "stage_index": stage_index,
        "stage_name": stage_name,
        "agent_name": agent_name,
        "model": model_name,
        "system_prompt": system_prompt,
        "agent_input": input_prompt,
        "agent_output": output_text,
        "output_code": output_code,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "latency_sec": float(latency_sec),
        "api_error": api_error,
        "tests_passed": int(eval_result.get("pass_count", 0) or 0),
        "tests_total": int(eval_result.get("total_count", 0) or 0),
        "is_correct": bool(eval_result.get("passed_all", False)),
        "exec_status": eval_result.get("exec_status", ""),
        "exec_error": eval_result.get("exec_error", ""),
        "eval_latency_sec": float(eval_result.get("eval_latency_sec", 0.0) or 0.0),
        "test_feedback": eval_result.get("feedback_text", ""),
        "failures_json": _safe_json_dumps(eval_result.get("failures", [])),
        "selected_agent": selected_agent,
        "selected_reason": selected_reason,
    }


def run_pipeline_for_task(
    *,
    run_id: str,
    task: Dict[str, Any],
    client: AnthropicChatClient,
    model: str,
    eval_timeout_sec: float,
    tests_preview_limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    rows: List[Dict[str, Any]] = []
    stage_index = 0
    debate_agents = ["A", "B", "C"]

    initial_requests: Dict[str, Dict[str, str]] = {}
    for agent in debate_agents:
        prompt = build_debate_initial_prompt(
            task_description=task["description"],
            function_stub=task["prompt_stub"],
            function_name_from_tests=task["function_name"],
            tests=task["tests"],
            agent_name=agent,
            tests_preview_limit=tests_preview_limit,
        )
        initial_requests[agent] = {
            "system_prompt": CODING_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "model": model,
        }

    initial_results = _parallel_calls(client, initial_requests)
    initial_outputs: Dict[str, Dict[str, Any]] = {}
    for agent in debate_agents:
        llm_res = initial_results[agent]
        eval_res = evaluate_code(task, llm_res["output_text"], timeout_sec=eval_timeout_sec)
        initial_outputs[agent] = {
            "code": eval_res["code"],
            "eval": eval_res,
            "llm": llm_res,
            "prompt": initial_requests[agent]["user_prompt"],
        }
        stage_index += 1
        rows.append(
            _make_stage_row(
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                stage_name="debate_initial",
                agent_name=f"debate_agent_{agent}",
                model_name=model,
                system_prompt=CODING_SYSTEM_PROMPT,
                input_prompt=initial_requests[agent]["user_prompt"],
                output_text=llm_res["output_text"],
                output_code=eval_res["code"],
                input_tokens=llm_res["usage"]["input_tokens"],
                output_tokens=llm_res["usage"]["output_tokens"],
                total_tokens=llm_res["usage"]["total_tokens"],
                latency_sec=llm_res["latency_sec"],
                api_error=llm_res["error"],
                eval_result=eval_res,
            )
        )

    final_debate_requests: Dict[str, Dict[str, str]] = {}
    for agent in debate_agents:
        peers = {a: initial_outputs[a]["code"] for a in debate_agents if a != agent}
        prompt = build_debate_revision_prompt(
            task_description=task["description"],
            function_stub=task["prompt_stub"],
            function_name_from_tests=task["function_name"],
            tests=task["tests"],
            agent_name=agent,
            own_code=initial_outputs[agent]["code"],
            peer_codes=peers,
            tests_preview_limit=tests_preview_limit,
        )
        final_debate_requests[agent] = {
            "system_prompt": CODING_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "model": model,
        }

    final_debate_results = _parallel_calls(client, final_debate_requests)
    final_outputs: Dict[str, Dict[str, Any]] = {}
    for agent in debate_agents:
        llm_res = final_debate_results[agent]
        eval_res = evaluate_code(task, llm_res["output_text"], timeout_sec=eval_timeout_sec)
        final_outputs[agent] = {
            "code": eval_res["code"],
            "eval": eval_res,
            "llm": llm_res,
            "prompt": final_debate_requests[agent]["user_prompt"],
        }
        stage_index += 1
        rows.append(
            _make_stage_row(
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                stage_name="debate_final",
                agent_name=f"debate_agent_{agent}",
                model_name=model,
                system_prompt=CODING_SYSTEM_PROMPT,
                input_prompt=final_debate_requests[agent]["user_prompt"],
                output_text=llm_res["output_text"],
                output_code=eval_res["code"],
                input_tokens=llm_res["usage"]["input_tokens"],
                output_tokens=llm_res["usage"]["output_tokens"],
                total_tokens=llm_res["usage"]["total_tokens"],
                latency_sec=llm_res["latency_sec"],
                api_error=llm_res["error"],
                eval_result=eval_res,
            )
        )

    candidate_codes = {agent: final_outputs[agent]["code"] for agent in debate_agents}
    consensus_prompt = build_consensus_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        tests=task["tests"],
        candidate_codes=candidate_codes,
        tests_preview_limit=tests_preview_limit,
    )
    consensus_res = client.call(
        system_prompt=CONSENSUS_SYSTEM_PROMPT,
        user_prompt=consensus_prompt,
        model=model,
    )
    selected_agent, selected_reason = _parse_consensus_choice(consensus_res["output_text"], debate_agents)
    if not selected_agent:
        order = {name: idx for idx, name in enumerate(debate_agents)}
        selected_agent = max(
            debate_agents,
            key=lambda x: (final_outputs[x]["eval"]["pass_count"], -order[x]),
        )
        fallback_reason = "Consensus output parse failed; fallback to best pass count."
        if selected_reason:
            selected_reason = f"{selected_reason} | {fallback_reason}"
        else:
            selected_reason = fallback_reason

    selected_code = final_outputs[selected_agent]["code"]
    selected_eval = final_outputs[selected_agent]["eval"]
    stage_index += 1
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="consensus",
            agent_name="consensus_agent",
            model_name=model,
            system_prompt=CONSENSUS_SYSTEM_PROMPT,
            input_prompt=consensus_prompt,
            output_text=consensus_res["output_text"],
            output_code=selected_code,
            input_tokens=consensus_res["usage"]["input_tokens"],
            output_tokens=consensus_res["usage"]["output_tokens"],
            total_tokens=consensus_res["usage"]["total_tokens"],
            latency_sec=consensus_res["latency_sec"],
            api_error=consensus_res["error"],
            eval_result=selected_eval,
            selected_agent=selected_agent,
            selected_reason=selected_reason,
        )
    )

    reflection_prompt = build_reflection_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        selected_code=selected_code,
    )
    reflection_res = client.call(
        system_prompt=CODING_SYSTEM_PROMPT,
        user_prompt=reflection_prompt,
        model=model,
    )
    reflection_eval = evaluate_code(task, reflection_res["output_text"], timeout_sec=eval_timeout_sec)
    reflected_code = reflection_eval["code"]
    stage_index += 1
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="reflection",
            agent_name="reflection_agent",
            model_name=model,
            system_prompt=CODING_SYSTEM_PROMPT,
            input_prompt=reflection_prompt,
            output_text=reflection_res["output_text"],
            output_code=reflected_code,
            input_tokens=reflection_res["usage"]["input_tokens"],
            output_tokens=reflection_res["usage"]["output_tokens"],
            total_tokens=reflection_res["usage"]["total_tokens"],
            latency_sec=reflection_res["latency_sec"],
            api_error=reflection_res["error"],
            eval_result=reflection_eval,
        )
    )

    verification_prompt = build_verification_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        reflected_code=reflected_code,
        tests=task["tests"],
        tests_preview_limit=tests_preview_limit,
    )
    verification_res = client.call(
        system_prompt=CODING_SYSTEM_PROMPT,
        user_prompt=verification_prompt,
        model=model,
    )
    verification_eval = evaluate_code(task, verification_res["output_text"], timeout_sec=eval_timeout_sec)
    verified_code = verification_eval["code"]
    stage_index += 1
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="verification",
            agent_name="verification_agent",
            model_name=model,
            system_prompt=CODING_SYSTEM_PROMPT,
            input_prompt=verification_prompt,
            output_text=verification_res["output_text"],
            output_code=verified_code,
            input_tokens=verification_res["usage"]["input_tokens"],
            output_tokens=verification_res["usage"]["output_tokens"],
            total_tokens=verification_res["usage"]["total_tokens"],
            latency_sec=verification_res["latency_sec"],
            api_error=verification_res["error"],
            eval_result=verification_eval,
        )
    )

    testing_eval = evaluate_code(task, verified_code, timeout_sec=eval_timeout_sec)
    testing_output = _safe_json_dumps(
        {
            "exec_status": testing_eval.get("exec_status", ""),
            "tests_passed": testing_eval.get("pass_count", 0),
            "tests_total": testing_eval.get("total_count", 0),
            "failures": testing_eval.get("failures", []),
        }
    )
    stage_index += 1
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="testing",
            agent_name="testing_stage",
            model_name="local_python_tester",
            system_prompt="",
            input_prompt="Run official MBPP tests against verification output.",
            output_text=testing_output,
            output_code=verified_code,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_sec=float(testing_eval.get("eval_latency_sec", 0.0) or 0.0),
            api_error="",
            eval_result=testing_eval,
        )
    )

    correction_prompt = build_error_correction_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        verified_code=verified_code,
        test_feedback=build_test_feedback(testing_eval),
        tests=task["tests"],
        tests_preview_limit=tests_preview_limit,
    )
    correction_res = client.call(
        system_prompt=CODING_SYSTEM_PROMPT,
        user_prompt=correction_prompt,
        model=model,
    )
    correction_eval = evaluate_code(task, correction_res["output_text"], timeout_sec=eval_timeout_sec)
    final_code = correction_eval["code"]
    stage_index += 1
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="error_correction",
            agent_name="error_correction_agent",
            model_name=model,
            system_prompt=CODING_SYSTEM_PROMPT,
            input_prompt=correction_prompt,
            output_text=correction_res["output_text"],
            output_code=final_code,
            input_tokens=correction_res["usage"]["input_tokens"],
            output_tokens=correction_res["usage"]["output_tokens"],
            total_tokens=correction_res["usage"]["total_tokens"],
            latency_sec=correction_res["latency_sec"],
            api_error=correction_res["error"],
            eval_result=correction_eval,
        )
    )

    return rows, correction_eval, final_code


def load_mbpp_sanitized_tasks(
    *,
    split: str,
    limit: Optional[int] = None,
    task_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    tasks: List[Dict[str, Any]] = []
    for rec in ds:
        task_id = int(rec["task_id"])
        if task_ids is not None and task_id not in task_ids:
            continue
        description = (rec.get("prompt") or "").strip()
        tests = list(rec.get("test_list") or [])
        function_name = extract_function_name_from_tests(tests)
        if not function_name:
            function_name = extract_function_name_from_code(rec.get("code", "") or "") or "solution"
        prompt_stub = build_prompt_stub(description, rec.get("code", "") or "", function_name)
        setup_code = "\n".join(rec.get("test_imports") or [])
        tasks.append(
            {
                "split": split,
                "task_id": task_id,
                "description": description,
                "tests": tests,
                "setup_code": setup_code,
                "reference_code": rec.get("code", "") or "",
                "function_name": function_name,
                "prompt_stub": prompt_stub,
            }
        )
    tasks.sort(key=lambda x: x["task_id"])
    if limit is not None:
        tasks = tasks[: max(0, int(limit))]
    return tasks


def _parse_task_ids(raw: Optional[str]) -> Optional[set[int]]:
    if not raw:
        return None
    ids: set[int] = set()
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        ids.add(int(chunk))
    return ids or None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run a multi-stage MBPP pipeline with debate -> consensus -> reflection -> "
            "verification -> testing -> error correction, and write one CSV per task."
        )
    )
    ap.add_argument("--splits", type=str, default="train,test", help="Comma-separated MBPP splits.")
    ap.add_argument("--limit_per_split", type=int, default=None, help="Optional limit per split.")
    ap.add_argument("--task_ids", type=str, default=None, help="Optional comma-separated task_ids filter.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for Anthropic API.")
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens per model call.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    ap.add_argument(
        "--eval_timeout_sec",
        type=float,
        default=DEFAULT_EVAL_TIMEOUT,
        help="Timeout for local test execution per stage.",
    )
    ap.add_argument(
        "--tests_preview_limit",
        type=int,
        default=DEFAULT_TESTS_PREVIEW_LIMIT,
        help="How many official tests to include in prompts (0 means all).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Root output directory (default is inside LAS_MBPP).",
    )
    ap.add_argument("--run_name", type=str, default=None, help="Optional run name folder.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing per-task CSV files.")
    ap.add_argument("--anthropic_api_key", type=str, default=None, help="Overrides ANTHROPIC_API_KEY.")
    ap.add_argument("--anthropic_base_url", type=str, default=None, help="Overrides ANTHROPIC_BASE_URL.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(DEFAULT_SPLITS)
    task_ids = _parse_task_ids(args.task_ids)

    api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    base_url = args.anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it or pass --anthropic_api_key."
        )

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{timestamp_tag}_{_slugify_name(args.run_name)}"
    else:
        run_id = timestamp_tag
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    client = AnthropicChatClient(
        api_key=api_key,
        base_url=base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    summary_rows: List[Dict[str, Any]] = []
    for split in splits:
        tasks = load_mbpp_sanitized_tasks(
            split=split,
            limit=args.limit_per_split,
            task_ids=task_ids,
        )
        split_dir = run_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        correct_count = 0
        processed_count = 0
        pbar = tqdm(tasks, desc=f"LAS_MBPP pipeline ({split})")
        for task in pbar:
            task_csv = split_dir / f"task_{task['task_id']}.csv"
            if task_csv.exists() and not args.overwrite:
                continue

            rows, final_eval, final_code = run_pipeline_for_task(
                run_id=run_id,
                task=task,
                client=client,
                model=args.model,
                eval_timeout_sec=args.eval_timeout_sec,
                tests_preview_limit=args.tests_preview_limit,
            )
            pd.DataFrame(rows).to_csv(task_csv, index=False)

            processed_count += 1
            if final_eval.get("passed_all"):
                correct_count += 1
            pass_rate = (correct_count / processed_count) if processed_count else 0.0
            pbar.set_postfix(pass1=f"{pass_rate:.2%}", processed=processed_count)

            summary_rows.append(
                {
                    "run_id": run_id,
                    "split": split,
                    "task_id": task["task_id"],
                    "function_name_from_tests": task["function_name"],
                    "final_is_correct": bool(final_eval.get("passed_all", False)),
                    "tests_passed": int(final_eval.get("pass_count", 0) or 0),
                    "tests_total": int(final_eval.get("total_count", 0) or 0),
                    "final_exec_status": final_eval.get("exec_status", ""),
                    "final_code": final_code,
                    "task_csv": str(task_csv),
                }
            )

    summary_path = run_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_id": run_id,
                "timestamp_utc": _utc_now_iso(),
                "splits": splits,
                "limit_per_split": args.limit_per_split,
                "task_ids": sorted(task_ids) if task_ids else None,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "eval_timeout_sec": args.eval_timeout_sec,
                "tests_preview_limit": args.tests_preview_limit,
                "output_dir": str(run_dir),
                "anthropic_base_url": base_url or "",
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Run completed. Per-task CSV files are in: {run_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Run config JSON: {config_path}")


if __name__ == "__main__":
    main()
