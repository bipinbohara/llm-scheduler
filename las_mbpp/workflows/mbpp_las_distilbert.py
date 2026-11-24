#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from las_mbpp.config import DEFAULT_GATE_CHECKPOINT, DEFAULT_RESULTS_DIR  # noqa: E402
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


DEFAULT_MODEL = "doubao-seed-1.6-lite"
DEFAULT_SPLITS = ("test", "train")
DEFAULT_EVAL_TIMEOUT = 60.0
DEFAULT_TESTS_PREVIEW_LIMIT = 20
DEFAULT_MAX_TOKENS = 500000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TASK_CONCURRENCY = 10
DEFAULT_LLM_API = "doubao"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_GATE_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_GATE_CHECKPOINT_PATH = DEFAULT_GATE_CHECKPOINT
DEFAULT_GATE_MAX_LENGTH = 512
DEFAULT_GATE_WEIGHT_DISTILBERT = 0.85
DEFAULT_GATE_WEIGHT_LOCAL_AGREEMENT = 0.07
DEFAULT_GATE_WEIGHT_ADHERE = 0.06
DEFAULT_GATE_WEIGHT_HISTORY = 0.02
DEFAULT_GATE_THRESHOLD = 0.803
DEFAULT_LAS_MAX_TOKENS = 4096
DEFAULT_LAS_TEMPERATURE = 0.0

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")

LAS_ROUTER_SYSTEM_PROMPT = (
    "You are a routing controller for a multi-agent LLM workflow. "
    "Given the current intermediate result, gate signals, and the available agents, "
    "decide where to send the request next. Be conservative on high-risk tasks."
)

ASSERT_CALL_RE = re.compile(r"assert\s+([A-Za-z_]\w*)\s*\(")
DEF_RE = re.compile(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*:")
FENCE_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Work around broken deepspeed+pydantic combos in some environments.
# This workflow does not use DeepSpeed; hide it from dependency probes.
if os.environ.get("LAS_MBPP_DISABLE_DEEPSPEED", "1") == "1":
    _orig_find_spec = importlib.util.find_spec

    def _find_spec_without_deepspeed(name: str, package=None):
        if name == "deepspeed" or name.startswith("deepspeed."):
            return None
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _find_spec_without_deepspeed

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as _transformers_exc:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = _transformers_exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    from jsonschema import ValidationError as JsonSchemaValidationError
    from jsonschema import validate as jsonschema_validate
except Exception:  # noqa: BLE001
    JsonSchemaValidationError = Exception
    jsonschema_validate = None

try:
    from pydantic import BaseModel, ValidationError
except Exception:  # noqa: BLE001
    BaseModel = object
    ValidationError = Exception


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


def _print_llm_api_error(backend: str, model: str, error_text: str) -> None:
    print(f"[LLM API ERROR] backend={backend} model={model} error={error_text}", flush=True)


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


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def parse_float(value: Any) -> Optional[float]:
    text = safe_text(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = safe_text(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


class SpecRowModel(BaseModel):  # type: ignore[misc,valid-type]
    stage_name: str
    agent_name: str
    function_name_from_tests: Optional[str] = None
    output_code: Optional[str] = None
    agent_output: Optional[str] = None
    selected_agent: Optional[str] = None
    tests_passed: Optional[float] = None
    tests_total: Optional[float] = None


SPEC_SCHEMA = {
    "type": "object",
    "required": ["stage_name", "agent_name"],
    "additionalProperties": False,
    "properties": {
        "stage_name": {"type": "string", "minLength": 1},
        "agent_name": {"type": "string", "minLength": 1},
        "function_name_from_tests": {"type": ["string", "null"]},
        "output_code": {"type": ["string", "null"]},
        "agent_output": {"type": ["string", "null"]},
        "selected_agent": {"type": ["string", "null"]},
        "tests_passed": {"type": ["number", "null"]},
        "tests_total": {"type": ["number", "null"]},
    },
}


def validate_with_pydantic(data: Dict[str, Any]) -> bool:
    try:
        if hasattr(SpecRowModel, "model_validate"):
            SpecRowModel.model_validate(data)  # pydantic v2
        elif hasattr(SpecRowModel, "parse_obj"):
            SpecRowModel.parse_obj(data)  # pydantic v1
        else:
            return False
        return True
    except ValidationError:
        return False


def validate_with_jsonschema(data: Dict[str, Any]) -> bool:
    if jsonschema_validate is None:
        return False
    try:
        jsonschema_validate(instance=data, schema=SPEC_SCHEMA)
        return True
    except JsonSchemaValidationError:
        return False


def compute_spec_adherence_score(
    *,
    row: Dict[str, Any],
    stage_name: str,
    agent_name: str,
    code_text: str,
) -> float:
    fn_name = safe_text(row.get("function_name_from_tests")).strip() or None
    selected_agent = safe_text(row.get("selected_agent")).strip() or None

    normalized = {
        "stage_name": stage_name,
        "agent_name": agent_name,
        "function_name_from_tests": fn_name,
        "output_code": safe_text(row.get("output_code")) or None,
        "agent_output": safe_text(row.get("agent_output")) or None,
        "selected_agent": selected_agent,
        "tests_passed": parse_float(row.get("tests_passed")),
        "tests_total": parse_float(row.get("tests_total")),
    }

    checks: List[float] = []
    checks.append(1.0 if validate_with_pydantic(normalized) else 0.0)
    checks.append(1.0 if validate_with_jsonschema(normalized) else 0.0)
    checks.append(1.0 if code_text and code_text != "[NO_OUTPUT]" else 0.0)

    if fn_name:
        fn_ok = bool(re.search(rf"(?m)^\s*def\s+{re.escape(fn_name)}\s*\(", code_text))
    else:
        fn_ok = True
    checks.append(1.0 if fn_ok else 0.0)

    try:
        ast.parse(code_text)
        ast_ok = True
    except SyntaxError:
        ast_ok = False
    checks.append(1.0 if ast_ok else 0.0)

    tests_passed = normalized["tests_passed"]
    tests_total = normalized["tests_total"]
    numeric_ok = True
    if tests_total is not None and tests_total < 0:
        numeric_ok = False
    if tests_passed is not None and tests_passed < 0:
        numeric_ok = False
    if tests_passed is not None and tests_total is not None and tests_passed - tests_total > 1e-12:
        numeric_ok = False
    checks.append(1.0 if numeric_ok else 0.0)

    if stage_name == "consensus":
        selected_ok = bool(selected_agent and re.fullmatch(r"[ABC]", selected_agent))
        checks.append(1.0 if selected_ok else 0.0)

    return float(sum(checks) / len(checks))


def tokenize_for_jaccard(code_text: str) -> set[str]:
    return set(TOKEN_RE.findall((code_text or "").lower()))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def compute_local_agreement_for_codes(codes: Sequence[str]) -> float:
    if len(codes) < 2:
        return 1.0
    token_sets = [tokenize_for_jaccard(code) for code in codes]
    scores: List[float] = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            scores.append(jaccard_similarity(token_sets[i], token_sets[j]))
    if not scores:
        return 1.0
    return float(sum(scores) / len(scores))


def _coerce_json_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return _safe_json_dumps(value)


@dataclass
class GateConfig:
    weight_distillbert: float
    weight_local_agreement: float
    weight_adhere: float
    weight_history: float
    threshold: float


@dataclass
class GateEvaluation:
    distilbert_score: float
    spec_adherence_score: float
    local_agreement_score: float
    agent_history_score: float
    gate_score: float
    gate_passed: bool
    gate_decision: str
    gate_reason: str
    gate_latency_sec: float
    gate_input_tokens: int
    gate_output_tokens: int
    gate_total_tokens: int
    gate_distilbert_latency_sec: float
    las_called: bool
    las_prompt: str
    las_output: str
    las_action: str
    las_reason: str
    las_parse_error: str
    las_latency_sec: float
    las_input_tokens: int
    las_output_tokens: int
    las_total_tokens: int
    las_api_error: str


@dataclass
class RouteDecision:
    action: str
    source_stage: str
    source_agent: str
    route_reason: str
    code: str
    eval_result: Dict[str, Any]


class AgentHistoryTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._correct_sum: Dict[str, float] = defaultdict(float)
        self._seen_count: Dict[str, int] = defaultdict(int)

    def score(self, agent_name: str) -> float:
        with self._lock:
            count = self._seen_count.get(agent_name, 0)
            if count <= 0:
                return 0.0
            return float(self._correct_sum.get(agent_name, 0.0) / count)

    def update(self, agent_name: str, is_correct: bool) -> None:
        with self._lock:
            self._seen_count[agent_name] = int(self._seen_count.get(agent_name, 0) + 1)
            if is_correct:
                self._correct_sum[agent_name] = float(self._correct_sum.get(agent_name, 0.0) + 1.0)


class DistilBertPassrateScorer:
    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: Path,
        max_length: int,
    ) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "transformers+torch are required for gate DistilBERT scoring. "
                f"Import error: {_TRANSFORMERS_IMPORT_ERROR}"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DistilBERT checkpoint not found: {checkpoint_path}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        self._max_length = int(max_length)
        self._lock = threading.Lock()

    def score(self, code_text: str) -> Tuple[float, int, float]:
        text = code_text or ""
        t0 = time.perf_counter()
        with self._lock, torch.no_grad():
            enc = self._tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            input_tokens = int(enc["input_ids"].shape[1]) if "input_ids" in enc else _estimate_tokens(text)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            logits = self._model(**enc).logits
            prob = float(torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item())
        latency = time.perf_counter() - t0
        return prob, int(input_tokens), latency


_GATE_SCORER_CACHE: Dict[Tuple[str, str, int], DistilBertPassrateScorer] = {}
_GATE_SCORER_CACHE_LOCK = threading.Lock()


def get_or_create_gate_scorer(
    *,
    model_name: str,
    checkpoint_path: Path,
    max_length: int,
) -> DistilBertPassrateScorer:
    key = (model_name, str(checkpoint_path.resolve()), int(max_length))
    with _GATE_SCORER_CACHE_LOCK:
        scorer = _GATE_SCORER_CACHE.get(key)
        if scorer is not None:
            return scorer
        scorer = DistilBertPassrateScorer(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            max_length=max_length,
        )
        _GATE_SCORER_CACHE[key] = scorer
        return scorer


def _build_las_prompt(
    *,
    task: Dict[str, Any],
    step_id: int,
    stage_name: str,
    agent_name: str,
    gate_eval: GateEvaluation,
    syntax_check_passed: bool,
    tests_passed: int,
    tests_total: int,
    code_snippet: str,
) -> str:
    stage_roles = {
        "debate_initial": "Generate an initial solution.",
        "debate_final": "Revise the solution after peer comparison.",
        "consensus": "Select the strongest candidate implementation.",
        "reflection": "Self-reflect and improve selected code.",
        "verification": "Verify against edge cases and refine if needed.",
        "error_correction": "Fix code using testing feedback.",
    }
    role = stage_roles.get(stage_name, "Generate or refine solution code.")
    syntax_text = "passed" if syntax_check_passed else "failed"
    artifact = (code_snippet or "").strip()
    if len(artifact) > 2000:
        artifact = artifact[:2000] + "\n# ... [truncated]"
    return (
        "[Task]\n"
        "You are solving a programming problem. The final goal is to produce a correct\n"
        "Python function that passes all tests.\n\n"
        "[Current step]\n"
        f'agent_name: "{agent_name}"\n'
        f'agent_role: "{role}"\n'
        f"step_id: {step_id}\n"
        f'stage_name: "{stage_name}"\n\n'
        # "[Gate features]\n"
        # f"spec_adherence: {gate_eval.spec_adherence_score:.6f}\n"
        # f"distilbert_score: {gate_eval.distilbert_score:.6f}\n"
        # f"agreement_score: {gate_eval.local_agreement_score:.6f}\n"
        # f"historical_reliability: {gate_eval.agent_history_score:.6f}\n"
        # f"gate_score_g: {gate_eval.gate_score:.6f}\n\n"
        "[Validator summary]\n"
        f"* syntax_check: {syntax_text}\n"
        f"* tests_passed: {tests_passed}/{tests_total}\n\n"
        "[Available routes]\n"
        '1. early_exit_to: "final_output_unit"\n'
        '2. test send_to: "testing_then_error_correction"\n'
        "3. continue processing: no change\n\n"
        "[Current artifact]\n"
        f"{artifact}\n\n"
        "Decision:\n"
        "Choose the single best action from the list above. "
        "If you think the output is perfect with no any error possible, send it to the early_exit. "
        "If you think it's good, then send to the testing agent. "
        "If you think it has obvious problem and can be further refined, then send it to continue processing.\n"
        "No need to provide reasons. "
        "Respond in strict JSON with fields:\n"
        '{"action": "<one of: early_exit, test, refinement>", '
        # '"reason": "<short natural language justification>"}' 不需要提供reason
    )


def _parse_las_action(raw_text: str) -> Tuple[str, str, str]:
    data = _extract_json_dict(raw_text)
    reason = ""
    parse_error = ""
    action = ""
    if data:
        action = str(data.get("action", "")).strip().lower()
        reason = _coerce_json_scalar(data.get("reason")).strip()
    if action not in {"early_exit", "test", "refinement"}:
        lowered = (raw_text or "").lower()
        if "early_exit" in lowered or "early exit" in lowered:
            action = "early_exit"
        elif re.search(r"\btest\b", lowered):
            action = "test"
        elif "refinement" in lowered or "continue" in lowered:
            action = "refinement"
        else:
            action = "refinement"
            parse_error = "Unable to parse action from LAS output; defaulted to refinement."
    return action, reason, parse_error


def evaluate_gate_and_las(
    *,
    task: Dict[str, Any],
    stage_index: int,
    stage_name: str,
    agent_name: str,
    code_text: str,
    row_snapshot: Dict[str, Any],
    eval_result: Dict[str, Any],
    local_agreement_score: float,
    history_score: float,
    gate_config: GateConfig,
    scorer: DistilBertPassrateScorer,
    las_client: LLMClient,
    las_model: str,
    las_llm_api: str,
) -> GateEvaluation:
    gate_t0 = time.perf_counter()
    dist_score, gate_input_tokens, dist_latency = scorer.score(code_text)
    spec_score = compute_spec_adherence_score(
        row=row_snapshot,
        stage_name=stage_name,
        agent_name=agent_name,
        code_text=code_text,
    )
    local_score = float(local_agreement_score)
    hist_score = float(history_score)

    gate_score = (
        gate_config.weight_distillbert * dist_score
        + gate_config.weight_local_agreement * local_score
        + gate_config.weight_adhere * spec_score
        + gate_config.weight_history * hist_score
    )
    gate_passed = bool(gate_score >= gate_config.threshold)
    gate_decision = "passed_threshold_call_las" if gate_passed else "below_threshold_skip_las"
    gate_reason = (
        f"gate_score={gate_score:.6f} >= threshold={gate_config.threshold:.6f}"
        if gate_passed
        else f"gate_score={gate_score:.6f} < threshold={gate_config.threshold:.6f}"
    )

    gate_latency_sec = time.perf_counter() - gate_t0
    gate_output_tokens = 0
    gate_total_tokens = int(gate_input_tokens + gate_output_tokens)

    las_called = False
    las_prompt = ""
    las_output = ""
    las_action = ""
    las_reason = ""
    las_parse_error = ""
    las_latency_sec = 0.0
    las_input_tokens = 0
    las_output_tokens = 0
    las_total_tokens = 0
    las_api_error = ""

    if gate_passed:
        las_called = True
        syntax_ok = not bool(eval_result.get("exec_error"))
        tests_passed = int(eval_result.get("pass_count", 0) or 0)
        tests_total = int(eval_result.get("total_count", 0) or 0)
        las_prompt = _build_las_prompt(
            task=task,
            step_id=stage_index,
            stage_name=stage_name,
            agent_name=agent_name,
            gate_eval=GateEvaluation(
                distilbert_score=dist_score,
                spec_adherence_score=spec_score,
                local_agreement_score=local_score,
                agent_history_score=hist_score,
                gate_score=gate_score,
                gate_passed=gate_passed,
                gate_decision=gate_decision,
                gate_reason=gate_reason,
                gate_latency_sec=gate_latency_sec,
                gate_input_tokens=gate_input_tokens,
                gate_output_tokens=gate_output_tokens,
                gate_total_tokens=gate_total_tokens,
                gate_distilbert_latency_sec=dist_latency,
                las_called=False,
                las_prompt="",
                las_output="",
                las_action="",
                las_reason="",
                las_parse_error="",
                las_latency_sec=0.0,
                las_input_tokens=0,
                las_output_tokens=0,
                las_total_tokens=0,
                las_api_error="",
            ),
            syntax_check_passed=syntax_ok,
            tests_passed=tests_passed,
            tests_total=tests_total,
            code_snippet=code_text,
        )
        las_response = las_client.call(
            system_prompt=LAS_ROUTER_SYSTEM_PROMPT,
            user_prompt=las_prompt,
            model=las_model,
        )
        las_output = las_response.get("output_text", "") or ""
        las_api_error = las_response.get("error", "") or ""
        usage = las_response.get("usage", {}) or {}
        las_input_tokens = int(usage.get("input_tokens", 0) or 0)
        las_output_tokens = int(usage.get("output_tokens", 0) or 0)
        las_total_tokens = int(usage.get("total_tokens", las_input_tokens + las_output_tokens) or 0)
        las_latency_sec = float(las_response.get("latency_sec", 0.0) or 0.0)
        las_action, las_reason, las_parse_error = _parse_las_action(las_output)
        if las_api_error and not las_parse_error:
            las_parse_error = f"LAS API error on backend={las_llm_api}: {las_api_error}"

    return GateEvaluation(
        distilbert_score=float(dist_score),
        spec_adherence_score=float(spec_score),
        local_agreement_score=float(local_score),
        agent_history_score=float(hist_score),
        gate_score=float(gate_score),
        gate_passed=bool(gate_passed),
        gate_decision=gate_decision,
        gate_reason=gate_reason,
        gate_latency_sec=float(gate_latency_sec),
        gate_input_tokens=int(gate_input_tokens),
        gate_output_tokens=int(gate_output_tokens),
        gate_total_tokens=int(gate_total_tokens),
        gate_distilbert_latency_sec=float(dist_latency),
        las_called=bool(las_called),
        las_prompt=las_prompt,
        las_output=las_output,
        las_action=las_action,
        las_reason=las_reason,
        las_parse_error=las_parse_error,
        las_latency_sec=float(las_latency_sec),
        las_input_tokens=int(las_input_tokens),
        las_output_tokens=int(las_output_tokens),
        las_total_tokens=int(las_total_tokens),
        las_api_error=las_api_error,
    )


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


class LLMClient(Protocol):
    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
        ...


def _object_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                data = method()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return data
    return {}


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
            error_text = f"{type(exc).__name__}: {exc}"
            _print_llm_api_error("anthropic", model, error_text)
            input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
            return {
                "ok": False,
                "error": error_text,
                "output_text": "",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "total_tokens": input_tokens,
                },
                "latency_sec": latency,
            }


class ArkChatClient:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "volcenginesdkarkruntime package is required for --llm_api ark. "
                "Install with: pip install volcenginesdkarkruntime"
            ) from exc
        self.client = Ark(base_url=base_url, api_key=api_key)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def _extract_text(self, response: Any) -> str:
        direct_text = getattr(response, "output_text", None)
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()

        data = _object_to_dict(response)
        if isinstance(data.get("output_text"), str) and data.get("output_text", "").strip():
            return data["output_text"].strip()

        chunks: List[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                i_data = _object_to_dict(item) if not isinstance(item, dict) else item
                content = i_data.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for block in content:
                        b_data = _object_to_dict(block) if not isinstance(block, dict) else block
                        text = b_data.get("text") or b_data.get("output_text")
                        if isinstance(text, str) and text:
                            chunks.append(text)

        choices = data.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                c_data = _object_to_dict(choice) if not isinstance(choice, dict) else choice
                message = c_data.get("message")
                m_data = _object_to_dict(message) if not isinstance(message, dict) else message
                content = m_data.get("content")
                if isinstance(content, str) and content:
                    chunks.append(content)
                elif isinstance(content, list):
                    for block in content:
                        b_data = _object_to_dict(block) if not isinstance(block, dict) else block
                        text = b_data.get("text")
                        if isinstance(text, str) and text:
                            chunks.append(text)
        return "\n".join(part for part in chunks if part).strip()

    def _extract_usage(self, response: Any) -> Tuple[int, int]:
        usage_obj = getattr(response, "usage", None)
        usage_data = _object_to_dict(usage_obj)
        if not usage_data:
            data = _object_to_dict(response)
            usage_data = _object_to_dict(data.get("usage"))
        input_tokens = int(
            usage_data.get("input_tokens", 0)
            or usage_data.get("prompt_tokens", 0)
            or usage_data.get("request_tokens", 0)
            or 0
        )
        output_tokens = int(
            usage_data.get("output_tokens", 0)
            or usage_data.get("completion_tokens", 0)
            or usage_data.get("response_tokens", 0)
            or 0
        )
        return input_tokens, output_tokens

    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            input_text = (
                f"[System]\n{system_prompt}\n\n[User]\n{user_prompt}"
                if system_prompt
                else user_prompt
            )
            response = self.client.responses.create(
                model=model,
                input=input_text,
            )
            latency = time.perf_counter() - t0
            output_text = self._extract_text(response)
            input_tokens, output_tokens = self._extract_usage(response)
            if input_tokens == 0:
                input_tokens = _estimate_tokens(input_text)
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
            error_text = f"{type(exc).__name__}: {exc}"
            _print_llm_api_error("ark", model, error_text)
            input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
            return {
                "ok": False,
                "error": error_text,
                "output_text": "",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "total_tokens": input_tokens,
                },
                "latency_sec": latency,
            }


def _parallel_calls(
    client: LLMClient,
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
                _print_llm_api_error("worker", req.get("model", ""), f"{type(exc).__name__}: {exc}")
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


def create_llm_client(
    *,
    llm_api: str,
    max_tokens: int,
    temperature: float,
    anthropic_api_key: Optional[str],
    anthropic_base_url: Optional[str],
    ark_api_key: Optional[str],
    ark_base_url: Optional[str],
) -> LLMClient:
    backend = (llm_api or DEFAULT_LLM_API).strip().lower()
    if backend in {"minimax", "anthropic"}:
        if not anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass --anthropic_api_key."
            )
        return AnthropicChatClient(
            api_key=anthropic_api_key,
            base_url=anthropic_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if backend in {"doubao", "ark"}:
        if not ark_api_key:
            raise RuntimeError("ARK_API_KEY is not set. Export it or pass --ark_api_key.")
        return ArkChatClient(
            api_key=ark_api_key,
            base_url=ark_base_url or DEFAULT_ARK_BASE_URL,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported llm_api backend: {llm_api}")


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
        "distilbert_score": "",
        "spec_adherence_score": "",
        "local_agreement_score": "",
        "agent_history_score": "",
        "gate_weight_distillbert": "",
        "gate_weight_local_agreement": "",
        "gate_weight_adhere": "",
        "gate_weight_history": "",
        "gate_threshold": "",
        "gate_score": "",
        "gate_passed": "",
        "gate_decision": "",
        "gate_reason": "",
        "gate_latency_sec": 0.0,
        "gate_input_tokens": 0,
        "gate_output_tokens": 0,
        "gate_total_tokens": 0,
        "gate_distilbert_latency_sec": 0.0,
        "gate_unit_latency_sec": 0.0,
        "las_called": False,
        "las_llm_api": "",
        "las_model": "",
        "las_prompt": "",
        "las_output": "",
        "las_action": "",
        "las_reason": "",
        "las_parse_error": "",
        "las_latency_sec": 0.0,
        "las_input_tokens": 0,
        "las_output_tokens": 0,
        "las_total_tokens": 0,
        "las_module_latency_sec": 0.0,
        "las_module_input_tokens": 0,
        "las_module_output_tokens": 0,
        "las_api_error": "",
        "route_action": "",
        "route_applied": False,
        "route_target_stage": "",
        "route_source_stage": "",
    }


def _apply_gate_logs_to_row(
    *,
    row: Dict[str, Any],
    gate_eval: GateEvaluation,
    gate_config: GateConfig,
    las_llm_api: str,
    las_model: str,
) -> None:
    row["distilbert_score"] = float(gate_eval.distilbert_score)
    row["spec_adherence_score"] = float(gate_eval.spec_adherence_score)
    row["local_agreement_score"] = float(gate_eval.local_agreement_score)
    row["agent_history_score"] = float(gate_eval.agent_history_score)
    row["gate_weight_distillbert"] = float(gate_config.weight_distillbert)
    row["gate_weight_local_agreement"] = float(gate_config.weight_local_agreement)
    row["gate_weight_adhere"] = float(gate_config.weight_adhere)
    row["gate_weight_history"] = float(gate_config.weight_history)
    row["gate_threshold"] = float(gate_config.threshold)
    row["gate_score"] = float(gate_eval.gate_score)
    row["gate_passed"] = bool(gate_eval.gate_passed)
    row["gate_decision"] = gate_eval.gate_decision
    row["gate_reason"] = gate_eval.gate_reason
    row["gate_latency_sec"] = float(gate_eval.gate_latency_sec)
    row["gate_input_tokens"] = int(gate_eval.gate_input_tokens)
    row["gate_output_tokens"] = int(gate_eval.gate_output_tokens)
    row["gate_total_tokens"] = int(gate_eval.gate_total_tokens)
    row["gate_distilbert_latency_sec"] = float(gate_eval.gate_distilbert_latency_sec)
    row["gate_unit_latency_sec"] = float(gate_eval.gate_latency_sec)
    row["las_called"] = bool(gate_eval.las_called)
    row["las_llm_api"] = las_llm_api if gate_eval.las_called else ""
    row["las_model"] = las_model if gate_eval.las_called else ""
    row["las_prompt"] = gate_eval.las_prompt
    row["las_output"] = gate_eval.las_output
    row["las_action"] = gate_eval.las_action
    row["las_reason"] = gate_eval.las_reason
    row["las_parse_error"] = gate_eval.las_parse_error
    row["las_latency_sec"] = float(gate_eval.las_latency_sec)
    row["las_input_tokens"] = int(gate_eval.las_input_tokens)
    row["las_output_tokens"] = int(gate_eval.las_output_tokens)
    row["las_total_tokens"] = int(gate_eval.las_total_tokens)
    row["las_module_latency_sec"] = float(gate_eval.las_latency_sec)
    row["las_module_input_tokens"] = int(gate_eval.las_input_tokens)
    row["las_module_output_tokens"] = int(gate_eval.las_output_tokens)
    row["las_api_error"] = gate_eval.las_api_error


def _maybe_build_route_decision(
    *,
    stage_name: str,
    agent_name: str,
    code: str,
    eval_result: Dict[str, Any],
    gate_eval: GateEvaluation,
) -> Optional[RouteDecision]:
    action = gate_eval.las_action.strip().lower()
    if action not in {"early_exit", "test"}:
        return None
    reason = gate_eval.las_reason or gate_eval.las_parse_error or gate_eval.gate_reason
    return RouteDecision(
        action=action,
        source_stage=stage_name,
        source_agent=agent_name,
        route_reason=reason,
        code=code,
        eval_result=eval_result,
    )


def _append_testing_and_error_correction(
    *,
    rows: List[Dict[str, Any]],
    run_id: str,
    task: Dict[str, Any],
    stage_index: int,
    client: LLMClient,
    model: str,
    eval_timeout_sec: float,
    tests_preview_limit: int,
    base_code: str,
    route_action: str,
    route_source_stage: str,
) -> Tuple[int, Dict[str, Any], str]:
    testing_eval = evaluate_code(task, base_code, timeout_sec=eval_timeout_sec)
    testing_output = _safe_json_dumps(
        {
            "exec_status": testing_eval.get("exec_status", ""),
            "tests_passed": testing_eval.get("pass_count", 0),
            "tests_total": testing_eval.get("total_count", 0),
            "failures": testing_eval.get("failures", []),
        }
    )
    stage_index += 1
    route_tag = route_source_stage or "verification"
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="testing",
            agent_name="testing_stage",
            model_name="local_python_tester",
            system_prompt="",
            input_prompt=f"Run official MBPP tests against code from {route_tag}.",
            output_text=testing_output,
            output_code=base_code,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_sec=float(testing_eval.get("eval_latency_sec", 0.0) or 0.0),
            api_error="",
            eval_result=testing_eval,
        )
    )
    rows[-1]["route_action"] = route_action
    rows[-1]["route_target_stage"] = "testing"
    rows[-1]["route_source_stage"] = route_source_stage
    rows[-1]["route_applied"] = bool(route_action)

    correction_prompt = build_error_correction_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        verified_code=base_code,
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
    rows[-1]["route_action"] = route_action
    rows[-1]["route_target_stage"] = "error_correction"
    rows[-1]["route_source_stage"] = route_source_stage
    rows[-1]["route_applied"] = bool(route_action)
    return stage_index, correction_eval, final_code


def _append_early_exit_stage(
    *,
    rows: List[Dict[str, Any]],
    run_id: str,
    task: Dict[str, Any],
    stage_index: int,
    route: RouteDecision,
) -> Tuple[int, Dict[str, Any], str]:
    stage_index += 1
    route_output = _safe_json_dumps(
        {
            "action": route.action,
            "source_stage": route.source_stage,
            "source_agent": route.source_agent,
            "reason": route.route_reason,
        }
    )
    rows.append(
        _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="final_output_unit",
            agent_name="las_router",
            model_name="router_local_control",
            system_prompt="",
            input_prompt="Apply LAS early-exit decision.",
            output_text=route_output,
            output_code=route.code,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            latency_sec=0.0,
            api_error="",
            eval_result=route.eval_result,
        )
    )
    rows[-1]["route_action"] = route.action
    rows[-1]["route_target_stage"] = "final_output_unit"
    rows[-1]["route_source_stage"] = route.source_stage
    rows[-1]["route_applied"] = True
    return stage_index, route.eval_result, route.code


def run_pipeline_for_task(
    *,
    run_id: str,
    task: Dict[str, Any],
    client: LLMClient,
    model: str,
    las_client: LLMClient,
    las_model: str,
    las_llm_api: str,
    gate_scorer: DistilBertPassrateScorer,
    gate_config: GateConfig,
    history_tracker: AgentHistoryTracker,
    eval_timeout_sec: float,
    tests_preview_limit: int,
    include_test_examples_in_previous_prompts: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    rows: List[Dict[str, Any]] = []
    stage_index = 0
    debate_agents = ["A", "B", "C"]
    route_decision: Optional[RouteDecision] = None

    def _gate_one_row(
        *,
        row: Dict[str, Any],
        stage_name: str,
        agent_name: str,
        code: str,
        eval_result: Dict[str, Any],
        local_agreement_score: float,
    ) -> Optional[RouteDecision]:
        history_score = history_tracker.score(agent_name)
        gate_eval = evaluate_gate_and_las(
            task=task,
            stage_index=int(row["stage_index"]),
            stage_name=stage_name,
            agent_name=agent_name,
            code_text=code,
            row_snapshot=row,
            eval_result=eval_result,
            local_agreement_score=local_agreement_score,
            history_score=history_score,
            gate_config=gate_config,
            scorer=gate_scorer,
            las_client=las_client,
            las_model=las_model,
            las_llm_api=las_llm_api,
        )
        _apply_gate_logs_to_row(
            row=row,
            gate_eval=gate_eval,
            gate_config=gate_config,
            las_llm_api=las_llm_api,
            las_model=las_model,
        )
        history_tracker.update(agent_name, bool(eval_result.get("passed_all", False)))
        return _maybe_build_route_decision(
            stage_name=stage_name,
            agent_name=agent_name,
            code=code,
            eval_result=eval_result,
            gate_eval=gate_eval,
        )

    initial_outputs: Dict[str, Dict[str, Any]] = {}
    generated_initial_agents: List[str] = []
    for agent in debate_agents:
        prompt = build_debate_initial_prompt(
            task_description=task["description"],
            function_stub=task["prompt_stub"],
            function_name_from_tests=task["function_name"],
            tests=task["tests"],
            agent_name=agent,
            tests_preview_limit=tests_preview_limit,
            include_official_test_examples=include_test_examples_in_previous_prompts,
        )
        llm_res = client.call(
            system_prompt=CODING_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=model,
        )
        eval_res = evaluate_code(task, llm_res["output_text"], timeout_sec=eval_timeout_sec)
        initial_outputs[agent] = {
            "code": eval_res["code"],
            "eval": eval_res,
            "llm": llm_res,
            "prompt": prompt,
        }
        generated_initial_agents.append(agent)
        stage_index += 1
        row = _make_stage_row(
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            stage_name="debate_initial",
            agent_name=f"debate_agent_{agent}",
            model_name=model,
            system_prompt=CODING_SYSTEM_PROMPT,
            input_prompt=prompt,
            output_text=llm_res["output_text"],
            output_code=eval_res["code"],
            input_tokens=llm_res["usage"]["input_tokens"],
            output_tokens=llm_res["usage"]["output_tokens"],
            total_tokens=llm_res["usage"]["total_tokens"],
            latency_sec=llm_res["latency_sec"],
            api_error=llm_res["error"],
            eval_result=eval_res,
        )
        rows.append(row)

        local_initial = compute_local_agreement_for_codes(
            [initial_outputs[a]["code"] for a in generated_initial_agents]
        )
        route_decision = _gate_one_row(
            row=row,
            stage_name="debate_initial",
            agent_name=f"debate_agent_{agent}",
            code=initial_outputs[agent]["code"],
            eval_result=initial_outputs[agent]["eval"],
            local_agreement_score=local_initial,
        )
        if route_decision is not None:
            row["route_action"] = route_decision.action
            row["route_target_stage"] = (
                "final_output_unit" if route_decision.action == "early_exit" else "testing"
            )
            row["route_source_stage"] = route_decision.source_stage
            row["route_applied"] = True
            break

    if route_decision is not None:
        if route_decision.action == "early_exit":
            stage_index, final_eval, final_code = _append_early_exit_stage(
                rows=rows,
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                route=route_decision,
            )
            return rows, final_eval, final_code
        stage_index, final_eval, final_code = _append_testing_and_error_correction(
            rows=rows,
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            client=client,
            model=model,
            eval_timeout_sec=eval_timeout_sec,
            tests_preview_limit=tests_preview_limit,
            base_code=route_decision.code,
            route_action=route_decision.action,
            route_source_stage=route_decision.source_stage,
        )
        return rows, final_eval, final_code

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
            include_official_test_examples=include_test_examples_in_previous_prompts,
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

    local_final = compute_local_agreement_for_codes([final_outputs[a]["code"] for a in debate_agents])
    for agent in debate_agents:
        candidate_row = rows[-3 + debate_agents.index(agent)]
        route_decision = _gate_one_row(
            row=candidate_row,
            stage_name="debate_final",
            agent_name=f"debate_agent_{agent}",
            code=final_outputs[agent]["code"],
            eval_result=final_outputs[agent]["eval"],
            local_agreement_score=local_final,
        )
        if route_decision is not None:
            candidate_row["route_action"] = route_decision.action
            candidate_row["route_target_stage"] = (
                "final_output_unit" if route_decision.action == "early_exit" else "testing"
            )
            candidate_row["route_source_stage"] = route_decision.source_stage
            candidate_row["route_applied"] = True
            break

    if route_decision is not None:
        if route_decision.action == "early_exit":
            stage_index, final_eval, final_code = _append_early_exit_stage(
                rows=rows,
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                route=route_decision,
            )
            return rows, final_eval, final_code
        stage_index, final_eval, final_code = _append_testing_and_error_correction(
            rows=rows,
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            client=client,
            model=model,
            eval_timeout_sec=eval_timeout_sec,
            tests_preview_limit=tests_preview_limit,
            base_code=route_decision.code,
            route_action=route_decision.action,
            route_source_stage=route_decision.source_stage,
        )
        return rows, final_eval, final_code

    candidate_codes = {agent: final_outputs[agent]["code"] for agent in debate_agents}
    consensus_prompt = build_consensus_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        tests=task["tests"],
        candidate_codes=candidate_codes,
        tests_preview_limit=tests_preview_limit,
        include_official_test_examples=include_test_examples_in_previous_prompts,
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

    consensus_row = rows[-1]
    route_decision = _gate_one_row(
        row=consensus_row,
        stage_name="consensus",
        agent_name="consensus_agent",
        code=selected_code,
        eval_result=selected_eval,
        local_agreement_score=1.0,
    )
    if route_decision is not None:
        consensus_row["route_action"] = route_decision.action
        consensus_row["route_target_stage"] = (
            "final_output_unit" if route_decision.action == "early_exit" else "testing"
        )
        consensus_row["route_source_stage"] = route_decision.source_stage
        consensus_row["route_applied"] = True
        if route_decision.action == "early_exit":
            stage_index, final_eval, final_code = _append_early_exit_stage(
                rows=rows,
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                route=route_decision,
            )
            return rows, final_eval, final_code
        stage_index, final_eval, final_code = _append_testing_and_error_correction(
            rows=rows,
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            client=client,
            model=model,
            eval_timeout_sec=eval_timeout_sec,
            tests_preview_limit=tests_preview_limit,
            base_code=route_decision.code,
            route_action=route_decision.action,
            route_source_stage=route_decision.source_stage,
        )
        return rows, final_eval, final_code

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

    reflection_row = rows[-1]
    route_decision = _gate_one_row(
        row=reflection_row,
        stage_name="reflection",
        agent_name="reflection_agent",
        code=reflected_code,
        eval_result=reflection_eval,
        local_agreement_score=1.0,
    )
    if route_decision is not None:
        reflection_row["route_action"] = route_decision.action
        reflection_row["route_target_stage"] = (
            "final_output_unit" if route_decision.action == "early_exit" else "testing"
        )
        reflection_row["route_source_stage"] = route_decision.source_stage
        reflection_row["route_applied"] = True
        if route_decision.action == "early_exit":
            stage_index, final_eval, final_code = _append_early_exit_stage(
                rows=rows,
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                route=route_decision,
            )
            return rows, final_eval, final_code
        stage_index, final_eval, final_code = _append_testing_and_error_correction(
            rows=rows,
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            client=client,
            model=model,
            eval_timeout_sec=eval_timeout_sec,
            tests_preview_limit=tests_preview_limit,
            base_code=route_decision.code,
            route_action=route_decision.action,
            route_source_stage=route_decision.source_stage,
        )
        return rows, final_eval, final_code

    verification_prompt = build_verification_prompt(
        task_description=task["description"],
        function_stub=task["prompt_stub"],
        function_name_from_tests=task["function_name"],
        reflected_code=reflected_code,
        tests=task["tests"],
        tests_preview_limit=tests_preview_limit,
        include_official_test_examples=include_test_examples_in_previous_prompts,
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

    verification_row = rows[-1]
    route_decision = _gate_one_row(
        row=verification_row,
        stage_name="verification",
        agent_name="verification_agent",
        code=verified_code,
        eval_result=verification_eval,
        local_agreement_score=1.0,
    )
    if route_decision is not None:
        verification_row["route_action"] = route_decision.action
        verification_row["route_target_stage"] = (
            "final_output_unit" if route_decision.action == "early_exit" else "testing"
        )
        verification_row["route_source_stage"] = route_decision.source_stage
        verification_row["route_applied"] = True
        if route_decision.action == "early_exit":
            stage_index, final_eval, final_code = _append_early_exit_stage(
                rows=rows,
                run_id=run_id,
                task=task,
                stage_index=stage_index,
                route=route_decision,
            )
            return rows, final_eval, final_code
        stage_index, final_eval, final_code = _append_testing_and_error_correction(
            rows=rows,
            run_id=run_id,
            task=task,
            stage_index=stage_index,
            client=client,
            model=model,
            eval_timeout_sec=eval_timeout_sec,
            tests_preview_limit=tests_preview_limit,
            base_code=route_decision.code,
            route_action=route_decision.action,
            route_source_stage=route_decision.source_stage,
        )
        return rows, final_eval, final_code

    stage_index, correction_eval, final_code = _append_testing_and_error_correction(
        rows=rows,
        run_id=run_id,
        task=task,
        stage_index=stage_index,
        client=client,
        model=model,
        eval_timeout_sec=eval_timeout_sec,
        tests_preview_limit=tests_preview_limit,
        base_code=verified_code,
        route_action="",
        route_source_stage="verification",
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


def _run_task_pipeline_job(
    *,
    run_id: str,
    task: Dict[str, Any],
    llm_api: str,
    model: str,
    max_tokens: int,
    temperature: float,
    anthropic_api_key: Optional[str],
    anthropic_base_url: Optional[str],
    ark_api_key: Optional[str],
    ark_base_url: Optional[str],
    las_llm_api: str,
    las_model: str,
    las_max_tokens: int,
    las_temperature: float,
    las_anthropic_api_key: Optional[str],
    las_anthropic_base_url: Optional[str],
    las_ark_api_key: Optional[str],
    las_ark_base_url: Optional[str],
    gate_scorer: DistilBertPassrateScorer,
    gate_config: GateConfig,
    history_tracker: AgentHistoryTracker,
    eval_timeout_sec: float,
    tests_preview_limit: int,
    include_test_examples_in_previous_prompts: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    client = create_llm_client(
        llm_api=llm_api,
        max_tokens=max_tokens,
        temperature=temperature,
        anthropic_api_key=anthropic_api_key,
        anthropic_base_url=anthropic_base_url,
        ark_api_key=ark_api_key,
        ark_base_url=ark_base_url,
    )
    las_client = create_llm_client(
        llm_api=las_llm_api,
        max_tokens=las_max_tokens,
        temperature=las_temperature,
        anthropic_api_key=las_anthropic_api_key,
        anthropic_base_url=las_anthropic_base_url,
        ark_api_key=las_ark_api_key,
        ark_base_url=las_ark_base_url,
    )
    return run_pipeline_for_task(
        run_id=run_id,
        task=task,
        client=client,
        model=model,
        las_client=las_client,
        las_model=las_model,
        las_llm_api=las_llm_api,
        gate_scorer=gate_scorer,
        gate_config=gate_config,
        history_tracker=history_tracker,
        eval_timeout_sec=eval_timeout_sec,
        tests_preview_limit=tests_preview_limit,
        include_test_examples_in_previous_prompts=include_test_examples_in_previous_prompts,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run an MBPP multi-agent pipeline with online gate scoring + LAS routing "
            "(early_exit/test/refinement) and write one CSV per task."
        )
    )
    ap.add_argument("--splits", type=str, default="train,test", help="Comma-separated MBPP splits.")
    ap.add_argument("--limit_per_split", type=int, default=None, help="Optional limit per split.")
    ap.add_argument("--task_ids", type=str, default=None, help="Optional comma-separated task_ids filter.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for selected LLM backend.")
    ap.add_argument(
        "--llm_api",
        type=str,
        default=DEFAULT_LLM_API,
        choices=["minimax", "doubao", "anthropic", "ark"],
        help=(
            "LLM API backend: minimax (Anthropic-compatible), doubao (Ark), "
            "or aliases anthropic/ark."
        ),
    )
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens per model call.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    ap.add_argument(
        "--las_llm_api",
        type=str,
        default=None,
        choices=["minimax", "doubao", "anthropic", "ark"],
        help="Optional separate LAS router backend. Defaults to --llm_api.",
    )
    ap.add_argument(
        "--las_model",
        type=str,
        default=None,
        help="Optional separate LAS router model. Defaults to --model.",
    )
    ap.add_argument(
        "--las_max_tokens",
        type=int,
        default=None,
        help="Optional LAS max_tokens. Defaults to --max_tokens.",
    )
    ap.add_argument(
        "--las_temperature",
        type=float,
        default=None,
        help="Optional LAS temperature. Defaults to --temperature.",
    )
    ap.add_argument(
        "--task_concurrency",
        type=int,
        default=DEFAULT_TASK_CONCURRENCY,
        help="Number of tasks to process in parallel.",
    )
    ap.add_argument(
        "--include_test_examples_in_previous_prompts",
        action="store_true",
        help=(
            "If set, include official test examples in debate/consensus/verification prompts. "
            "Default is off. Error-correction prompt always includes test examples."
        ),
    )
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
    ap.add_argument("--ark_api_key", type=str, default=None, help="Overrides ARK_API_KEY.")
    ap.add_argument(
        "--ark_base_url",
        type=str,
        default=None,
        help=f"Overrides ARK_BASE_URL (default: {DEFAULT_ARK_BASE_URL}).",
    )
    ap.add_argument(
        "--las_anthropic_api_key",
        type=str,
        default=None,
        help="Optional LAS ANTHROPIC_API_KEY override.",
    )
    ap.add_argument(
        "--las_anthropic_base_url",
        type=str,
        default=None,
        help="Optional LAS ANTHROPIC_BASE_URL override.",
    )
    ap.add_argument(
        "--las_ark_api_key",
        type=str,
        default=None,
        help="Optional LAS ARK_API_KEY override.",
    )
    ap.add_argument(
        "--las_ark_base_url",
        type=str,
        default=None,
        help=f"Optional LAS ARK_BASE_URL override (default: {DEFAULT_ARK_BASE_URL}).",
    )
    ap.add_argument(
        "--gate_model_name",
        type=str,
        default=DEFAULT_GATE_MODEL_NAME,
        help="Base model name used by DistilBERT gate classifier.",
    )
    ap.add_argument(
        "--gate_checkpoint_path",
        type=Path,
        default=DEFAULT_GATE_CHECKPOINT_PATH,
        help="Path to DistilBERT checkpoint file.",
    )
    ap.add_argument(
        "--gate_max_length",
        type=int,
        default=DEFAULT_GATE_MAX_LENGTH,
        help="Tokenizer max length for DistilBERT gate scorer.",
    )
    ap.add_argument(
        "--weight_distillbert",
        type=float,
        default=DEFAULT_GATE_WEIGHT_DISTILBERT,
        help="Gate weight for DistilBERT score.",
    )
    ap.add_argument(
        "--weight_local_agreement",
        type=float,
        default=DEFAULT_GATE_WEIGHT_LOCAL_AGREEMENT,
        help="Gate weight for local agreement score.",
    )
    ap.add_argument(
        "--weight_adhere",
        type=float,
        default=DEFAULT_GATE_WEIGHT_ADHERE,
        help="Gate weight for spec adherence score.",
    )
    ap.add_argument(
        "--weight_history",
        type=float,
        default=DEFAULT_GATE_WEIGHT_HISTORY,
        help="Gate weight for agent history score.",
    )
    ap.add_argument(
        "--gate_threshold",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
        help="Gate threshold for invoking LAS.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        splits = list(DEFAULT_SPLITS)
    task_ids = _parse_task_ids(args.task_ids)

    anthropic_api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    anthropic_base_url = args.anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL")
    ark_api_key = args.ark_api_key or os.environ.get("ARK_API_KEY")
    ark_base_url = args.ark_base_url or os.environ.get("ARK_BASE_URL") or DEFAULT_ARK_BASE_URL

    las_llm_api = (args.las_llm_api or args.llm_api).strip().lower()
    las_model = (args.las_model or args.model).strip()
    las_max_tokens = int(args.las_max_tokens if args.las_max_tokens is not None else args.max_tokens)
    las_temperature = float(args.las_temperature if args.las_temperature is not None else args.temperature)
    las_anthropic_api_key = args.las_anthropic_api_key or anthropic_api_key
    las_anthropic_base_url = args.las_anthropic_base_url or anthropic_base_url
    las_ark_api_key = args.las_ark_api_key or ark_api_key
    las_ark_base_url = args.las_ark_base_url or os.environ.get("LAS_ARK_BASE_URL") or ark_base_url

    gate_config = GateConfig(
        weight_distillbert=float(args.weight_distillbert),
        weight_local_agreement=float(args.weight_local_agreement),
        weight_adhere=float(args.weight_adhere),
        weight_history=float(args.weight_history),
        threshold=float(args.gate_threshold),
    )

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_id = f"{timestamp_tag}_{_slugify_name(args.run_name)}"
    else:
        run_id = timestamp_tag
    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _ = create_llm_client(
        llm_api=args.llm_api,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        anthropic_api_key=anthropic_api_key,
        anthropic_base_url=anthropic_base_url,
        ark_api_key=ark_api_key,
        ark_base_url=ark_base_url,
    )
    _ = create_llm_client(
        llm_api=las_llm_api,
        max_tokens=las_max_tokens,
        temperature=las_temperature,
        anthropic_api_key=las_anthropic_api_key,
        anthropic_base_url=las_anthropic_base_url,
        ark_api_key=las_ark_api_key,
        ark_base_url=las_ark_base_url,
    )
    gate_scorer = get_or_create_gate_scorer(
        model_name=args.gate_model_name,
        checkpoint_path=Path(args.gate_checkpoint_path),
        max_length=args.gate_max_length,
    )
    history_tracker = AgentHistoryTracker()

    summary_rows: List[Dict[str, Any]] = []
    for split in splits:
        tasks = load_mbpp_sanitized_tasks(
            split=split,
            limit=args.limit_per_split,
            task_ids=task_ids,
        )
        split_dir = run_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        tasks_to_run: List[Tuple[Dict[str, Any], Path]] = []
        for task in tasks:
            task_csv = split_dir / f"task_{task['task_id']}.csv"
            if task_csv.exists() and not args.overwrite:
                continue
            tasks_to_run.append((task, task_csv))

        if not tasks_to_run:
            continue

        correct_count = 0
        processed_count = 0
        pbar = tqdm(total=len(tasks_to_run), desc=f"LAS_MBPP pipeline ({split})")
        max_workers = max(1, int(args.task_concurrency))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {}
            for task, task_csv in tasks_to_run:
                fut = pool.submit(
                    _run_task_pipeline_job,
                    run_id=run_id,
                    task=task,
                    llm_api=args.llm_api,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    anthropic_api_key=anthropic_api_key,
                    anthropic_base_url=anthropic_base_url,
                    ark_api_key=ark_api_key,
                    ark_base_url=ark_base_url,
                    las_llm_api=las_llm_api,
                    las_model=las_model,
                    las_max_tokens=las_max_tokens,
                    las_temperature=las_temperature,
                    las_anthropic_api_key=las_anthropic_api_key,
                    las_anthropic_base_url=las_anthropic_base_url,
                    las_ark_api_key=las_ark_api_key,
                    las_ark_base_url=las_ark_base_url,
                    gate_scorer=gate_scorer,
                    gate_config=gate_config,
                    history_tracker=history_tracker,
                    eval_timeout_sec=args.eval_timeout_sec,
                    tests_preview_limit=args.tests_preview_limit,
                    include_test_examples_in_previous_prompts=args.include_test_examples_in_previous_prompts,
                )
                future_map[fut] = (task, task_csv)

            for fut in as_completed(future_map):
                task, task_csv = future_map[fut]
                try:
                    rows, final_eval, final_code = fut.result()
                except Exception as exc:  # noqa: BLE001
                    err_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    print(
                        f"[TASK ERROR] split={split} task_id={task['task_id']} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    _print_llm_api_error(args.llm_api, args.model, err_text.strip())
                    failure_eval = {
                        "pass_count": 0,
                        "total_count": len(task.get("tests", []) or []),
                        "passed_all": False,
                        "exec_status": "Pipeline execution error.",
                        "exec_error": str(exc),
                        "failures": [],
                        "feedback_text": "Pipeline execution error.",
                        "eval_latency_sec": 0.0,
                    }
                    rows = [
                        _make_stage_row(
                            run_id=run_id,
                            task=task,
                            stage_index=1,
                            stage_name="pipeline_error",
                            agent_name="pipeline_runner",
                            model_name=args.model,
                            system_prompt="",
                            input_prompt="",
                            output_text="",
                            output_code="",
                            input_tokens=0,
                            output_tokens=0,
                            total_tokens=0,
                            latency_sec=0.0,
                            api_error=err_text.strip(),
                            eval_result=failure_eval,
                        )
                    ]
                    final_eval = failure_eval
                    final_code = ""

                pd.DataFrame(rows).to_csv(task_csv, index=False)
                processed_count += 1
                final_route_action = ""
                if rows:
                    final_route_action = safe_text(rows[-1].get("route_action")).strip()
                if final_eval.get("passed_all"):
                    correct_count += 1
                pass_rate = (correct_count / processed_count) if processed_count else 0.0
                pbar.set_postfix(pass1=f"{pass_rate:.2%}", processed=processed_count)
                pbar.update(1)

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
                        "final_route_action": final_route_action,
                        "task_csv": str(task_csv),
                    }
                )
        pbar.close()

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
                "llm_api": args.llm_api,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "las_llm_api": las_llm_api,
                "las_model": las_model,
                "las_max_tokens": las_max_tokens,
                "las_temperature": las_temperature,
                "task_concurrency": args.task_concurrency,
                "include_test_examples_in_previous_prompts": args.include_test_examples_in_previous_prompts,
                "eval_timeout_sec": args.eval_timeout_sec,
                "tests_preview_limit": args.tests_preview_limit,
                "gate_model_name": args.gate_model_name,
                "gate_checkpoint_path": str(Path(args.gate_checkpoint_path).resolve()),
                "gate_max_length": args.gate_max_length,
                "weight_distillbert": gate_config.weight_distillbert,
                "weight_local_agreement": gate_config.weight_local_agreement,
                "weight_adhere": gate_config.weight_adhere,
                "weight_history": gate_config.weight_history,
                "gate_threshold": gate_config.threshold,
                "output_dir": str(run_dir),
                "anthropic_base_url": anthropic_base_url or "",
                "ark_base_url": ark_base_url or "",
                "las_anthropic_base_url": las_anthropic_base_url or "",
                "las_ark_base_url": las_ark_base_url or "",
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
