"""Derived reports for committed evaluation task records."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import fmean, pstdev

import pandas as pd


def distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not values:
        return {"count": 0, **dict.fromkeys(("sum", "mean", "std", "min", "max", "p50", "p90", "p95", "p99"))}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lower = int(index)
        return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)

    return {
        "count": len(values), "sum": sum(values), "mean": fmean(values), "std": pstdev(values),
        "min": values[0], "max": values[-1],
        **{f"p{p}": percentile(p / 100) for p in (50, 90, 95, 99)},
    }


def coverage(calls, field, getter):
    values = [value for call in calls if (value := getter(call)) is not None]
    return {field: sum(values) if values else None, f"{field}_covered_calls": len(values)}


def call_metrics(calls):
    return {
        "logical_calls": len(calls), "failed_calls": sum(call.get("ok") is False for call in calls),
        "http_attempts": sum(len(call.get("http_attempts", [])) for call in calls),
        "http_covered_calls": sum(bool(call.get("http_attempts")) for call in calls),
        "http_retries": sum(max(0, len(call.get("http_attempts", [])) - 1) for call in calls),
        "finish_reason_covered_calls": sum(bool(call.get("finish_reason")) for call in calls),
        "finish_reasons": dict(Counter(call["finish_reason"] for call in calls if call.get("finish_reason"))),
        "failure_categories": dict(Counter(call["failure_category"] for call in calls if call.get("failure_category"))),
        **coverage(calls, "known_api_fees", lambda call: call.get("api_fee")),
        **coverage(calls, "known_total_tokens", lambda call: (call.get("provider_usage") or {}).get("total_tokens")),
        **coverage(calls, "known_input_tokens", lambda call: (call.get("provider_usage") or {}).get("input_tokens")),
        **coverage(calls, "known_output_tokens", lambda call: (call.get("provider_usage") or {}).get("output_tokens")),
        "call_seconds": distribution(call.get("wall_sec") for call in calls),
    }


def summarize(records, planned):
    calls = [call for task in records for call in task["calls"]]
    samples = [task["summary"] for task in records]
    by_model, by_role, by_function = defaultdict(list), defaultdict(list), defaultdict(list)
    for task in records:
        for call in task["calls"]:
            by_model[call["request"]["model"]].append((task, call))
            by_role[call.get("agent_name") or call.get("attribution") or "unattributed"].append((task, call))
        for span in task.get("spans", []):
            by_function[span["name"]].append(span)

    def aggregate(groups):
        output = {}
        for name, pairs in groups.items():
            used = {(task["item"]["split"], task["item"]["task_id"]): task for task, _ in pairs}
            output[name] = {
                "tasks_used": len(used),
                "tasks_correct_when_used": sum(task["summary"]["final_is_correct"] for task in used.values()),
                **call_metrics([call for _, call in pairs]),
            }
        return output

    correct = sum(row["final_is_correct"] for row in samples)
    sample_times = distribution(row.get("sample_wall_seconds", row.get("task_wall_sec")) for row in samples)
    summary = {
        "planned_tasks": planned, "completed_tasks": len(samples), "correct_tasks": correct,
        "failed_tasks": len(samples) - correct, "outcomes": dict(Counter(row["outcome"] for row in samples)),
        "invalid_tasks": sum(row["outcome"] == "invalid" for row in samples),
        "accuracy": correct / len(samples) if samples else None,
        "sample_coverage": len(samples) / planned if planned else None,
        "observed_samples_per_second": sample_times["count"] / sample_times["sum"] if sample_times["sum"] else None,
        "extraction_status": dict(Counter(row.get("extraction_status", "unknown") for row in samples)),
        "calls_by_phase": dict(Counter(call.get("attribution", "unattributed") for call in calls)),
        "answer_reuse_count": sum(row.get("answer_reuse_count", 0) for row in samples),
        "node_failures": sum(row.get("node_failures", 0) for row in samples),
        **call_metrics(calls),
        "model_metrics": aggregate(by_model), "role_metrics": aggregate(by_role),
        "function_metrics": {
            name: {
                "status_counts": dict(Counter(span["status"] for span in spans)),
                "inclusive_seconds": distribution(span.get("inclusive_seconds") for span in spans),
                "exclusive_seconds": distribution(span.get("exclusive_seconds") for span in spans),
            }
            for name, spans in by_function.items()
        },
        "latency_metrics": {
            "sample_wall_seconds": sample_times,
            **{key: distribution(row.get(key) for row in samples) for key in (
                "task_seconds", "graph_seconds", "scoring_seconds", "process_cpu_seconds", "event_sink_seconds",
            )},
            "completion_tokens_per_call_second": distribution(call.get("completion_tokens_per_call_second") for call in calls),
            "http_response_headers_seconds": distribution(
                span.get("inclusive_seconds") for call in calls for span in call.get("http_attempts", [])
            ),
        },
    }
    if any("evalplus_base_correct" in row for row in samples):
        summary["benchmark_scores"] = {
            "base_pass_at_1": sum(row.get("evalplus_base_correct", False) for row in samples) / len(samples),
            "plus_pass_at_1": sum(row.get("evalplus_plus_correct", False) for row in samples) / len(samples),
        }
    return summary


def chrome_trace(spans, **metadata):
    events = []
    for span in spans:
        at = datetime.fromisoformat(span["started_at"]).timestamp() * 1e6
        event = {
            "name": span["name"], "cat": span["kind"], "pid": span["pid"], "tid": span["thread_id"],
            "ts": at, "args": {
                "span_id": span["span_id"], "parent_span_id": span["parent_span_id"],
                "trace_id": span["trace_id"], "task_attempt": span["task_attempt"], "task_id": span["task_id"],
                "status": span.get("status", "incomplete"), **span["attributes"],
            },
        }
        duration = span.get("inclusive_seconds")
        if duration is None:
            event.update(ph="i", s="t")
            event["args"]["observed_seconds"] = span.get("observed_seconds")
        else:
            event.update(ph="X", dur=duration * 1e6)
        events.append(event)
    return {"traceEvents": sorted(events, key=lambda event: event["ts"]), "displayTimeUnit": "ms", "metadata": metadata}


def flatten(record, prefix=""):
    for name, value in record.items():
        key = prefix + name
        if isinstance(value, dict):
            yield from flatten(value, key + ".")
        else:
            yield key, value


def excel_cell(value):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    if not isinstance(value, str):
        return value
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    if value.lstrip().startswith(("=", "+", "-", "@")):
        value = "'" + value
    return value[:32767]


def workbook(path, report):
    tasks = report["tasks"]
    summary = report["summary"]
    tables = {name: [] for name in ("tasks", "agents", "calls", "events", "rounds", "spans", "http_attempts")}
    for task in tasks:
        keys = {key: task.get(key) for key in ("task_attempt", "trace_id", "evaluation_index")}
        keys.update(split=task["item"]["split"], task_id=task["item"]["task_id"], run_id=report["metadata"]["run_id"])
        tables["tasks"].append({
            **keys, **task["summary"], "query": task["task_description"], "reference_answer": task.get("reference_answer"),
            "raw_response": task.get("raw_response"), "extracted_answer": task["summary"]["final_code"],
        })
        names = sorted(set(task["execution"]["observed_nodes"]) | {f"debate_agent_{name}" for name in "ABC"})
        for agent in names:
            assigned_calls = [call for call in task["calls"] if call.get("agent_name") == agent]
            model = task["execution"]["observed_nodes"].get(agent)
            if model is None and agent.startswith("debate_agent_"):
                model = report["metadata"]["settings"].get("model")
            execution = task["execution"]
            final_agent = "final_output_unit" if execution["final_node"] == "final_output_unit" else execution["final_source_agent"]
            tables["agents"].append({
                **keys, "agent": agent, "model": model, "final_node": final_agent == agent,
                "executions": sum(span["kind"] == "node" and span["attributes"].get("agent") == agent for span in task.get("spans", [])),
                **dict(flatten(call_metrics(assigned_calls))),
            })
        for call in task["calls"]:
            tables["calls"].append({**keys, **dict(flatten({key: v for key, v in call.items() if key != "http_attempts"}))})
            for attempt in call.get("http_attempts", []):
                tables["http_attempts"].append({**keys, **dict(flatten(attempt))})
        for event in task.get("events", []):
            tables["events"].append({**keys, **dict(flatten(event))})
            if event["event"] == "round":
                tables["rounds"].append({**keys, **event})
        for span in task.get("spans", []):
            tables["spans"].append({**keys, **dict(flatten(span))})
    for name in ("model_metrics", "role_metrics", "function_metrics", "latency_metrics"):
        tables[name] = [{"name": key, **dict(flatten(values))} for key, values in summary[name].items()]
    tables["run_summary"] = [
        {"field": name, "value": value}
        for name, value in flatten({key: value for key, value in report.items() if key != "tasks"})
    ]
    rollovers = {"calls", "events", "rounds", "spans", "http_attempts"}
    max_rows = 1_048_575
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, rows in tables.items():
            if name not in rollovers:
                pd.DataFrame([{key: excel_cell(val) for key, val in row.items()} for row in rows]).to_excel(writer, sheet_name=name, index=False)
                continue
            for idx, offset in enumerate(range(0, max(len(rows), 1), max_rows), 1):
                chunk = rows[offset:offset + max_rows]
                pd.DataFrame([{key: excel_cell(val) for key, val in row.items()} for row in chunk]).to_excel(writer, sheet_name=f"{name}_{idx}", index=False)
