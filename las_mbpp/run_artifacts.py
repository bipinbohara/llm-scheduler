"""Durable run records and derived exports shared by the two experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import fcntl
import pandas as pd


def add_run_arguments(parser) -> None:
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume an existing run under this dataset's output directory.")
    parser.add_argument("--excel", action="store_true", help="Refresh the Excel workbook after each completion.")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_url(url: str | None) -> str:
    parts = urlsplit(url or "")
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


def url_secrets(*urls):
    for url in urls:
        parts = urlsplit(url or "")
        if parts.password:
            yield unquote(parts.password)
        for _, value in parse_qsl(parts.query):
            yield value


def redactor(secrets):
    secrets = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(secret) for secret in secrets)) if secrets else None

    def redact(value):
        if isinstance(value, str) and pattern:
            return pattern.sub("[REDACTED]", value)
        if isinstance(value, dict):
            return {key: redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [redact(item) for item in value]
        return value

    return redact


def identity(task: dict) -> dict:
    metadata = {
        key: task.get(key)
        for key in (
            "description", "prompt_stub", "function_name", "tests", "evaluation_tests",
            "setup_code", "reference_code", "evalplus_version", "release_hash", "atol",
        )
    }
    return {
        "split": task["split"],
        "task_id": task["task_id"],
        "benchmark_id": task.get("benchmark_id"),
        "sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
    }


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data) -> None:
    atomic_write(path, lambda temp: temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))


class TracedClient:
    def __init__(self, client, component: str) -> None:
        self.client = client
        self.component = component
        self.calls: list[dict] = []
        self.lock = Lock()

    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> dict:
        started = now()
        clock = time.perf_counter()
        with self.lock:
            call = {
                "call_id": f"{self.component}-{len(self.calls) + 1}",
                "component": self.component,
                "started_utc": started,
                "request": {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt},
            }
            self.calls.append(call)
        try:
            response = self.client.call(system_prompt=system_prompt, user_prompt=user_prompt, model=model)
        except Exception as exc:
            call.update(error=f"{type(exc).__name__}: {exc}", ok=False)
            raise
        else:
            call.update(
                response=response.get("output_text", ""), usage=response.get("usage"),
                ok=response.get("ok"), error=response.get("error"), finish_reason=response.get("finish_reason"),
                http_status=response.get("http_status"), attempts=response.get("attempts"),
                usage_reported=response.get("usage_reported"),
            )
            return response
        finally:
            call["wall_sec"] = time.perf_counter() - clock
            attempts = call.get("attempts")
            call["retries"] = attempts - 1 if attempts is not None else None


def associate_calls(calls: list[dict], stages: list[dict]) -> None:
    remaining = list(calls)
    for call in calls:
        call["attribution"] = "routing" if call["component"] == "router" else "unattributed"
    for row in stages:
        stage = row["stage_name"]
        candidates = [
            ("workflow", row.get("agent_input"), row.get("model")),
            ("router", row.get("las_prompt"), row.get("las_model")),
        ]
        for component, prompt, model in candidates:
            for i, call in enumerate(remaining):
                req = call["request"]
                if (component, prompt, model) != (call["component"], req["user_prompt"], req["model"]):
                    continue
                category = {
                    "debate_initial": "worker",
                    "debate_revision": "collaboration",
                    "debate_final": "collaboration",
                    "consensus": "aggregation",
                }.get(stage, "extra_processing")
                call.update(
                    stage_index=row["stage_index"], stage_name=stage, agent_name=row["agent_name"],
                    attribution="routing" if component == "router" else category,
                )
                row[f"{component}_call_id"] = call["call_id"]
                remaining.pop(i)
                break


def execution_info(rows: list[dict], calls: list[dict]) -> dict:
    debate_stages = {"debate_initial", "debate_final"}
    observed = {}
    workers = set()
    rounds = set()
    for row in rows:
        stage, agent = row["stage_name"], row["agent_name"]
        if stage == "pipeline_error":
            continue
        node = stage if stage == "final_output_unit" else agent
        observed[node] = row["model"] if row.get("workflow_call_id") else None
        if stage in debate_stages:
            workers.add(agent)
            rounds.add(stage)
    router = next((call for call in calls if call["component"] == "router"), None)
    if router:
        observed["las_router"] = router["request"]["model"]

    producer = rows[-1] if rows else {}
    final_node = producer.get("stage_name")
    if final_node == "final_output_unit":
        source = (producer["route_source_stage"], producer["route_source_agent"])
        producer = next((row for row in rows if (row["stage_name"], row["agent_name"]) == source), {})
        if producer.get("stage_name") == "consensus":
            chosen = producer.get("selected_agent")
            producer = next((row for row in rows if row["stage_name"] == "debate_final"
                             and row["agent_name"] == f"debate_agent_{chosen}"), producer)
    return {
        "collaboration_mode": "debate", "observed_rounds": len(rounds),
        "allocated_workers": 3, "participating_workers": sorted(workers),
        "observed_nodes": observed, "total_observed_nodes": len(observed),
        "stage_invocations": len(rows), "final_node": final_node,
        "final_source_agent": producer.get("agent_name"), "final_source_model": producer.get("model"),
    }


class RunArtifacts:
    def __init__(self, args, settings: dict, tasks: list[dict], secrets=()) -> None:
        self.redact = redactor(secrets)
        self.settings = settings
        self.items = [identity(task) for task in tasks]
        self.resume = args.resume
        self.excel = args.excel
        self.records: dict[tuple, dict] = {}
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite cannot be combined.")
        if self.excel:
            try:
                import openpyxl  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('Excel output requires: pip install -e ".[excel]"') from exc
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.run_name or "").strip("._-")
        self.run_id = args.resume or datetime.now().strftime("%Y%m%d_%H%M%S_%f") + (f"_{label}" if label else "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_id):
            raise ValueError("--resume must be a run ID, without directory components.")
        self.path = Path(args.output_dir).resolve() / args.dataset / self.run_id
        self.config_path = self.path / "run_config.json"

    def __enter__(self):
        if self.resume and not self.config_path.is_file():
            raise FileNotFoundError(f"No saved run configuration at {self.config_path}. Check --dataset and --output_dir.")
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                saved = json.loads(self.config_path.read_text())
                if saved["settings"] != self.settings or saved["selected_items"] != self.items:
                    raise ValueError("Saved inputs or settings differ. Re-run with the original flags, dataset and checkpoint.")
                for item in self.items:
                    key = (item["split"], item["task_id"])
                    path = self.record_path(key)
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record["item"] != item or record["summary"]["run_id"] != self.run_id:
                            raise ValueError(f"Completion record does not belong to this run: {path}")
                        self.records[key] = record
                        if not path.with_suffix(".csv").exists():
                            self.export_task(key, record)
            else:
                write_json(self.config_path, {
                    **self.settings, "run_id": self.run_id, "timestamp_utc": now(),
                    "output_dir": str(self.path), "settings": self.settings,
                    "selected_items": self.items,
                })
            self.refresh()
            self.log(f"{'Resumed' if self.resume else 'Started'} {self.run_id} | {len(self.records)}/{len(self.items)} complete")
            if self.settings.get("checkpoint"):
                self.log("Gate: " + json.dumps(self.settings["checkpoint"]))
            return self
        except BaseException:
            self.lock.close()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc:
                self.log(f"Interrupted or failed: {type(exc).__name__}: {exc}")
            else:
                self.log(f"Finished {self.run_id} | output: {self.path}")
        finally:
            self.lock.close()

    def log(self, message: str) -> None:
        line = f"{now()} {self.redact(message)}"
        print(line, flush=True)
        with (self.path / "run.log").open("a", encoding="utf-8") as log:
            log.write(line + "\n")

    def record_path(self, key: tuple) -> Path:
        split, task_id = key
        return self.path / split / f"task_{task_id}.json"

    def completed(self, task: dict) -> bool:
        return (task["split"], task["task_id"]) in self.records

    def export_task(self, key, record) -> None:
        atomic_write(self.record_path(key).with_suffix(".csv"), lambda temp: pd.DataFrame(record["stages"]).to_csv(temp, index=False))

    def save(self, task, rows, final, code, calls, elapsed, extra) -> None:
        key = (task["split"], task["task_id"])
        associate_calls(calls, rows)
        status = final["exec_status"]
        if not code.strip() or status in {"Compilation failed.", "Pipeline execution error."} or status.startswith("Official EvalPlus base suite: invalid"):
            outcome = "invalid"
        elif final["passed_all"] and extra.get("evalplus_plus_correct", True):
            outcome = "correct"
        else:
            outcome = "incorrect"
        summary = {
            "run_id": self.run_id, "dataset": self.settings["dataset"], "split": key[0], "task_id": key[1],
            "benchmark_id": task.get("benchmark_id"), "function_name_from_tests": task["function_name"],
            "final_is_correct": outcome == "correct", "outcome": outcome,
            "tests_passed": final["pass_count"], "tests_total": final["total_count"],
            "final_exec_status": status, "final_code": code,
            "final_route_action": rows[-1].get("route_action", "") if rows else "",
            "task_wall_sec": elapsed, "logical_calls": len(calls),
            "known_attempts": sum(call.get("attempts") or 0 for call in calls),
            "calls_with_unknown_attempts": sum(call.get("attempts") is None for call in calls),
            "input_tokens": sum((call.get("usage") or {}).get("input_tokens", 0) for call in calls),
            "output_tokens": sum((call.get("usage") or {}).get("output_tokens", 0) for call in calls),
            "total_tokens": sum((call.get("usage") or {}).get("total_tokens", 0) for call in calls),
            "task_csv": str(self.record_path(key).with_suffix(".csv")),
            **extra,
        }
        record = self.redact({
            "summary": summary, "task_description": task["description"],
            "reference_answer": task.get("reference_code"), "final_evaluation": final,
            "stages": rows, "calls": calls, "execution": execution_info(rows, calls),
        })
        record["item"] = identity(task)
        record["summary"]["run_id"] = self.run_id
        write_json(self.record_path(key), record)
        self.records[key] = record
        self.export_task(key, record)
        self.refresh()
        self.log(
            f"{key[0]}/{key[1]} {outcome} | calls={len(calls)} tokens={summary['total_tokens']} "
            f"wall={elapsed:.3f}s | {len(self.records)}/{len(self.items)} complete"
        )

    def refresh(self) -> None:
        ordered = ((item["split"], item["task_id"]) for item in self.items)
        summaries = [self.records[key]["summary"] for key in ordered if key in self.records]
        if not summaries:
            return
        frame = pd.DataFrame(summaries)
        atomic_write(self.path / "summary.csv", lambda temp: frame.to_csv(temp, index=False))
        if self.excel:
            # Excel cells have a 32,767-character limit. Full text stays in CSV/JSON.
            def excel_value(cell):
                if not isinstance(cell, str):
                    return cell
                cell = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cell)
                if cell[:1] in ("=", "+", "-", "@"):
                    cell = "'" + cell
                return cell[:32767]

            excel = frame.apply(lambda column: column.map(excel_value))
            atomic_write(self.path / "summary.xlsx", lambda temp: excel.to_excel(temp, index=False))
