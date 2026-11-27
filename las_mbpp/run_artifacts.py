"""Durable run records and derived exports shared by the two experiment runners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import json
import os
import platform
import random
import re
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event, Lock, RLock, Thread
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import fcntl
import pandas as pd

from las_mbpp import journals
from las_mbpp.experiment_report import call_metrics, chrome_trace, excel_cell, summarize, workbook
from las_mbpp.local_models import LOCAL_MODELS, positive_int
from las_mbpp.telemetry import CALL, CURRENT, NODE, PARENT, TaskTrace, finish_http_attempts, span


def add_run_arguments(parser) -> None:
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume an existing run under this dataset's output directory.")
    parser.add_argument("--excel", "--export_excel", dest="excel", action="store_true", help="Export the analysis workbook.")
    parser.add_argument("--result_file", type=Path, help="Explicit consolidated JSON path. Companion files share its stem.")
    parser.add_argument("--export_every", type=positive_int, default=1, help="Write full snapshots every N committed tasks.")
    parser.add_argument("--heartbeat_seconds", type=float, default=60, help="Heartbeat interval in seconds. Zero disables it.")


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


CALL_PHASES = {
    "debate_initial": "worker", "debate_final": "collaboration", "consensus": "aggregation",
    "reflection": "extra_processing", "verification": "extra_processing",
    "error_correction": "extra_processing", "router_decision": "routing",
}


class TracedClient:
    def __init__(self, client, component: str) -> None:
        self.client = client
        self.component = component
        self.calls: list[dict] = []
        self.lock = Lock()

    def call(self, *, system_prompt: str, user_prompt: str, model: str) -> dict:
        trace = CURRENT.get()
        stage, agent = NODE.get()
        clock = time.perf_counter()
        request = {"model": model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        with self.lock:
            call_id = f"{self.component}-{len(self.calls) + 1}"
            if trace:
                call_id = f"{trace.fields['task_attempt']}/{call_id}"
            call = {
                "call_id": call_id, "component": self.component, "started_utc": now(),
                "request": request, "stage_name": stage, "agent_name": agent,
                "backend": self.client.__class__.__name__,
                "attribution": CALL_PHASES.get(stage, "unattributed"),
                "tier": LOCAL_MODELS[model].tier if model in LOCAL_MODELS else None,
                "parameter": getattr(self.client, "generation_parameters", {
                    "max_tokens": getattr(self.client, "max_tokens", None),
                    "temperature": getattr(self.client, "temperature", None),
                }),
                "prompt_chars": len(system_prompt) + len(user_prompt), "message_count": 2,
                "prompt_sha256": hashlib.sha256(journals.json_line(request)).hexdigest(),
                "http_attempts": [],
            }
            self.calls.append(call)
        if trace:
            settings = trace.run.settings
            prefix = "las_" if self.component == "router" else ""
            endpoint = next((settings.get(prefix + name + "_base_url") for name in ("local", "anthropic", "ark")
                             if settings.get(prefix + name + "_base_url")
                             and name in call["backend"].lower()), None)
            call["endpoint_sha256"] = hashlib.sha256(endpoint.encode()).hexdigest() if endpoint else None
            call["backend"] = settings.get(prefix + "llm_api", call["backend"])
        with span("llm.generate", "llm_call", call_id=call["call_id"], model=model) as attributes:
            call["span_id"] = PARENT.get()
            parent_call = CALL.get()
            call["parent_call_id"] = parent_call[1]["call_id"] if parent_call else None
            context = (trace, call, {})
            token = CALL.set(context if trace else None)
            if trace:
                trace.event("llm_call_started", call=call, call_id=call["call_id"])
            try:
                response = self.client.call(system_prompt=system_prompt, user_prompt=user_prompt, model=model)
            except BaseException as exc:
                call.update(error=f"{type(exc).__name__}: {exc}", ok=False,
                            failure_category="interrupted" if isinstance(exc, KeyboardInterrupt) else "client_exception")
                raise
            else:
                usage = response.get("provider_usage")
                call.update(
                    response=response.get("output_text", ""), usage=response.get("usage"),
                    provider_usage=usage, ok=response.get("ok"), error=response.get("error"),
                    finish_reason=response.get("finish_reason"), http_status=response.get("http_status"),
                    reported_model=response.get("reported_model"), response_id=response.get("response_id"),
                    api_fee=response.get("api_fee"), ttft_seconds=None,
                    reasoning_only=response.get("reasoning_only"), failure_category=response.get("failure_category"),
                )
                response["_trace_call_id"] = call["call_id"]
                return response
            finally:
                finish_http_attempts(context)
                CALL.reset(token)
                call["finished_utc"] = now()
                call["wall_sec"] = time.perf_counter() - clock
                call["response_chars"] = len(call.get("response") or "")
                call["response_sha256"] = hashlib.sha256(call["response"].encode()).hexdigest() if "response" in call else None
                call["attempts"] = len(call["http_attempts"]) or None
                call["retries"] = call["attempts"] - 1 if call["attempts"] is not None else None
                tokens = (call.get("provider_usage") or {}).get("output_tokens")
                call["completion_tokens_per_call_second"] = tokens / call["wall_sec"] if tokens is not None and call["wall_sec"] else None
                if not call.get("failure_category"):
                    error = (call.get("error") or "").lower()
                    finish = call.get("finish_reason")
                    if finish in {"length", "max_tokens", "max_output_tokens"}:
                        call["failure_category"] = "generation_limit"
                    elif "context" in error and ("length" in error or "window" in error):
                        call["failure_category"] = "context_limit"
                    elif call.get("reasoning_only"):
                        call["failure_category"] = "reasoning_only"
                    elif not call["response_chars"] and (call.get("ok") or "no answer text" in error):
                        call["failure_category"] = "empty_response"
                    elif not call.get("ok"):
                        call["failure_category"] = "provider_error"
                attributes["_status"] = "ok" if call.get("ok") else "error"
                if trace:
                    trace.event("llm_call", call=call, call_id=call["call_id"])


def associate_calls(calls: list[dict], stages: list[dict]) -> None:
    by_id = {call["call_id"]: call for call in calls}
    for row in stages:
        workflow_call = by_id.get(row.get("workflow_call_id"))
        if workflow_call:
            workflow_call["stage_index"] = row["stage_index"]
        router_call = by_id.get(row.get("router_call_id"))
        if router_call:
            router_call.update(stage_index=row["stage_index"], source_stage=row["stage_name"],
                               source_agent=row["agent_name"])


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
    def __init__(self, args, secrets=()) -> None:
        self.redact = redactor(secrets)
        self.dataset = args.dataset
        self.settings: dict = {}
        self.items: list[dict] = []
        self.indices = {}
        self.resume = args.resume
        self.excel = args.excel
        self.export_every = args.export_every
        self.period = args.heartbeat_seconds
        self.records: dict[tuple, dict] = {}
        self.guard = RLock()
        self.activity = {}
        self.stop = Event()
        self.heartbeat = None
        self.opened_at = now()
        self.started = time.monotonic()
        self.status = "initializing"
        self.configured = False
        self.last_commit = None
        self.router_snapshot = None
        self.device = None
        self.dataset_release_hashes = []
        self.initial_rng = random.getstate()
        self.resume_history = []
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite cannot be combined.")
        if not math.isfinite(self.period) or self.period < 0:
            raise ValueError("--heartbeat_seconds must be finite and nonnegative.")
        if self.excel:
            try:
                import openpyxl  # noqa: F401
            except ImportError as exc:
                raise RuntimeError('Excel output requires: pip install -e ".[excel]"') from exc
        label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.run_name or "").strip("._-")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_id = args.resume or stamp + (f"_{label}" if label else "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_id):
            raise ValueError("--resume must be a run ID, without directory components.")
        result = args.result_file.resolve() if args.result_file else None
        if result and result.suffix != ".json":
            raise ValueError("--result_file must end in .json.")
        self.path = result.parent if result else Path(args.output_dir).resolve() / args.dataset / self.run_id
        self.stem = result.stem if result else "experiment"
        if result and self.stem == "run_config":
            raise ValueError("--result_file cannot be named run_config.json.")
        self.config_path = self.path / "run_config.json"
        self.heartbeat_index = 0
        self.journal_paths = {kind: self.file(f"{kind}.jsonl") for kind in ("tasks", "events", "spans", "heartbeat")}

    def file(self, suffix):
        return self.path / f"{self.stem}.{suffix}"

    def __enter__(self):
        if self.resume and not self.config_path.is_file():
            raise FileNotFoundError(f"No saved run configuration at {self.config_path}. Check --dataset and --output_dir.")
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = journals.open_writer_lock(self.path / ".run.lock")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.experiment_lock = journals.open_writer_lock(self.file("lock"))
            fcntl.flock(self.experiment_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not self.resume and (self.config_path.exists() or self.file("json").exists()):
                raise FileExistsError(f"Run output already exists at {self.path}. Use --resume or choose another directory.")
            if self.resume:
                saved = json.loads(self.config_path.read_text())
                if saved["run_id"] != self.run_id or saved.get("result_stem", "experiment") != self.stem:
                    raise ValueError("Run ID or result filename differs from the saved run.")
            self.file("log").touch()
            for name, path in self.journal_paths.items():
                if name == "tasks":
                    if not self.resume:
                        path.touch()
                    continue  # Preserve detection of runs created before the task journal existed.
                rows, _ = journals.recover(path, repair=True, keep=name == "heartbeat")
                if name == "heartbeat" and rows:
                    self.heartbeat_index = rows[-1]["heartbeat_index"] + 1
                if name != "heartbeat" or self.period:
                    path.touch()
            journals.sync_directory(self.path)
            self.log(f"{'Resuming' if self.resume else 'Starting'} {self.run_id} | output: {self.path}")
            if not self.resume:
                self.refresh_progress()
                self.refresh_snapshot()
            if self.period:
                self.write_heartbeat("startup")
                self.heartbeat = Thread(target=self.heartbeat_loop, name="run-heartbeat", daemon=True)
                self.heartbeat.start()
            return self
        except BaseException:
            if hasattr(self, "experiment_lock"):
                self.experiment_lock.close()
            self.lock.close()
            raise

    def configure(self, settings, tasks):
        self.settings = settings
        self.dataset_release_hashes = sorted({task["release_hash"] for task in tasks if task.get("release_hash")})
        self.items = [identity(task) for task in tasks]
        self.indices = {(item["split"], item["task_id"]): index for index, item in enumerate(self.items)}
        journal = self.journal_paths["tasks"]
        checkpoint = self.file("checkpoint.json")
        if self.resume:
            saved = json.loads(self.config_path.read_text())
            if saved["settings"] != self.settings or saved["selected_items"] != self.items:
                raise ValueError("Saved inputs or settings differ. Re-run with the original flags, dataset and checkpoint.")
            self.opened_at = saved["timestamp_utc"]
            self.resume_history = saved.get("resume_history", []) + [{"resumed_at": now()}]
            if checkpoint.exists():
                state = json.loads(checkpoint.read_text())
                if state["resume_signature"] != self.signature():
                    raise ValueError("Saved checkpoint resume signature differs.")
                self.initial_rng = state["initial_rng_state"]
                self.router_snapshot = state.get("router_snapshot")
            if journal.exists():
                records, _ = journals.recover(journal, repair=True)
            elif saved.get("result_stem"):
                raise FileNotFoundError(f"The authoritative task journal is missing: {journal}")
            else:
                records = [json.loads(path.read_text()) for key in self.indices if (path := self.record_path(key)).exists()]
                atomic_write(journal, lambda path: path.write_bytes(b"".join(journals.json_line(row) for row in records)))
            if saved.get("result_stem") and records and settings.get("checkpoint") and not self.router_snapshot:
                raise ValueError("The saved LAS run is missing its router snapshot identity.")
            for record in records:
                item = record["item"]
                key = (item["split"], item["task_id"])
                if (key not in self.indices or item != self.items[self.indices[key]] or key in self.records
                        or record["summary"]["run_id"] != self.run_id):
                    raise ValueError(f"Unexpected or duplicate task in {journal}: {key}")
                with self.guard:
                    self.records[key] = record
                if not self.record_path(key).exists():
                    write_json(self.record_path(key), record)
                if not self.record_path(key).with_suffix(".csv").exists():
                    self.export_task(key, record)
            self.last_commit = records[-1].get("committed_at") if records else None
            rng = records[-1].get("rng_state", self.initial_rng) if records else self.initial_rng
            random.setstate((rng[0], tuple(rng[1]), rng[2]))
        else:
            journal.touch()
            journals.sync_directory(self.path)
        write_json(self.config_path, {
            **self.settings, "run_id": self.run_id, "timestamp_utc": self.opened_at,
            "output_dir": str(self.path), "settings": self.settings, "selected_items": self.items,
            "result_stem": self.stem, "resume_history": self.resume_history,
        })
        self.log("Effective settings " + json.dumps(self.settings))
        with self.guard:
            self.configured = True
            self.status = "running"
        self.refresh_progress()
        self.refresh_snapshot()

    def signature(self):
        return hashlib.sha256(json.dumps([self.settings, self.items], sort_keys=True).encode()).hexdigest()

    def save_router(self, write_weights, load_weights):
        from las_mbpp.checkpoints import checkpoint_identity

        path = self.file("router.pth")
        if self.router_snapshot is not None:
            actual = checkpoint_identity(path, 0)
            if actual["sha256"] != self.router_snapshot["sha256"]:
                raise ValueError(f"Saved router snapshot has a different SHA-256: {path}")
            load_weights(path)
        else:
            # Snapshot before any task can start. This also upgrades legacy runs on resume.
            atomic_write(path, write_weights)
            data = checkpoint_identity(path, 0)
            self.router_snapshot = {"file": path.name, "sha256": data["sha256"], "bytes": data["bytes"]}
        self.log("Router snapshot " + json.dumps(self.router_snapshot))
        self.refresh_progress()
        self.refresh_snapshot()

    def __exit__(self, exc_type, exc, tb):
        with self.guard:
            self.status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed" if exc_type else "completed"
        try:
            self.stop.set()
            if self.heartbeat:
                self.heartbeat.join()
                self.write_heartbeat("exit")
            if exc_type:
                self.log(f"Interrupted or failed: {exc_type.__name__}: {exc}", "ERROR")
                self.log("".join(traceback.format_exception(exc_type, exc, tb)).rstrip(), "DEBUG")
            else:
                self.log(f"Finished {self.run_id} | output: {self.path}")
            if self.configured or not self.resume:
                self.refresh_progress()
                self.refresh_snapshot()
                self.log("Final metrics " + json.dumps({
                    key: self.summary[key] for key in ("completed_tasks", "correct_tasks", "accuracy")
                }))
        except Exception as export_error:
            if exc_type is None:
                raise
            self.log(f"Final export failed: {export_error}", "ERROR")
        finally:
            self.experiment_lock.close()
            self.lock.close()

    def log(self, message, level="INFO"):
        line = f"{now()} {level} {self.redact(message)}"
        with self.guard:
            with self.file("log").open("a", encoding="utf-8") as log:
                log.write(line + "\n")
            if level != "DEBUG":
                print(line, flush=True)
                with (self.path / "run.log").open("a", encoding="utf-8") as log:
                    log.write(line + "\n")

    def emit(self, record):
        row = self.redact(record)
        with self.guard:
            journals.append(self.journal_paths["events"], row)
            if row["event"] in {"llm_call_started", "llm_call", "task_started", "task_saved",
                               "router_decision", "round", "stage_result", "task_error", "task_interrupted"}:
                self.log(json.dumps(row, ensure_ascii=False), "DEBUG")
            if row["event"] == "llm_call" and row["call"].get("failure_category") in {"generation_limit", "context_limit"}:
                self.log(f"{row['call']['failure_category']} on {row['call_id']}", "WARNING")
        return row

    def emit_span(self, record):
        row = self.redact({"schema_version": 1, **record})
        with self.guard:
            journals.append(self.journal_paths["spans"], row)
        return row

    @contextmanager
    def task(self, task):
        key = (task["split"], task["task_id"])
        trace = TaskTrace(self, task, self.indices[key])
        token = CURRENT.set(trace)
        with self.guard:
            self.activity[trace.fields["task_attempt"]] = {**trace.fields, "spans": {}}
        trace.event("task_started")
        try:
            with span("evaluation.sample", "sample"):
                yield trace
        except BaseException as exc:
            trace.event("task_interrupted" if isinstance(exc, KeyboardInterrupt) else "task_error",
                        error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            trace.cpu_seconds = time.process_time() - trace.cpu
            with self.guard:
                self.activity.pop(trace.fields["task_attempt"], None)
            CURRENT.reset(token)

    def active_span(self, record):
        with self.guard:
            self.activity[record["task_attempt"]]["spans"][record["span_id"]] = record

    def inactive_span(self, record):
        with self.guard:
            self.activity[record["task_attempt"]]["spans"].pop(record["span_id"], None)

    def heartbeat_loop(self):
        deadline = self.started + self.period
        while not self.stop.wait(max(0.0, deadline - time.monotonic())):
            try:
                self.write_heartbeat("interval")
            except Exception as exc:
                self.log(f"Heartbeat write failed: {exc}", "ERROR")
            elapsed = time.monotonic() - deadline
            deadline += self.period * max(1, math.floor(elapsed / self.period) + 1)

    def write_heartbeat(self, trigger):
        with self.guard:
            active = []
            clock = time.perf_counter_ns()
            for item in list(self.activity.values())[:20]:
                latest = max(item["spans"].values(), key=lambda row: row["start_monotonic_ns"], default=None)
                active.append({
                    "task_id": item["task_id"], "split": item["split"], "task_attempt": item["task_attempt"],
                    "active_stage": latest["name"] if latest else None,
                    "active_stage_seconds": (clock - latest["start_monotonic_ns"]) / 1e9 if latest else None,
                })
            last = next(reversed(self.records.values()), None)
            current = active[0] if len(self.activity) == 1 else {}
            journals.append(self.journal_paths["heartbeat"], {
                "schema_version": 1, "event": "run_heartbeat", "trigger": trigger, "timestamp": now(),
                "run_id": self.run_id, "heartbeat_index": self.heartbeat_index, "phase": "evaluation",
                "status": self.status, "elapsed_seconds": time.monotonic() - self.started,
                "completed_tasks": len(self.records), "planned_tasks": len(self.items) if self.configured else None,
                "failed_tasks": sum(not task["summary"]["final_is_correct"] for task in self.records.values()),
                "last_committed_task_id": last["item"]["task_id"] if last else None,
                "last_commit_at": self.last_commit, "in_flight_count": len(self.activity), "active_tasks": active,
                "active_task_id": current.get("task_id"), "task_attempt": current.get("task_attempt"),
                "active_stage": current.get("active_stage"), "active_stage_seconds": current.get("active_stage_seconds"),
            })
            self.heartbeat_index += 1

    def record_path(self, key: tuple) -> Path:
        split, task_id = key
        return self.path / split / f"task_{task_id}.json"

    def completed(self, task: dict) -> bool:
        return (task["split"], task["task_id"]) in self.records

    def export_task(self, key, record) -> None:
        atomic_write(self.record_path(key).with_suffix(".csv"), lambda temp: pd.DataFrame(record["stages"]).to_csv(temp, index=False))

    def save(self, task, rows, final, code, calls, elapsed, extra, trace) -> None:
        key = (task["split"], task["task_id"])
        associate_calls(calls, rows)
        status = final["exec_status"]
        if not code.strip() or status in {"Compilation failed.", "Pipeline execution error."} or status.startswith("Official EvalPlus base suite: invalid"):
            outcome = "invalid"
        elif final["passed_all"] and extra.get("evalplus_plus_correct", True):
            outcome = "correct"
        else:
            outcome = "incorrect"
        timings = {span["name"]: span.get("inclusive_seconds") for span in trace.spans}
        final_call_id = rows[-1].get("workflow_call_id") if rows else None
        if rows and rows[-1].get("stage_name") == "final_output_unit":
            source = rows[-1]
            selected = next((row for row in rows if row["stage_name"] == source["route_source_stage"]
                             and row["agent_name"] == source["route_source_agent"]), {})
            if selected.get("stage_name") == "consensus":
                agent = f"debate_agent_{selected['selected_agent']}"
                selected = next((row for row in rows if row["stage_name"] == "debate_final"
                                 and row["agent_name"] == agent), selected)
            final_call_id = selected.get("workflow_call_id")
        raw_response = next((call.get("response") for call in calls if call["call_id"] == final_call_id), None)
        summary = {
            "sample_wall_seconds": timings.get("evaluation.sample", elapsed),
            "task_seconds": timings.get("graph.execute"),
            "graph_seconds": timings.get("graph.execute"),
            "scoring_seconds": timings.get("evaluation.final_scoring"),
            "final_base_evaluation_seconds": final.get("eval_latency_sec"),
            "reference_preparation_seconds": task.get("reference_preparation_seconds"),
            "process_cpu_seconds": trace.cpu_seconds,
            "event_sink_seconds": trace.sink_seconds,
            "extraction_status": "missing" if not code.strip() else "invalid" if outcome == "invalid" else "extracted",
            "extraction_method": "python_code_block_or_source",
            "grader_method": "evalplus" if "evalplus" in task else "native_python",
            "answer_reuse_count": sum(event["event"] == "reuse_validated_answer" for event in trace.events),
            "node_failures": sum(event["event"] == "node_execution" and event.get("status") != "ok" for event in trace.events),
            **call_metrics(calls),
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
            **trace.fields, "schema_version": 1, "committed_at": now(), "rng_state": random.getstate(),
            "summary": summary, "task_description": task["description"], "raw_response": raw_response,
            "events": trace.events, "spans": trace.spans,
            "grade": {
                "correct": outcome == "correct", "outcome": outcome, "method": summary["grader_method"],
                "execution_status": status, "failure_stage": rows[-1]["stage_name"] if rows and outcome != "correct" else None,
                **extra,
            },
            "reference_answer": task.get("reference_code"), "final_evaluation": final,
            "stages": rows, "calls": calls,
            "execution": {
                **execution_info(rows, calls), "selected_worker_model": self.settings.get("model"),
                "selected_router_model": self.settings.get("las_model"),
                "called_models": sorted({call["request"]["model"] for call in calls}),
            },
        })
        record["item"] = identity(task)
        record["summary"]["run_id"] = self.run_id
        with self.guard:
            journals.append(self.journal_paths["tasks"], record)
            self.records[key] = record
            self.last_commit = record["committed_at"]
        trace.event("task_saved", outcome=outcome)
        write_json(self.record_path(key), record)
        self.export_task(key, record)
        self.refresh_progress()
        if len(self.records) % self.export_every == 0:
            self.refresh_snapshot()
        self.log(
            f"{key[0]}/{key[1]} {outcome} | calls={len(calls)} tokens={summary['total_tokens']} "
            f"wall={elapsed:.3f}s | {len(self.records)}/{len(self.items)} complete"
        )

    @cached_property
    def provenance(self):
        packages = {"llm-as-scheduler-mbpp", "datasets", "torch", "transformers", "evalplus", "httpx", "openpyxl"}
        versions = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()
                    if dist.metadata.get("Name", "").lower() in packages}
        sources = Path(__file__).parent
        code = hashlib.sha256()
        for path in sorted(sources.rglob("*.py")):
            code.update(str(path.relative_to(sources)).encode())
            code.update(path.read_bytes())
        return {
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
            "code_sha256": code.hexdigest(),
            "prompt_source_sha256": hashlib.sha256((sources / "prompts.py").read_bytes()).hexdigest(),
        }

    def metadata(self):
        return {
            **self.provenance,
            "run_id": self.run_id, "started_at": self.opened_at, "settings": self.settings,
            "dataset": self.dataset, "planned_tasks": len(self.items) if self.configured else None,
            "grader": "evalplus" if self.dataset.endswith("-plus") else "native_python",
            "device": self.device,
            "dataset_release_hashes": self.dataset_release_hashes,
            "selection_sha256": hashlib.sha256(journals.json_line(self.items)).hexdigest(),
            "router_snapshot": self.router_snapshot, "resume_history": self.resume_history,
            "output": {"result_file": str(self.file("json")), "export_every": self.export_every,
                       "export_excel": self.excel, "heartbeat_seconds": self.period},
        }

    def refresh_progress(self):
        self.summary = summarize(list(self.records.values()), len(self.items) if self.configured else None)
        metadata = self.metadata()
        write_json(self.file("summary.json"), {
            "schema_version": 1, "status": self.status, "metadata": metadata, "summary": self.summary,
        })
        last = next(reversed(self.records.values()), None)
        write_json(self.file("checkpoint.json"), {
            "schema_version": 1, "run_id": self.run_id, "completed_tasks": len(self.records),
            "last_committed_task_id": last["item"]["task_id"] if last else None,
            "last_committed_split": last["item"]["split"] if last else None,
            "last_commit_at": self.last_commit,
            "initial_rng_state": self.initial_rng,
            "latest_committed_rng_state": last.get("rng_state", self.initial_rng) if last else self.initial_rng,
            "resume_signature": self.signature(), "router_snapshot": self.router_snapshot,
        })
        if self.records:
            frame = pd.DataFrame([row["summary"] for row in self.ordered_records()])
            atomic_write(self.path / "summary.csv", lambda temp: frame.to_csv(temp, index=False))

    def ordered_records(self):
        return [self.records[key] for key in self.indices if key in self.records]

    def refresh_snapshot(self):
        records = self.ordered_records()
        report = {
            "schema_version": 1, "status": self.status, "metadata": self.metadata(),
            "summary": self.summary, "tasks": records,
        }
        write_json(self.file("json"), report)
        write_json(self.file("trace.json"), chrome_trace(
            [span for row in records for span in row.get("spans", [])], run_id=self.run_id,
        ))
        if self.excel:
            atomic_write(self.file("xlsx"), lambda path: workbook(path, report))
            if records:
                frame = pd.DataFrame([row["summary"] for row in records])
                excel = frame.apply(lambda column: column.map(excel_cell))
                atomic_write(self.path / "summary.xlsx", lambda temp: excel.to_excel(temp, index=False))
