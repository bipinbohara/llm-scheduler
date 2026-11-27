"""Live task telemetry. Context follows a task through its worker threads."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
from functools import wraps
from threading import RLock, get_ident
from uuid import uuid4

CURRENT = ContextVar("experiment_task", default=None)
PARENT = ContextVar("experiment_span", default=None)
CALL = ContextVar("experiment_call", default=None)


def now():
    return datetime.now(timezone.utc).isoformat()


def submit(pool, fn, *args, **kwargs):
    return pool.submit(copy_context().run, fn, *args, **kwargs)


class TaskTrace:
    def __init__(self, run, task, evaluation_index):
        self.run = run
        self.fields = {
            "run_id": run.run_id, "split": task["split"], "task_id": task["task_id"],
            "evaluation_index": evaluation_index, "task_attempt": uuid4().hex, "trace_id": uuid4().hex,
        }
        self.lock = RLock()
        self.events = []
        self.spans = []
        self.open_spans = {}
        self.sink_seconds = 0.0
        self.clock = time.perf_counter_ns()
        self.cpu = time.process_time()
        self.index = 0

    def event(self, event, **fields):
        with self.lock:
            self.index += 1
            row = {
                "schema_version": 1, **self.fields, "event_index": self.index, "event": event,
                "timestamp": now(), "span_id": PARENT.get(), **fields,
            }
            start = time.perf_counter()
            self.events.append(self.run.emit(row))
            self.sink_seconds += time.perf_counter() - start

    def begin(self, name, kind, fields):
        with self.lock:
            tick = time.perf_counter_ns()
            record = {
                **self.fields, "span_id": uuid4().hex, "parent_span_id": PARENT.get(),
                "name": name, "kind": kind, "invocation_index": self.index + 1,
                "started_at": now(), "start_offset_seconds": (tick - self.clock) / 1e9,
                "start_monotonic_ns": tick, "pid": os.getpid(), "thread_id": get_ident(),
                "attributes": fields,
            }
            self.open_spans[record["span_id"]] = record
            self.run.active_span(record)
            self.event("span_started", **record)
            return record

    def finish(self, span, status="ok", error=None):
        with self.lock:
            tick = time.perf_counter_ns()
            span.update(
                finished_at=now(), finish_monotonic_ns=tick, status=status, error=error,
                inclusive_seconds=(tick - span["start_monotonic_ns"]) / 1e9 if status != "incomplete" else None,
                observed_seconds=(tick - span["start_monotonic_ns"]) / 1e9,
            )
            if span["inclusive_seconds"] is not None:
                cursor = span["start_monotonic_ns"]
                covered = 0
                children = sorted(
                    (child["start_monotonic_ns"], child["finish_monotonic_ns"])
                    for child in self.spans if child.get("parent_span_id") == span["span_id"]
                )
                for start, finish in children:
                    if finish > cursor:
                        covered += finish - max(start, cursor)
                        cursor = finish
                span["exclusive_seconds"] = max(0.0, span["inclusive_seconds"] - covered / 1e9)
            else:
                span["exclusive_seconds"] = None
            self.open_spans.pop(span["span_id"])
            self.run.inactive_span(span)
            self.event("span_finished", **span)
            start = time.perf_counter()
            self.spans.append(self.run.emit_span(span))
            self.sink_seconds += time.perf_counter() - start


@contextmanager
def span(name, kind="function", **fields):
    trace = CURRENT.get()
    if trace is None:
        yield {}
        return
    record = trace.begin(name, kind, fields)
    token = PARENT.set(record["span_id"])
    status, error = "ok", None
    try:
        yield fields
        status = fields.pop("_status", "ok")
    except BaseException as exc:
        status, error = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace.finish(record, status, error)
        PARENT.reset(token)


def event(event_name, **fields):
    trace = CURRENT.get()
    if trace is not None:
        trace.event(event_name, **fields)


def observe(name, kind="function", **fixed):
    def decorate(fn):
        @wraps(fn)
        def traced(*args, **kwargs):
            fields = {**fixed, **{key: kwargs[key] for key in ("source_call_id", "stage") if key in kwargs}}
            with span(name, kind, **fields) as attributes:
                node_fields = {"stage": name, **fields}
                if kind == "node":
                    event("node_execution_started", node_id=PARENT.get(), **node_fields)
                if kind == "processing":
                    event("processing_stage_started", name=name)
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    if kind == "node":
                        event("node_execution", node_id=PARENT.get(), status="error",
                              error=type(exc).__name__, **node_fields)
                    if kind == "processing":
                        event("processing_stage", name=name, status="error")
                    raise
                if kind == "node":
                    event("node_execution", node_id=PARENT.get(), status="ok", **node_fields)
                if kind == "processing":
                    event("processing_stage", name=name, status="ok")
                if isinstance(result, dict) and "passed_all" in result:
                    attributes.update(
                        passed=result["passed_all"], status=result.get("exec_status"),
                        passed_count=result.get("pass_count"), total_count=result.get("total_count"),
                    )
                return result
        return traced
    return decorate


def model_node(client, stage, agent, **request):
    with span(stage, "node", agent=agent, model=request["model"]) as attributes:
        event("node_execution_started", node_id=PARENT.get(), stage=stage, agent=agent, model=request["model"])
        token = NODE.set((stage, agent))
        try:
            response = client.call(**request)
        except BaseException as exc:
            event("node_execution", node_id=PARENT.get(), stage=stage, agent=agent,
                  status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error")
            raise
        else:
            attributes["_status"] = "ok" if response.get("ok") else "error"
            event("node_execution", node_id=PARENT.get(), stage=stage, agent=agent,
                  status=attributes["_status"], output=response.get("output_text"), error=response.get("error"),
                  call_id=response.get("_trace_call_id"))
            return response
        finally:
            NODE.reset(token)


NODE = ContextVar("experiment_node", default=(None, None))


def http_request(request):
    context = CALL.get()
    if context is None:
        return
    trace, call, pending = context
    index = len(call["http_attempts"]) + len(pending) + 1
    fields = {"call_id": call["call_id"], "attempt_id": uuid4().hex, "attempt_index": index,
              "method": request.method}
    record = trace.begin("http.response_headers", "http_attempt", fields)
    # A retry can follow an exception before headers were received.
    pending[request] = record


def http_response(response):
    context = CALL.get()
    if context is None:
        return
    trace, call, pending = context
    record = pending.pop(response.request, None)
    if record is None:
        return
    record["attributes"].update(
        status_code=response.status_code,
        request_id=response.headers.get("x-request-id") or response.headers.get("request-id"),
    )
    trace.finish(record, "error" if response.is_error else "ok")
    call["http_attempts"].append(record)


def finish_http_attempts(context):
    trace, call, pending = context
    for record in pending.values():
        trace.finish(record, "incomplete")
        call["http_attempts"].append(record)
    call["http_attempts"].sort(key=lambda row: row["attributes"]["attempt_index"])


HTTP_HOOKS = {"request": [http_request], "response": [http_response]}
