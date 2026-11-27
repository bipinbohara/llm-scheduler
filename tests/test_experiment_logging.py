from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest

from las_mbpp.cli import export_trace
from las_mbpp.journals import recover
from las_mbpp.run_artifacts import RunArtifacts, TracedClient
from las_mbpp.telemetry import model_node
from las_mbpp.workflows.mbpp_baseline import AnthropicChatClient


def test_recover_refuses_interior_corruption(tmp_path):
    path = tmp_path / "experiment.events.jsonl"
    data = b'{"event":1}\n{"missing":\n{"event":3}\n'
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Interior JSONL corruption"):
        recover(path, repair=True)
    assert path.read_bytes() == data
    assert not list(tmp_path.glob("*.bak"))


def test_trace_export_includes_interrupted_spans(monkeypatch, tmp_path):
    source = tmp_path / "experiment.events.jsonl"
    span = dict(event="span_started", started_at="2026-01-01T00:00:00+00:00", name="waiting", kind="llm_call",
                span_id="s", parent_span_id=None, trace_id="t", task_attempt="a", task_id=1,
                pid=1, thread_id=1, attributes={})
    source.write_bytes(json.dumps(span).encode() + b'\n{"truncated":')
    output = tmp_path / "recovered.trace.json"
    monkeypatch.setattr("sys.argv", ["export_trace", str(source), str(output)])
    export_trace.main()
    report = json.loads(output.read_text())
    assert report["traceEvents"][0]["ph"] == "i"
    assert report["traceEvents"][0]["args"]["status"] == "incomplete"
    assert report["metadata"]["ignored_partial_tail_bytes"] == len(b'{"truncated":')


def test_sdk_retry_traces_two_http_attempts_under_one_node(tmp_path):
    pytest.importorskip("anthropic")
    request_count = 0
    secret = "fixture-secret-for-redaction"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            self.rfile.read(int(self.headers["Content-Length"]))
            request_count += 1
            if request_count == 1:
                body = {"type": "error", "error": {"type": "rate_limit_error", "message": "retry this call"}}
                status = 429
            else:
                body = {
                    "id": "msg_fixture", "type": "message", "role": "assistant", "model": "test-model",
                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                }
                status = 200
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("request-id", f"request-{request_count}")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    args = SimpleNamespace(resume=None, excel=False, export_every=1, heartbeat_seconds=0,
                           overwrite=False, dataset="mbpp", output_dir=tmp_path,
                           result_file=tmp_path / "pilot.json", run_name="retry")
    settings = {"dataset": "mbpp", "llm_api": "anthropic"}
    task = {"split": "test", "task_id": 1, "description": "retry", "function_name": "solve"}
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with RunArtifacts(args, [secret]) as run:
                run.configure(settings, [task])
                backend = AnthropicChatClient(
                    api_key=secret, base_url=f"http://127.0.0.1:{server.server_port}",
                    max_tokens=4, temperature=0,
                )
                with run.task(task) as trace:
                    result = model_node(TracedClient(backend, "workflow"), "debate_initial", "debate_agent_A",
                                        system_prompt=secret, user_prompt="retry", model="test-model")
                assert result["ok"]
                event, = [row for row in trace.events if row["event"] == "llm_call"]
                assert event["call"]["retries"] == 1
                assert [item["attributes"]["status_code"] for item in event["call"]["http_attempts"]] == [429, 200]
                assert len([row for row in trace.events if row["event"] == "node_execution"]) == 1
                assert event["call"]["request"]["system_prompt"] == "[REDACTED]"
        finally:
            server.shutdown()
            thread.join()
    assert (tmp_path / "pilot.json").exists()
    assert not (tmp_path / "pilot.heartbeat.jsonl").exists()
    assert secret not in (tmp_path / "pilot.log").read_text()
    assert secret not in (tmp_path / "pilot.events.jsonl").read_text()


def test_heartbeat_advances_without_committing_a_task(tmp_path):
    import time

    from las_mbpp.telemetry import span

    args = SimpleNamespace(resume=None, excel=False, export_every=25, heartbeat_seconds=0.03,
                           overwrite=False, dataset="mbpp", output_dir=tmp_path,
                           result_file=tmp_path / "pilot.json", run_name="heartbeat")
    task = {"split": "test", "task_id": 1, "description": "pending", "function_name": "solve"}
    heartbeat_path = tmp_path / "pilot.heartbeat.jsonl"
    with RunArtifacts(args) as run:
        run.configure({"dataset": "mbpp"}, [task])
        with run.task(task), span("held_operation"):
            deadline = time.monotonic() + 5
            intervals = []
            while time.monotonic() < deadline:
                rows, _ = recover(heartbeat_path)
                intervals = [row for row in rows if row["active_stage"] == "held_operation"]
                if len(intervals) >= 2:
                    break
                time.sleep(0.02)
            assert len(intervals) >= 2
            assert intervals[1]["active_stage_seconds"] > intervals[0]["active_stage_seconds"]
            assert intervals[1]["active_task_id"] == 1
            assert all(row["completed_tasks"] == 0 for row in intervals)
    rows, _ = recover(heartbeat_path)
    assert rows[0]["trigger"] == "startup"
    assert rows[-1]["trigger"] == "exit"
    assert rows[-1]["in_flight_count"] == 0


def test_grader_child_does_not_hold_the_experiment_lock(tmp_path):
    import fcntl
    import multiprocessing

    context = multiprocessing.get_context("fork")
    release = context.Event()
    args = SimpleNamespace(resume=None, excel=False, export_every=1, heartbeat_seconds=0,
                           overwrite=False, dataset="mbpp", output_dir=tmp_path,
                           result_file=tmp_path / "pilot.json", run_name="lock")
    child = context.Process(target=release.wait, args=(8,))
    try:
        with RunArtifacts(args):
            child.start()
        assert child.is_alive()
        for name in (".run.lock", "pilot.lock"):
            with (tmp_path / name).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
        if child.pid:
            child.join(8)
