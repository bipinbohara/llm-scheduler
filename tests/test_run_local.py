import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from las_mbpp.checkpoints import checkpoint_identity
from las_mbpp.cli import run_from_config
from las_mbpp.local_models import LOCAL_MODELS, LocalChatClient
from las_mbpp.workflows import mbpp_baseline, mbpp_las_distilbert


MODEL = "openai/gpt-oss-20b"
ROUTER = "openai/gpt-oss-120b"


@pytest.fixture
def local_endpoint(monkeypatch):
    pytest.importorskip("httpx")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), request))
            if request["model"] == ROUTER:
                content = '{"action": "early_exit"}'
            else:
                content = "import math\n\ndef helper(x):\n    return x\n\ndef increment(x):\n    return math.floor(helper(x)) + 1"
                if "consensus" in request["messages"][0]["content"].lower():
                    content = '{"selected_agent": "A"}'
            body = json.dumps({
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for model in LOCAL_MODELS.values():
        monkeypatch.setenv(f"{model.env_prefix}_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        monkeypatch.delenv(f"{model.env_prefix}_API_KEY", raising=False)
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture_tasks(*, split, limit, task_ids):
    return [
        {"split": split, "task_id": task_id, "function_name": "increment",
         "description": "Add one.", "prompt_stub": "def increment(x): ...",
         "tests": ["assert increment(10) == 11"], "reference_code": "def increment(x): return x + 1"}
        for task_id in (1, 2)
    ][:limit]


def launch(monkeypatch, config, *options):
    monkeypatch.setattr(sys, "argv", ["run_from_config", str(config), *map(str, options)])
    run_from_config.main()


def test_every_local_profile_uses_its_own_endpoint_and_optional_key(monkeypatch, local_endpoint):
    client = LocalChatClient(max_tokens=42, temperature=0.2)
    for name, spec in LOCAL_MODELS.items():
        if name == ROUTER:
            monkeypatch.setenv(f"{spec.env_prefix}_API_KEY", "sample-local-test-key")
        result = client.call(system_prompt="system", user_prompt="user", model=name)
        assert result["ok"]
        assert result["usage"]["total_tokens"] == 5
        endpoint, authorization, request = local_endpoint[-1]
        assert endpoint == "/v1/chat/completions"
        assert authorization == ("Bearer sample-local-test-key" if name == ROUTER else None)
        assert request["max_tokens"] == 42
        assert request["model"] == name


def test_baseline_config_resumes_after_interruption(monkeypatch, tmp_path, local_endpoint):
    monkeypatch.setattr(mbpp_baseline, "load_mbpp_sanitized_tasks", fixture_tasks)
    real_pipeline = mbpp_baseline.run_pipeline_for_task

    def interrupt(**kwargs):
        if kwargs["task"]["task_id"] == 2:
            raise KeyboardInterrupt
        return real_pipeline(**kwargs)

    monkeypatch.setattr(mbpp_baseline, "run_pipeline_for_task", interrupt)
    try:
        from openpyxl import load_workbook
    except ImportError:
        load_workbook = None
    args = ("--output_dir", tmp_path, "--limit_per_split", 2)
    if load_workbook is not None:
        args += ("--excel",)
    with pytest.raises(KeyboardInterrupt):
        launch(monkeypatch, "configs/mbpp_local_baseline.yaml", *args)

    run, = (tmp_path / "mbpp").iterdir()
    first = run / "test/task_1.json"
    existing = first.read_bytes()
    record = json.loads(existing)
    assert record["summary"]["final_is_correct"]
    assert record["summary"]["logical_calls"] == len(local_endpoint) == 10
    assert record["summary"]["total_tokens"] == 50
    assert len(json.loads((run / "run_config.json").read_text())["selected_items"]) == 2
    assert "Interrupted" in (run / "run.log").read_text()
    (run / "test/task_1.csv").unlink()
    monkeypatch.setattr(mbpp_baseline, "run_pipeline_for_task", real_pipeline)
    launch(monkeypatch, "configs/mbpp_local_baseline.yaml", *args, "--resume", run.name)
    assert first.read_bytes() == existing
    assert (run / "test/task_1.csv").exists()
    assert (run / "test/task_2.json").exists()
    assert len(local_endpoint) == 20
    assert len((run / "summary.csv").read_text().splitlines()) > 2
    if load_workbook is not None:
        workbook = load_workbook(run / "summary.xlsx")
        assert workbook.active.max_row == 3
        workbook.close()
    with pytest.raises(ValueError, match="settings differ"):
        launch(monkeypatch, "configs/mbpp_local_baseline.yaml", *args, "--resume", run.name, "--max_tokens", 77)


def test_las_config_uses_separate_router_and_strict_checkpoint(monkeypatch, tmp_path, local_endpoint):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vocab = model_dir / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ndef\n")
    DistilBertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_dir)
    model = DistilBertForSequenceClassification(DistilBertConfig(
        vocab_size=6, n_layers=1, n_heads=2, dim=8, hidden_dim=16, num_labels=2,
    ))
    model.save_pretrained(model_dir)
    checkpoint = tmp_path / "trained.pt"
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setattr(mbpp_las_distilbert, "load_mbpp_sanitized_tasks", fixture_tasks)
    out = tmp_path / "results"
    args = ("--output_dir", out, "--gate_model_name", model_dir, "--gate_threshold", 0,
            "--max_tokens", 100, "--las_max_tokens", 20)
    launch(monkeypatch, "configs/mbpp_local_las.yaml", *args, "--gate_checkpoint_path", checkpoint)

    run, = (out / "mbpp").iterdir()
    config = json.loads((run / "run_config.json").read_text())
    assert config["checkpoint"]["sha256"] == checkpoint_identity(checkpoint, 0)["sha256"]
    assert config["checkpoint"]["load_outcome"] == "strict_load_succeeded"
    result = json.loads((run / "test/task_1.json").read_text())
    assert result["summary"]["final_is_correct"]
    assert result["summary"]["final_route_action"] == "early_exit"
    assert {req["model"]: req["max_tokens"] for _, _, req in local_endpoint} == {MODEL: 100, ROUTER: 20}
    assert any(call["attribution"] == "routing" for call in result["calls"])
    control = mbpp_las_distilbert.get_or_create_gate_scorer(
        model_name=str(model_dir), checkpoint_path=None, max_length=32, seed=7,
    )
    assert control.checkpoint["condition"] == "untrained_gate"
    assert not torch.equal(control._model.classifier.weight, model.classifier.weight)


def test_gate_pointer_and_native_timeout(tmp_path):
    pointer = tmp_path / "checkpoint.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 123\n")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        checkpoint_identity(pointer, 0)
    task = {"function_name": "increment", "tests": ["assert increment(0) == 1"]}
    start = time.perf_counter()
    result = mbpp_baseline.evaluate_code(task, "while True: pass", timeout_sec=0.05)
    assert result["exec_error"] == "Timeout"
    assert time.perf_counter() - start < 5
