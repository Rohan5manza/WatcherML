"""Core test suite. Run with: pytest tests/ -v

Covers: run lifecycle, failure capsule diagnosis, structured diff, reproduction
capsule export, the Ollama advisor client (against a fake local server -- no
real Ollama install or network access required), and autopilot's deterministic
fallback + hard iteration cap.
"""
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import watcherml as watcher
from watcherml import advisor
from watcherml.diff import compare_runs
from watcherml.export import export_capsule
from watcherml.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(root=str(tmp_path / ".watcherml"))


def test_successful_run_is_recorded(storage):
    with watcher.init(project="t", config={"lr": 0.1}, storage=storage) as run:
        run.log_metric("acc", 0.9)
    row = storage.get_run(run.run_id)
    assert row["exit_status"] == "success"
    assert storage.final_metrics(run.run_id)["acc"] == 0.9


def test_failed_run_produces_diagnosed_capsule(storage):
    run_id = None
    try:
        with watcher.init(project="t", config={"batch_size": 32}, storage=storage) as run:
            run_id = run.run_id
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    except RuntimeError:
        pass

    row = storage.get_run(run_id)
    assert row["exit_status"] == "failed"
    failure = storage.get_failure(run_id)
    diagnosis = json.loads(failure["diagnosis_json"])
    assert diagnosis["rule"] == "cuda_out_of_memory"


def test_unrecognized_exception_is_unclassified_not_miscategorized(storage):
    run_id = None
    try:
        with watcher.init(project="t", config={}, storage=storage) as run:
            run_id = run.run_id
            raise ValueError("something completely unrelated happened")
    except ValueError:
        pass
    failure = storage.get_failure(run_id)
    diagnosis = json.loads(failure["diagnosis_json"])
    assert diagnosis["rule"] == "unclassified"


def test_compare_runs_reports_config_and_metric_changes(storage):
    with watcher.init(project="t", config={"lr": 0.1}, storage=storage) as run_a:
        run_a.log_metric("acc", 0.5)
    with watcher.init(project="t", config={"lr": 0.01}, storage=storage) as run_b:
        run_b.log_metric("acc", 0.8)

    diff = compare_runs(storage, run_a.run_id, run_b.run_id)
    changed_keys = {c["key"] for c in diff["config_diff"]}
    assert "lr" in changed_keys
    acc_change = next(m for m in diff["metric_diff"] if m["metric"] == "acc")
    assert acc_change["delta"] == pytest.approx(0.3)


def test_export_capsule_contains_manifest_and_config(storage, tmp_path):
    with watcher.init(project="t", config={"lr": 0.01, "seed": 1}, storage=storage) as run:
        run.log_metric("acc", 0.8)
    out_path = export_capsule(storage, run.run_id, out_path=str(tmp_path / "capsule.zip"))
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["config"]["seed"] == 1


def test_redact_hides_common_secret_patterns():
    from watcherml import redact
    text = 'api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"'
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in redact.redact(text)


# --- advisor tests, against a fake local Ollama-compatible server ------------

class _FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models": []}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(length))
        reply = json.dumps({"config": {"batch_size": 16}, "rationale": "test"})
        payload = json.dumps({"message": {"role": "assistant", "content": reply}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_ollama():
    server = HTTPServer(("localhost", 11434), _FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "http://localhost:11434"
    server.shutdown()
    server.server_close()


def test_advisor_degrades_silently_without_ollama():
    assert advisor.is_available(host="http://localhost:19999") is False
    result = advisor.explain_failure(
        {"exception_type": "X", "message": "y",
         "diagnosis": {"rule": "r", "summary": "s"}, "evidence": {}},
        host="http://localhost:19999",
    )
    assert result is None


def test_advisor_suggest_next_config_against_fake_server(fake_ollama):
    assert advisor.is_available(host=fake_ollama) is True
    suggestion = advisor.suggest_next_config(
        run_history=[{"config": {"batch_size": 32}, "status": "failed"}],
        goal_metric="val_accuracy", host=fake_ollama,
    )
    assert suggestion["config"]["batch_size"] == 16


# --- autopilot -----------------------------------------------------------

def test_autopilot_deterministic_fallback_halves_batch_size_until_success(storage):
    from watcherml.autopilot import autopilot as run_autopilot

    def flaky_train(config):
        if config["batch_size"] >= 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"val_accuracy": 0.8}

    result = run_autopilot(
        project="t", train_fn=flaky_train,
        base_config={"batch_size": 32}, goal_metric="val_accuracy",
        max_iterations=6, storage=storage,
    )
    assert result.best_metric_value == 0.8
    final_config = json.loads(storage.get_run(result.best_run_id)["config_json"])
    assert final_config["batch_size"] < 32


def test_autopilot_respects_hard_iteration_cap(storage, monkeypatch):
    import importlib
    # watcherml/__init__.py re-exports the `autopilot` *function* under the package's
    # `autopilot` attribute, shadowing the submodule of the same name -- so
    # `import watcherml.autopilot` would resolve to the function, not the module.
    # importlib.import_module bypasses that and gets the real submodule.
    autopilot_module = importlib.import_module("watcherml.autopilot")

    call_count = {"n": 0}

    def fake_suggest(*args, **kwargs):
        call_count["n"] += 1
        return {"config": {"lr": 1.0 / (call_count["n"] + 1)}, "rationale": "keep going"}

    monkeypatch.setattr(autopilot_module.advisor, "is_available", lambda host=None: True)
    monkeypatch.setattr(autopilot_module.advisor, "suggest_next_config", fake_suggest)

    def always_succeeds(config):
        return {"val_accuracy": 0.5}

    result = autopilot_module.autopilot(
        project="t", train_fn=always_succeeds,
        base_config={"lr": 0.1}, goal_metric="val_accuracy",
        max_iterations=9999, storage=storage,
    )
    assert len(result.run_ids) == autopilot_module.HARD_ITERATION_CAP