"""Tests for the local web UI's JSON API, using FastAPI's TestClient
(no real socket/server needed -- safe for CI)."""
import pytest
from fastapi.testclient import TestClient

import watcherml as watcher
from watcherml.storage import Storage
from watcherml.webapp import create_app


@pytest.fixture
def client_and_ids(tmp_path):
    storage = Storage(root=str(tmp_path / ".watcherml"))

    with watcher.init(project="demo", config={"batch_size": 32}, storage=storage) as run_a:
        pass
    matching_success_id = run_a.run_id  # same config as the failure below -> higher similarity
    try:
        with watcher.init(project="demo", config={"batch_size": 32}, storage=storage) as run_b:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    except RuntimeError:
        pass
    failed_id = run_b.run_id
    with watcher.init(project="demo", config={"batch_size": 16}, storage=storage) as run_c:
        run_c.log_metric("val_accuracy", 0.8)
    success_id = run_c.run_id

    app = create_app(storage)
    return TestClient(app), failed_id, success_id, matching_success_id


def test_index_and_static_assets_are_served(client_and_ids):
    client, _, _, _ = client_and_ids
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_list_projects(client_and_ids):
    client, failed_id, success_id, _ = client_and_ids
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    projects = resp.json()
    demo = next(p for p in projects if p["name"] == "demo")
    assert demo["run_count"] == 3
    assert demo["failure_count"] == 1


def test_get_run_detail(client_and_ids):
    client, _, success_id, _ = client_and_ids
    resp = client.get(f"/api/runs/{success_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_metrics"]["val_accuracy"] == 0.8
    assert body["has_failure"] is False


def test_get_failure_capsule(client_and_ids):
    client, failed_id, success_id, matching_success_id = client_and_ids
    resp = client.get(f"/api/runs/{failed_id}/failure")
    assert resp.status_code == 200
    body = resp.json()
    assert body["diagnosis"]["rule"] == "cuda_out_of_memory"
    # similarity-based selection should prefer the run with a MATCHING config
    # over the more recent but less similar one -- recency is not relevance.
    assert body["comparison_to_last_success"]["run_id"] == matching_success_id
    assert "similarity_score" in body["comparison_to_last_success"]
    assert body["evidence_index"]
    assert body["diagnosis"]["evidence_ids"]


def test_get_failure_capsule_404_for_successful_run(client_and_ids):
    client, _, success_id, _ = client_and_ids
    resp = client.get(f"/api/runs/{success_id}/failure")
    assert resp.status_code == 404


def test_compare_endpoint(client_and_ids):
    client, failed_id, success_id, _ = client_and_ids
    resp = client.get(f"/api/compare?a={failed_id}&b={success_id}")
    assert resp.status_code == 200
    diff = resp.json()
    changed_keys = {c["key"] for c in diff["config_diff"]}
    assert "batch_size" in changed_keys




def test_run_not_found_is_404(client_and_ids):
    client, _, _, _ = client_and_ids
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_overview_endpoint(client_and_ids):
    client, failed_id, success_id, matching_success_id = client_and_ids
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_count"] == 1
    assert body["run_count"] == 3
    assert any(r["run_id"] == failed_id for r in body["runs_needing_attention"])


def test_global_runs_with_filters(client_and_ids):
    client, failed_id, success_id, matching_success_id = client_and_ids
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 3

    resp = client.get("/api/runs?status=failed")
    body = resp.json()
    assert len(body) == 1
    assert body[0]["run_id"] == failed_id


def test_rename_and_resolve_run(client_and_ids):
    client, failed_id, success_id, matching_success_id = client_and_ids
    resp = client.patch(f"/api/runs/{failed_id}", json={"display_name": "My Named Run"})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "My Named Run"

    resp = client.patch(f"/api/runs/{failed_id}", json={"resolved": True, "resolved_note": "fixed it"})
    assert resp.status_code == 200
    assert resp.json()["resolved"] is True

    resp = client.get(f"/api/runs/{failed_id}")
    assert resp.json()["resolved_note"] == "fixed it"


def test_default_display_name_falls_back_to_config_heuristic(tmp_path):
    storage = Storage(root=str(tmp_path / ".watcherml"))
    with watcher.init(project="demo", config={"model": "resnet50", "batch_size": 32},
                       storage=storage) as run:
        pass
    app = create_app(storage)
    client = TestClient(app)
    body = client.get(f"/api/runs/{run.run_id}").json()
    assert body["display_name"] == "resnet50 \u2014 batch 32"


def test_export_endpoint_returns_a_zip(client_and_ids):
    client, _, success_id, _ = client_and_ids
    resp = client.get(f"/api/runs/{success_id}/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"


def test_campaigns_and_memory_endpoints_empty_when_none_run(client_and_ids):
    client, _, _, _ = client_and_ids
    assert client.get("/api/campaigns").json() == []
    assert client.get("/api/memory").json() == []


def test_campaigns_and_memory_endpoints_after_a_real_recovery(tmp_path):
    from watcherml import recovery as recovery_module

    storage = Storage(root=str(tmp_path / ".watcherml"))
    try:
        with watcher.init(project="demo2", config={"batch_size": 32}, storage=storage) as run:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    except RuntimeError:
        pass
    failed_id = run.run_id

    def train_fn(config, max_steps=None):
        if config["batch_size"] >= 32:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"val_accuracy": 0.8}

    recovery_module.recover_from_oom(
        project="demo2", failed_run_id=failed_id, train_fn=train_fn, storage=storage)

    app = create_app(storage)
    client = TestClient(app)

    campaigns = client.get("/api/campaigns").json()
    assert len(campaigns) == 1
    campaign_id = campaigns[0]["campaign_id"]
    assert campaigns[0]["best_run_id"] is not None

    detail = client.get(f"/api/campaigns/{campaign_id}").json()
    assert detail["trials"]
    assert any(t["phase"] == "probe" for t in detail["trials"])

    mem = client.get("/api/memory").json()
    assert len(mem) >= 1
    assert mem[0]["failure_class"] == "cuda_out_of_memory"
    assert mem[0]["attempts"] >= 1
    assert 0.0 <= mem[0]["success_rate"] <= 1.0

def test_settings_endpoint(client_and_ids):
    client, _, _, _ = client_and_ids

    response = client.get("/api/settings")
    assert response.status_code == 200

    body = response.json()

    assert "data_directory" in body
    assert "database" in body
    assert "gpu" in body

    assert "ollama_available" not in body
    assert "ollama_host" not in body
    assert "ollama_default_model" not in body