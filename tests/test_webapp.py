"""Tests for the local web UI's JSON API, using FastAPI's TestClient
(no real socket/server needed -- safe for CI)."""
import pytest
from fastapi.testclient import TestClient

import watcherml as watcher
from watcherml.storage import Storage
from watcherml.webapp import create_app
from watcherml.recovery import recover_from_oom
from watcherml.recovery_contract import (
    MetricGuard,
    RecoveryBudget,
    VerificationRequirements,
)


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


def test_rename_run_and_reject_manual_resolution(client_and_ids):
    client, failed_id, _, _ = client_and_ids

    response = client.patch(
        f"/api/runs/{failed_id}",
        json={"display_name": "My Named Run"},
    )
    assert response.status_code == 200
    assert response.json()["display_name"] == "My Named Run"

    response = client.patch(
        f"/api/runs/{failed_id}",
        json={
            "resolved": True,
            "resolved_note": "fixed it manually",
        },
    )
    assert response.status_code == 409
    assert "verifier" in response.json()["detail"].lower()

    response = client.get(f"/api/runs/{failed_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["display_name"] == "My Named Run"
    assert body["resolved"] is False
    assert body["resolved_note"] is None

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
    storage = Storage(root=str(tmp_path / ".watcherml"))

    source_config = {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "oom_batch_limit": 16,
        "full_steps": 20,
    }

    try:
        with watcher.init(
            project="demo2",
            config=source_config,
            storage=storage,
        ) as run:
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 2 GiB"
            )
    except RuntimeError:
        pass

    failed_id = run.run_id

    project_root = tmp_path / "training-project"
    project_root.mkdir()

    training_module = project_root / "web_recovery_training.py"
    training_module.write_text(
        """
def train(config, max_steps=None):
    if config["batch_size"] > config.get("oom_batch_limit", 16):
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 2 GiB"
        )

    steps = (
        max_steps
        if max_steps is not None
        else config.get("full_steps", 20)
    )

    return {
        "validation_loss": 0.4,
        "steps_completed": steps,
        "throughput": 100.0,
    }
""".lstrip(),
        encoding="utf-8",
    )

    verification = VerificationRequirements(
        minimum_progress_steps=20,
        metric_guards=(
            MetricGuard(
                name="validation_loss",
                direction="minimize",
                baseline_value=0.5,
                max_regression=0.1,
            ),
        ),
        confirmation_runs=1,
    )

    budget = RecoveryBudget(
        max_trials=3,
        max_probe_trials=1,
        max_full_trials=1,
        probe_steps=3,
        trial_timeout_seconds=20,
        campaign_timeout_seconds=60,
    )

    result = recover_from_oom(
        failed_id,
        "web_recovery_training:train",
        verification,
        budget=budget,
        storage=storage,
        project_root=project_root,
        trials_root=tmp_path / "trials",
        print_summary=False,
    )

    assert result.verified is True

    app = create_app(storage)
    client = TestClient(app)

    campaigns_response = client.get("/api/campaigns")
    assert campaigns_response.status_code == 200
    campaigns = campaigns_response.json()

    assert len(campaigns) == 1
    campaign = campaigns[0]
    campaign_id = campaign["campaign_id"]

    assert campaign["verified"] is True
    assert campaign["verification_status"] == "verified"
    assert campaign["verified_candidate_id"] is not None
    assert campaign["verified_run_ids"]
    assert campaign["trial_count"] >= 3

    detail_response = client.get(
        f"/api/campaigns/{campaign_id}"
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()

    assert detail["trials"]
    assert detail["proposals"]
    assert detail["verifications"]
    assert detail["artifact"]["available"] is True

    phases = {trial["phase"] for trial in detail["trials"]}
    assert "probe" in phases
    assert "full" in phases
    assert "confirmation" in phases

    memory_response = client.get("/api/memory")
    assert memory_response.status_code == 200
    memory = memory_response.json()

    assert len(memory) >= 1
    assert memory[0]["failure_class"] == "cuda_out_of_memory"
    assert memory[0]["attempts"] >= 1
    assert memory[0]["verified_recoveries"] >= 1
    assert 0.0 <= memory[0]["success_rate"] <= 1.0
    assert 0.0 <= memory[0]["verification_rate"] <= 1.0

def test_settings_endpoint(client_and_ids):
    client, _, _, _ = client_and_ids

    response = client.get("/api/settings")
    assert response.status_code == 200

    body = response.json()

    assert "data_directory" in body
    assert "database_path" in body
    assert "gpu" in body

    assert "ollama_available" not in body
    assert "ollama_host" not in body
    assert "ollama_default_model" not in body