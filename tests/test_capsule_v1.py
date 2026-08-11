"""Step 2 acceptance tests: deterministic failure-capsule schema v1."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import zipfile
from types import SimpleNamespace

from watcherml.capsule import build_evidence_index, build_failure_capsule
from watcherml.capsule_schema import CAPSULE_SCHEMA_VERSION, validate_capsule
from watcherml.export import export_capsule
from watcherml.failures import diagnose
from watcherml.storage import Storage


def test_cuda_oom_rule_is_deterministic_and_does_not_match_cpu_memory_error():
    result = diagnose(
        "RuntimeError", "CUDA out of memory. Tried to allocate 2.00 GiB", "trace")
    assert result["rule"] == "cuda_out_of_memory"
    assert result["match_kind"] == "deterministic"
    assert result["recoverable_by_bounded_trial"] is True

    cpu_result = diagnose("MemoryError", "cannot allocate memory", "trace")
    assert cpu_result["rule"] == "unclassified"


def test_evidence_ids_do_not_shift_when_categories_are_missing():
    index = build_evidence_index({"gpu": {"available": True}, "recent_metrics": [{"x": 1}]})
    assert [(item["id"], item["category"]) for item in index] == [
        ("EV-5", "gpu"),
        ("EV-10", "recent_metrics"),
    ]


def test_capsule_v1_has_stable_contract_and_persists(tmp_path, monkeypatch):
    storage = Storage(root=str(tmp_path / ".watcherml"))
    storage.upsert_run(
        "run-1", project="demo", config_json={"batch_size": 16},
        started_at=1.0, exit_status="running",
    )
    storage.log_metric("run-1", "loss", 2.5, 7, 2.0)

    monkeypatch.setattr(
        "watcherml.capsule.collectors.collect_torch_cuda_state",
        lambda: {
            "torch_available": True,
            "cuda_available": True,
            "torch_version": "2.test",
            "allocated_bytes": 100,
            "reserved_bytes": 200,
        },
    )
    run = SimpleNamespace(
        run_id="run-1",
        project="demo",
        config={"batch_size": 16, "gradient_accumulation_steps": 2},
        storage=storage,
        sampler=SimpleNamespace(stats=SimpleNamespace(summary=lambda: {
            "ram": {"mean": 20.0, "peak": 25.0}, "sample_count": 2,
        })),
        gpu_info={"available": True, "gpus": [{"name": "test-gpu"}]},
        git_info={"available": True, "commit": "abc"},
        env_info={
            "python_version": "3.11", "platform": "test",
            "packages": {"torch": "2.test"}, "package_count": 1,
            "fingerprint": "env-123",
        },
        dataset_fingerprint="data-123",
        started_at=1.0,
        _notebook_cells=[],
    )

    try:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        capsule = build_failure_capsule(run, exc_type, exc_value, exc_tb)

    assert capsule["capsule_schema_version"] == CAPSULE_SCHEMA_VERSION
    assert capsule["failure"]["class"] == "cuda_out_of_memory"
    assert capsule["evidence"]["training_state"]["last_logged_step"] == 7
    assert capsule["evidence"]["training_state"]["effective_batch_size_per_process"] == 32
    assert 0 <= capsule["capture_completeness"] <= 10
    assert validate_capsule(capsule) == []

    storage.save_failure(
        "run-1", capsule["exception_type"], capsule["message"],
        capsule["traceback"], capsule["diagnosis"], capsule["evidence"],
        capsule=capsule,
    )
    stored = storage.get_failure_capsule("run-1")
    assert stored == capsule


def test_existing_database_is_migrated_without_deleting_rows(tmp_path):
    root = tmp_path / ".watcherml"
    root.mkdir()
    db_path = root / "watcher.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, project TEXT)")
    conn.execute("CREATE TABLE failures (run_id TEXT PRIMARY KEY, exception_type TEXT, "
                 "message TEXT, traceback TEXT, diagnosis_json TEXT, evidence_json TEXT)")
    conn.execute("INSERT INTO runs (run_id, project) VALUES ('old-run', 'demo')")
    conn.commit()
    conn.close()

    storage = Storage(root=str(root))
    assert storage.get_run("old-run")["project"] == "demo"
    run_columns = {row[1] for row in storage._conn.execute("PRAGMA table_info(runs)")}
    failure_columns = {row[1] for row in storage._conn.execute("PRAGMA table_info(failures)")}
    assert {"capsule_schema_version", "capture_completeness"} <= run_columns
    assert {"capsule_json", "failure_class", "captured_at"} <= failure_columns


def test_export_is_byte_stable_and_every_payload_is_checksummed(tmp_path):
    storage = Storage(root=str(tmp_path / ".watcherml"))
    storage.upsert_run(
        "run-export", project="demo", config_json={"batch_size": 8},
        started_at=1.0, ended_at=2.0, duration_seconds=1.0,
        exit_status="failed", git_json={},
        env_json={"python_version": "3.11", "packages": {"watcherml": "0.1"}},
    )
    capsule = {
        "capsule_schema_version": "1.0",
        "run_id": "run-export",
        "failure_class": "cuda_out_of_memory",
        "exception_type": "RuntimeError",
        "message": "CUDA out of memory",
        "traceback": "trace",
        "diagnosis": {"rule": "cuda_out_of_memory"},
        "evidence": {},
    }
    storage.save_failure(
        "run-export", "RuntimeError", "CUDA out of memory", "trace",
        capsule["diagnosis"], {}, capsule=capsule,
    )

    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    export_capsule(storage, "run-export", str(first))
    export_capsule(storage, "run-export", str(second))
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert "failure-capsule.json" in archive.namelist()
        for item in manifest["contents"]:
            payload = archive.read(item["path"])
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]
            assert len(payload) == item["size_bytes"]
