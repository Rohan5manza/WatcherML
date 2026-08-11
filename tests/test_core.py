"""Core WatcherML tests.

Covers run lifecycle, deterministic failure diagnosis, run comparison,
capsule export, and secret redaction.
"""
import json
import zipfile


import pytest

import watcherml as watcher
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
    with watcher.init(
        project="t",
        config={"lr": 0.01, "seed": 1},
        storage=storage,
    ) as run:
        run.log_metric("acc", 0.8)

    out_path = export_capsule(
        storage,
        run.run_id,
        out_path=str(tmp_path / "capsule.zip"),
    )

    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()

        assert "manifest.json" in names
        assert "config.json" in names

        manifest = json.loads(zf.read("manifest.json"))
        config = json.loads(zf.read("config.json"))

        assert manifest["schema"]["name"] == "watcherml.run-export"
        assert manifest["schema"]["version"] == "1.0"
        assert config["seed"] == 1

        manifest_paths = {
            item["path"]
            for item in manifest["contents"]
        }
        assert "config.json" in manifest_paths

        
def test_redact_hides_common_secret_patterns():
    from watcherml import redact
    text = 'api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"'
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in redact.redact(text)



