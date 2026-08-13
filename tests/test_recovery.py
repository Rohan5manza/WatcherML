"""Tests for deterministic, bounded CUDA OOM recovery campaigns."""
import json


import pytest

import watcherml as watcher
from watcherml import recovery
from watcherml.storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(root=str(tmp_path / ".watcherml"))


@pytest.fixture
def failed_oom_run(storage):
    """A run that failed with a real cuda_out_of_memory diagnosis, matching
    what observe() expects to find."""
    try:
        with watcher.init(project="t", config={"batch_size": 32, "lr": 1e-3},
                           storage=storage) as run:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.10 GiB")
    except RuntimeError:
        pass
    return run.run_id


# ============================================================================
# Policy engine
# ============================================================================

def test_validate_patch_accepts_allowed_keys_only():
    cleaned, rejected = recovery.validate_patch({
        "batch_size": 8,
        "precision": "bf16",
        "learning_rate": 1e-5,
    })

    assert cleaned == {
        "batch_size": 8,
    }

    assert set(rejected) == {
        "precision",
        "learning_rate",
    }


def test_validate_patch_rejects_invalid_and_non_v1_values():
    cleaned, rejected = recovery.validate_patch({
        "batch_size": -1,
        "gradient_accumulation_steps": 0,
        "gradient_checkpointing": "yes",
        "precision": "bf16",
        "num_workers": 4,
    })

    assert cleaned == {}
    assert set(rejected) == {
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "precision",
        "num_workers",
    }

def test_validate_patch_handles_non_dict_input():
    cleaned, rejected = recovery.validate_patch("not a dict")
    assert cleaned == {}
    assert rejected


def test_validate_patch_coerces_numeric_strings():
    cleaned, rejected = recovery.validate_patch({"batch_size": "16"})
    assert cleaned == {"batch_size": 16}
    assert rejected == []


# ============================================================================
# Observer
# ============================================================================

def test_observe_builds_factual_report_from_failure_capsule(storage, failed_oom_run):
    observation = recovery.observe(storage, failed_oom_run)
    assert observation["failure_class"] == "cuda_out_of_memory"
    assert observation["config"]["batch_size"] == 32
    assert observation["run_id"] == failed_oom_run


def test_observe_raises_for_run_that_did_not_fail(storage):
    with watcher.init(project="t", config={}, storage=storage) as run:
        pass
    with pytest.raises(ValueError):
        recovery.observe(storage, run.run_id)


def test_observe_raises_for_unknown_run(storage):
    with pytest.raises(ValueError):
        recovery.observe(storage, "does-not-exist")




def test_recover_from_oom_deterministic_fallback_finds_a_working_config(storage, failed_oom_run):
    def train_fn(config, max_steps=None):
        if config["batch_size"] >= 32:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"val_accuracy": 0.8}

    report = recovery.recover_from_oom(
        project="t", failed_run_id=failed_oom_run, train_fn=train_fn, storage=storage,
    )
    assert report["best_run_id"] is None
    assert report["verification_status"] == "pending_confirmation"

    provisional_run_id = report["provisional_best_run_id"]
    assert provisional_run_id is not None

    best_config = json.loads(
        storage.get_run(provisional_run_id)["config_json"]
)
    assert best_config["batch_size"] < 32
    # every trial (probe + full) must be independently inspectable as a normal run
    trials = storage.list_recovery_trials(report["campaign_id"])
    assert len(trials) >= 1
    assert all(storage.get_run(t["run_id"]) is not None for t in trials)


def test_recover_from_oom_reports_when_nothing_survives(storage, failed_oom_run):
    def always_ooms(config, max_steps=None):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")

    report = recovery.recover_from_oom(
        project="t", failed_run_id=failed_oom_run, train_fn=always_ooms, storage=storage,
    )
    assert report["best_run_id"] is None
    assert report["survivors"] == 0


def test_recover_from_oom_respects_max_trials_budget(storage, failed_oom_run):
    def always_succeeds(config, max_steps=None):
        return {"val_accuracy": 0.5}

    contract = recovery.RecoveryContract(max_trials=1, max_candidates=3)
    report = recovery.recover_from_oom(
        project="t", failed_run_id=failed_oom_run, train_fn=always_succeeds,
        contract=contract, storage=storage,
    )
    assert report["trials_run"] <= 1


def test_recover_from_oom_falls_back_when_train_fn_has_no_max_steps_param(storage, failed_oom_run):
    """train_fn(config) with no max_steps kwarg must still work -- probing
    just degrades to a full call rather than crashing."""
    def train_fn_no_probe_support(config):
        if config["batch_size"] >= 32:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"val_accuracy": 0.7}

    report = recovery.recover_from_oom(
        project="t", failed_run_id=failed_oom_run, train_fn=train_fn_no_probe_support,
        storage=storage,
    )
    assert report["best_run_id"] is None
    assert report["provisional_best_run_id"] is not None
    assert report["verification_status"] == "pending_confirmation"




# ============================================================================
# Memory: campaigns and trials are independently inspectable afterward
# ============================================================================

def test_campaign_and_trials_are_persisted_for_later_inspection(storage, failed_oom_run):
    def train_fn(config, max_steps=None):
        if config["batch_size"] >= 16:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return {"val_accuracy": 0.8}

    report = recovery.recover_from_oom(
        project="t", failed_run_id=failed_oom_run, train_fn=train_fn, storage=storage,
    )
    campaign = storage.get_recovery_campaign(report["campaign_id"])
    assert campaign is not None
    assert campaign["source_run_id"] == failed_oom_run
    assert campaign["best_run_id"] == report["best_run_id"]

    trials = storage.list_recovery_trials(report["campaign_id"])
    assert any(t["phase"] == "probe" for t in trials)