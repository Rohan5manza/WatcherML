"""Acceptance tests for the WatcherML training-entrypoint contract."""
from __future__ import annotations

import json
import sys

import pytest

from watcherml.entrypoint import (
    ENTRYPOINT_SCHEMA_VERSION,
    EntrypointError,
    EntrypointResultError,
    EntrypointSignatureError,
    TrainingEntrypoint,
    invoke_entrypoint,
    validate_config,
    validate_entrypoint,
)


def _write_module(tmp_path, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    sys.modules.pop(name, None)


def test_entrypoint_schema_round_trip():
    spec = TrainingEntrypoint("training.jobs:train", working_directory="jobs")
    payload = spec.to_dict()
    assert payload["schema"]["version"] == ENTRYPOINT_SCHEMA_VERSION
    assert TrainingEntrypoint.from_dict(json.loads(json.dumps(payload))) == spec


@pytest.mark.parametrize(
    "target",
    ["train", "train.py:main", "train:", ":main", "train:<lambda>", "train:bad-name"],
)
def test_invalid_targets_are_rejected(target):
    with pytest.raises(EntrypointError):
        TrainingEntrypoint(target)


@pytest.mark.parametrize("directory", ["/tmp/project", "../project", "C:\\project"])
def test_working_directory_must_be_portable_and_relative(directory):
    with pytest.raises(EntrypointError):
        TrainingEntrypoint("train:main", working_directory=directory)


def test_validate_and_invoke_importable_callable(tmp_path):
    _write_module(
        tmp_path,
        "train_valid",
        """
def main(config, max_steps=None):
    steps = max_steps if max_steps is not None else config["full_steps"]
    return {"loss": 0.25, "steps_completed": steps}
""",
    )
    spec = TrainingEntrypoint("train_valid:main")
    validation = validate_entrypoint(
        spec,
        project_root=str(tmp_path),
        require_max_steps=True,
    )
    assert validation.supports_max_steps is True
    metrics = invoke_entrypoint(
        spec,
        {"full_steps": 100},
        project_root=str(tmp_path),
        max_steps=10,
    )
    assert metrics == {"loss": 0.25, "steps_completed": 10.0}


def test_probe_validation_rejects_callable_without_max_steps(tmp_path):
    _write_module(
        tmp_path,
        "train_unbounded",
        "def main(config):\n    return {'loss': 1.0}\n",
    )
    spec = TrainingEntrypoint("train_unbounded:main")
    with pytest.raises(EntrypointSignatureError, match="will not silently run full training"):
        validate_entrypoint(
            spec,
            project_root=str(tmp_path),
            require_max_steps=True,
        )


def test_callable_must_have_named_config_parameter(tmp_path):
    _write_module(
        tmp_path,
        "train_bad_signature",
        "def main(settings, max_steps=None):\n    return {}\n",
    )
    with pytest.raises(EntrypointSignatureError, match="named 'config'"):
        validate_entrypoint(
            TrainingEntrypoint("train_bad_signature:main"),
            project_root=str(tmp_path),
        )


def test_config_must_be_finite_json():
    assert validate_config({"batch_size": 8, "layers": [1, 2]}) == {
        "batch_size": 8,
        "layers": [1, 2],
    }
    with pytest.raises(EntrypointError, match="non-string key"):
        validate_config({1: "bad"})
    with pytest.raises(EntrypointError, match="non-finite"):
        validate_config({"learning_rate": float("nan")})
    with pytest.raises(EntrypointError, match="JSON values only"):
        validate_config({"layers": (1, 2)})


def test_returned_metrics_must_be_finite_numbers(tmp_path):
    _write_module(
        tmp_path,
        "train_bad_metrics",
        "def main(config, max_steps=None):\n    return {'loss': 'not-a-number'}\n",
    )
    with pytest.raises(EntrypointResultError, match="real number"):
        invoke_entrypoint(
            TrainingEntrypoint("train_bad_metrics:main"),
            {},
            project_root=str(tmp_path),
            max_steps=1,
        )
