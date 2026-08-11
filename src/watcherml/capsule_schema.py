"""Public schema helpers for WatcherML deterministic failure capsules.

The capsule format is deliberately versioned independently from the Python
package.  Recovery code may only consume fields documented by this module;
new evidence can be added in a backwards-compatible release, but existing
field meanings must not change without a schema-version bump.
"""
from __future__ import annotations

from typing import Any


CAPSULE_SCHEMA_NAME = "watcherml.failure-capsule"
CAPSULE_SCHEMA_VERSION = "1.0"
OOM_FAILURE_CLASS = "cuda_out_of_memory"


# Ten independently useful evidence groups make the score easy to explain.
# The score measures capture completeness, not reproducibility or diagnosis
# confidence.
_COMPLETENESS_CHECKS = (
    ("failure_identity", lambda failure, evidence: bool(
        failure.get("exception_type") and failure.get("message"))),
    ("traceback", lambda failure, evidence: bool(failure.get("traceback"))),
    ("config", lambda failure, evidence: bool(evidence.get("config"))),
    ("training_state", lambda failure, evidence: any(
        value is not None for value in (evidence.get("training_state") or {}).values())),
    ("resource_state_at_failure", lambda failure, evidence: bool(
        evidence.get("resource_state_at_failure"))),
    ("gpu", lambda failure, evidence: bool(
        (evidence.get("gpu") or {}).get("available"))),
    ("framework", lambda failure, evidence: bool(
        (evidence.get("framework") or {}).get("python_version"))),
    ("environment_fingerprint", lambda failure, evidence: bool(
        (evidence.get("environment") or {}).get("fingerprint"))),
    ("git", lambda failure, evidence: bool(
        (evidence.get("git") or {}).get("available"))),
    ("dataset_fingerprint", lambda failure, evidence: bool(
        (evidence.get("dataset") or {}).get("fingerprint"))),
)


def calculate_capture_completeness(failure: dict, evidence: dict) -> dict:
    """Return an auditable 0-10 evidence-completeness result."""
    present: list[str] = []
    missing: list[str] = []
    for name, check in _COMPLETENESS_CHECKS:
        try:
            is_present = bool(check(failure, evidence))
        except Exception:
            is_present = False
        (present if is_present else missing).append(name)
    return {
        "score": len(present),
        "maximum": len(_COMPLETENESS_CHECKS),
        "present": present,
        "missing": missing,
    }


def validate_capsule(capsule: dict[str, Any]) -> list[str]:
    """Validate the stable v1 contract and return human-readable errors.

    This intentionally avoids a runtime dependency on a JSON-schema package.
    It is used as a final guard before a capsule is persisted or exported.
    """
    errors: list[str] = []
    schema = capsule.get("schema")
    if not isinstance(schema, dict):
        errors.append("schema must be an object")
    else:
        if schema.get("name") != CAPSULE_SCHEMA_NAME:
            errors.append(f"schema.name must be {CAPSULE_SCHEMA_NAME!r}")
        if schema.get("version") != CAPSULE_SCHEMA_VERSION:
            errors.append(f"schema.version must be {CAPSULE_SCHEMA_VERSION!r}")

    for key in ("run_id", "project", "captured_at", "failure", "evidence", "capture"):
        if key not in capsule:
            errors.append(f"missing required field: {key}")

    failure = capsule.get("failure")
    if not isinstance(failure, dict):
        errors.append("failure must be an object")
    else:
        for key in ("class", "exception_type", "message", "traceback", "classification"):
            if key not in failure:
                errors.append(f"missing required failure field: {key}")

    if not isinstance(capsule.get("evidence"), dict):
        errors.append("evidence must be an object")

    capture = capsule.get("capture")
    if not isinstance(capture, dict):
        errors.append("capture must be an object")
    elif not isinstance(capture.get("score"), int) or not 0 <= capture["score"] <= 10:
        errors.append("capture.score must be an integer from 0 to 10")
    return errors
