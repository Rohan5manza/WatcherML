"""Serializable training-entrypoint contract for isolated WatcherML trials.

V1 intentionally supports importable Python callables only. Notebook closures,
lambdas, bound methods, and pickled functions are rejected because a fresh
process must be able to reconstruct the training program from explicit code
and JSON data.
"""
from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterator, Mapping, Optional


ENTRYPOINT_SCHEMA_NAME = "watcherml.training-entrypoint"
ENTRYPOINT_SCHEMA_VERSION = "1.0"
ENTRYPOINT_KIND = "python_callable"
MAX_CONFIG_BYTES = 1_000_000

_TARGET_PATTERN = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)


class EntrypointError(ValueError):
    """Base exception for invalid or unresolvable entrypoints."""


class EntrypointResolutionError(EntrypointError):
    """Raised when an entrypoint cannot be imported or resolved."""


class EntrypointSignatureError(EntrypointError):
    """Raised when the callable does not satisfy the v1 function contract."""


class EntrypointResultError(EntrypointError):
    """Raised when training returns invalid metrics."""


@dataclass(frozen=True)
class TrainingEntrypoint:
    """Portable reference to a training callable.

    ``target`` uses ``module.path:function_name`` syntax. ``working_directory``
    is relative to a separately supplied project root, making the serialized
    contract portable instead of embedding one developer's absolute path.
    """

    target: str
    working_directory: str = "."

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not _TARGET_PATTERN.fullmatch(self.target):
            raise EntrypointError(
                "target must use importable 'module.path:function_name' syntax"
            )
        if self.target.split(":", 1)[0].endswith(".py"):
            raise EntrypointError(
                "target uses a module name, not a filename; use 'train:main', "
                "not 'train.py:main'"
            )
        normalized = _validate_relative_directory(self.working_directory)
        object.__setattr__(self, "working_directory", normalized)

    @property
    def module_name(self) -> str:
        return self.target.split(":", 1)[0]

    @property
    def callable_path(self) -> str:
        return self.target.split(":", 1)[1]

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": ENTRYPOINT_SCHEMA_NAME,
                "version": ENTRYPOINT_SCHEMA_VERSION,
            },
            "kind": ENTRYPOINT_KIND,
            "target": self.target,
            "working_directory": self.working_directory,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingEntrypoint":
        if not isinstance(payload, dict):
            raise EntrypointError("entrypoint payload must be an object")
        schema = payload.get("schema") or {}
        if schema.get("name") != ENTRYPOINT_SCHEMA_NAME:
            raise EntrypointError(
                f"entrypoint schema.name must be {ENTRYPOINT_SCHEMA_NAME!r}"
            )
        if schema.get("version") != ENTRYPOINT_SCHEMA_VERSION:
            raise EntrypointError(
                f"entrypoint schema.version must be {ENTRYPOINT_SCHEMA_VERSION!r}"
            )
        if payload.get("kind") != ENTRYPOINT_KIND:
            raise EntrypointError(f"entrypoint kind must be {ENTRYPOINT_KIND!r}")
        return cls(
            target=payload.get("target"),
            working_directory=payload.get("working_directory", "."),
        )


@dataclass(frozen=True)
class EntrypointValidation:
    target: str
    signature: str
    supports_max_steps: bool
    project_root: str
    working_directory: str

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "signature": self.signature,
            "supports_max_steps": self.supports_max_steps,
            "project_root": self.project_root,
            "working_directory": self.working_directory,
        }


def validate_entrypoint(
    spec: TrainingEntrypoint,
    *,
    project_root: str = ".",
    require_max_steps: bool = False,
) -> EntrypointValidation:
    """Import and validate the callable contract.

    Importing a Python module executes its top-level code. Call this only for a
    user-selected local project. The future worker performs the same validation
    again inside the fresh trial process.
    """
    callable_obj, working_directory = _resolve_callable(spec, project_root)
    signature = inspect.signature(callable_obj)
    supports_max_steps = _validate_signature(
        signature,
        require_max_steps=require_max_steps,
    )
    return EntrypointValidation(
        target=spec.target,
        signature=str(signature),
        supports_max_steps=supports_max_steps,
        project_root=str(Path(project_root).resolve()),
        working_directory=str(working_directory),
    )


def invoke_entrypoint(
    spec: TrainingEntrypoint,
    config: dict,
    *,
    project_root: str = ".",
    max_steps: Optional[int] = None,
) -> dict[str, float]:
    """Invoke a validated entrypoint and normalize its returned metrics.

    This function exists so the future subprocess worker and tests share one
    contract implementation. Calling it directly does not provide isolation.
    """
    normalized_config = validate_config(config)
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise EntrypointError("max_steps must be a positive integer or None")

    callable_obj, working_directory = _resolve_callable(spec, project_root)
    signature = inspect.signature(callable_obj)
    supports_max_steps = _validate_signature(
        signature,
        require_max_steps=max_steps is not None,
    )
    kwargs = {"config": normalized_config}
    if supports_max_steps:
        kwargs["max_steps"] = max_steps
    # Resolve relative datasets, checkpoints, and output paths exactly as the
    # user's declared project entrypoint expects. The worker is a short-lived
    # process, but keeping this restoration-safe also makes direct tests sane.
    with _temporary_current_directory(working_directory):
        result = callable_obj(**kwargs)
    return validate_metrics(result)


def validate_config(config: dict) -> dict:
    """Require a finite, JSON-round-trippable configuration object."""
    if not isinstance(config, dict):
        raise EntrypointError("config must be a dictionary")
    _validate_json_value(config, path="config")
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_CONFIG_BYTES:
        raise EntrypointError(
            f"serialized config exceeds the {MAX_CONFIG_BYTES}-byte v1 limit"
        )
    return json.loads(encoded.decode("utf-8"))


def validate_metrics(result) -> dict[str, float]:
    """Require a string-to-finite-number metric mapping."""
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise EntrypointResultError(
            "training entrypoint must return a metric mapping or None"
        )
    metrics: dict[str, float] = {}
    for name, value in result.items():
        if not isinstance(name, str) or not name:
            raise EntrypointResultError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EntrypointResultError(
                f"metric {name!r} must be a real number, got {type(value).__name__}"
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise EntrypointResultError(f"metric {name!r} must be finite")
        metrics[name] = normalized
    return metrics


def _validate_relative_directory(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EntrypointError("working_directory must be a non-empty relative path")
    raw = value.strip().replace("\\", "/")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(value.strip())
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
        raise EntrypointError(
            "working_directory must stay inside the separately supplied project root"
        )
    normalized = posix.as_posix()
    return normalized or "."


def _project_working_directory(spec: TrainingEntrypoint, project_root: str) -> Path:
    root = Path(project_root).resolve()
    if not root.exists() or not root.is_dir():
        raise EntrypointResolutionError(f"project root does not exist: {root}")
    working_directory = (root / spec.working_directory).resolve()
    try:
        working_directory.relative_to(root)
    except ValueError as exc:
        raise EntrypointResolutionError(
            "entrypoint working directory escapes the project root"
        ) from exc
    if not working_directory.exists() or not working_directory.is_dir():
        raise EntrypointResolutionError(
            f"entrypoint working directory does not exist: {working_directory}"
        )
    return working_directory


def _resolve_callable(
    spec: TrainingEntrypoint,
    project_root: str,
) -> tuple[Callable, Path]:
    working_directory = _project_working_directory(spec, project_root)
    try:
        with _temporary_import_path(working_directory):
            with _temporary_current_directory(working_directory):
                importlib.invalidate_caches()
                module = importlib.import_module(spec.module_name)
    except Exception as exc:
        raise EntrypointResolutionError(
            f"could not import module {spec.module_name!r} from {working_directory}: {exc}"
        ) from exc

    value = module
    try:
        for attribute in spec.callable_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise EntrypointResolutionError(
            f"callable {spec.callable_path!r} was not found in {spec.module_name!r}"
        ) from exc
    if not callable(value):
        raise EntrypointResolutionError(f"entrypoint {spec.target!r} is not callable")
    if "<locals>" in getattr(value, "__qualname__", ""):
        raise EntrypointResolutionError("local functions and closures are not supported")
    return value, working_directory


def _validate_signature(
    signature: inspect.Signature,
    *,
    require_max_steps: bool,
) -> bool:
    parameters = signature.parameters
    config_parameter = parameters.get("config")
    if config_parameter is None:
        raise EntrypointSignatureError(
            "training callable must define a parameter named 'config'"
        )
    if config_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise EntrypointSignatureError("'config' must accept keyword invocation")

    max_steps_parameter = parameters.get("max_steps")
    supports_max_steps = max_steps_parameter is not None
    if max_steps_parameter is not None and (
        max_steps_parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ):
        raise EntrypointSignatureError("'max_steps' must accept keyword invocation")
    if require_max_steps and not supports_max_steps:
        raise EntrypointSignatureError(
            "bounded probe trials require a 'max_steps' parameter; WatcherML will "
            "not silently run full training as a probe"
        )
    return supports_max_steps


def _validate_json_value(value, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EntrypointError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EntrypointError(f"{path} contains a non-string key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise EntrypointError(
        f"{path} contains unsupported type {type(value).__name__}; use JSON values only"
    )


@contextmanager
def _temporary_import_path(directory: Path) -> Iterator[None]:
    value = str(directory)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


@contextmanager
def _temporary_current_directory(directory: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)
