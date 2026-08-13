"""Deterministic recovery-capability discovery for WatcherML.

This module answers one narrow question before recovery planning begins:
which OOM-relevant controls can this particular training entrypoint actually
accept?

Capabilities are facts, not proposals.  Discovering ``micro_batch_size`` does
not authorize WatcherML to change it.  The policy/planner layer must still
choose a transition, validate it against the capability, enforce the recovery
contract, and persist the resulting intervention.

V1 supports two safe discovery mechanisms:

* explicit declarations that map a canonical capability to a config path;
* conservative detection of known aliases in a JSON configuration.

Ambiguous aliases are never guessed.  Framework imports, GPU probing, source
rewriting, dependency installation, and arbitrary environment mutation do not
belong here.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .entrypoint import EntrypointError, validate_config


CAPABILITY_SCHEMA_NAME = "watcherml.capability-manifest"
CAPABILITY_SCHEMA_VERSION = "1.0"

CAPABILITY_LOCATIONS = frozenset({"config", "environment"})
CAPABILITY_PERMISSIONS = frozenset(
    {"automatic", "approval_required", "disabled"}
)
CAPABILITY_RISKS = frozenset({"low", "medium", "high"})
CAPABILITY_VALUE_TYPES = frozenset(
    {"integer", "number", "boolean", "string"}
)
CAPABILITY_OPERATIONS = frozenset(
    {"decrease", "increase", "enable", "disable", "set"}
)

_CONFIG_PATH_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$"
)
_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Approval can be made stricter by a project declaration, never weaker.
_PERMISSION_RANK = {
    "automatic": 0,
    "approval_required": 1,
    "disabled": 2,
}


class CapabilityError(ValueError):
    """Raised when a declaration or persisted manifest is invalid."""


class UnsupportedCapabilityError(CapabilityError):
    """Raised when a requested capability was not discovered."""


class CapabilityTransitionError(CapabilityError):
    """Raised when a proposed value violates a capability contract."""


@dataclass(frozen=True)
class _CapabilityDefinition:
    capability_id: str
    location: str
    target: Optional[str]
    aliases: Tuple[str, ...]
    value_type: str
    operations: Tuple[str, ...]
    permission: str
    risk: str
    description: str
    expected_effect: str
    semantic_change: bool
    choices: Tuple[Union[str, int, float, bool], ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None


def _definition(
    capability_id: str,
    *,
    aliases: Sequence[str] = (),
    value_type: str,
    operations: Sequence[str],
    permission: str,
    risk: str,
    description: str,
    expected_effect: str,
    semantic_change: bool,
    location: str = "config",
    target: Optional[str] = None,
    choices: Sequence[Union[str, int, float, bool]] = (),
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> _CapabilityDefinition:
    return _CapabilityDefinition(
        capability_id=capability_id,
        location=location,
        target=target,
        aliases=tuple(aliases),
        value_type=value_type,
        operations=tuple(operations),
        permission=permission,
        risk=risk,
        description=description,
        expected_effect=expected_effect,
        semantic_change=semantic_change,
        choices=tuple(choices),
        minimum=minimum,
        maximum=maximum,
    )


# Canonical semantics are intentionally independent of framework-specific key
# names. Projects with custom config layouts declare an explicit path.
_DEFINITIONS: Tuple[_CapabilityDefinition, ...] = (
    _definition(
        "micro_batch_size",
        aliases=(
            "batch_size",
            "micro_batch_size",
            "per_device_train_batch_size",
            "train_batch_size",
        ),
        value_type="integer",
        operations=("decrease",),
        permission="automatic",
        risk="low",
        minimum=1,
        description="Examples processed per device in one forward/backward pass.",
        expected_effect="Lower activation memory per training step.",
        semantic_change=False,
    ),
    _definition(
        "gradient_accumulation_steps",
        aliases=(
            "gradient_accumulation_steps",
            "grad_accumulation_steps",
            "accumulate_grad_batches",
        ),
        value_type="integer",
        operations=("increase",),
        permission="automatic",
        risk="low",
        minimum=1,
        description="Micro-batches accumulated before an optimizer update.",
        expected_effect=(
            "Preserve effective batch size after reducing micro-batch size."
        ),
        semantic_change=False,
    ),
    _definition(
        "gradient_checkpointing",
        aliases=(
            "gradient_checkpointing",
            "activation_checkpointing",
            "use_gradient_checkpointing",
        ),
        value_type="boolean",
        operations=("enable",),
        permission="automatic",
        risk="low",
        description="Recompute selected activations during backward propagation.",
        expected_effect="Trade additional compute for lower activation memory.",
        semantic_change=False,
    ),
    _definition(
        "sequence_length",
        aliases=(
            "sequence_length",
            "max_seq_length",
            "max_sequence_length",
            "block_size",
        ),
        value_type="integer",
        operations=("decrease",),
        permission="approval_required",
        risk="medium",
        minimum=1,
        description="Maximum token length processed by the training workload.",
        expected_effect="Reduce activation and attention-memory growth.",
        semantic_change=True,
    ),
    _definition(
        "precision",
        aliases=(
            "precision",
            "mixed_precision",
            "compute_dtype",
            "torch_dtype",
        ),
        value_type="string",
        operations=("set",),
        permission="approval_required",
        risk="medium",
        choices=(
            "fp32",
            "float32",
            "tf32",
            "fp16",
            "float16",
            "bf16",
            "bfloat16",
        ),
        description="Numeric format used for model computation.",
        expected_effect="Potentially reduce tensor memory and improve throughput.",
        semantic_change=True,
    ),
    _definition(
        "attention_backend",
        aliases=(
            "attention_backend",
            "attention_implementation",
            "attn_implementation",
            "attn_impl",
        ),
        value_type="string",
        operations=("set",),
        permission="approval_required",
        risk="medium",
        choices=("eager", "sdpa", "flash_attention_2", "flash_attention_3"),
        description="Attention-kernel implementation selected by the workload.",
        expected_effect="Use a more memory-efficient attention implementation.",
        semantic_change=False,
    ),
    _definition(
        "memory_efficient_attention",
        aliases=(
            "memory_efficient_attention",
            "use_memory_efficient_attention",
            "xformers_memory_efficient_attention",
        ),
        value_type="boolean",
        operations=("enable",),
        permission="approval_required",
        risk="medium",
        description="Enable a workload-provided memory-efficient attention path.",
        expected_effect="Reduce materialized attention intermediates.",
        semantic_change=False,
    ),
    _definition(
        "activation_offload",
        aliases=(
            "activation_offload",
            "offload_activations",
            "cpu_checkpointing",
        ),
        value_type="boolean",
        operations=("enable",),
        permission="approval_required",
        risk="medium",
        description="Move saved activations away from accelerator memory.",
        expected_effect="Trade transfer overhead and host RAM for lower VRAM use.",
        semantic_change=False,
    ),
    _definition(
        "optimizer_state_offload",
        aliases=(
            "optimizer_state_offload",
            "offload_optimizer",
            "optimizer_offload",
        ),
        value_type="boolean",
        operations=("enable",),
        permission="approval_required",
        risk="medium",
        description="Move optimizer state away from accelerator memory.",
        expected_effect="Reduce persistent optimizer-state VRAM.",
        semantic_change=False,
    ),
    _definition(
        "parameter_offload",
        aliases=(
            "parameter_offload",
            "offload_parameters",
            "param_offload",
        ),
        value_type="boolean",
        operations=("enable",),
        permission="approval_required",
        risk="high",
        description="Move model parameters away from accelerator memory.",
        expected_effect="Reduce persistent parameter VRAM at substantial I/O cost.",
        semantic_change=False,
    ),
    _definition(
        "optimizer_bits",
        aliases=("optimizer_bits", "optim_bits", "optimizer_state_bits"),
        value_type="integer",
        operations=("decrease",),
        permission="approval_required",
        risk="high",
        choices=(8, 16, 32),
        description="Bit width used by a workload-supported optimizer state.",
        expected_effect="Reduce persistent optimizer-state memory.",
        semantic_change=True,
    ),
    _definition(
        "model_cache",
        aliases=("use_cache", "model_cache", "kv_cache"),
        value_type="boolean",
        operations=("disable",),
        permission="approval_required",
        risk="medium",
        description="Inference-style model cache retained by the workload.",
        expected_effect="Avoid retaining cache tensors during training.",
        semantic_change=False,
    ),
    _definition(
        "allocator_configuration",
        location="environment",
        target="PYTORCH_CUDA_ALLOC_CONF",
        value_type="string",
        operations=("set",),
        permission="approval_required",
        risk="medium",
        description="PyTorch CUDA allocator behavior configured before import.",
        expected_effect="Mitigate allocator fragmentation when evidence supports it.",
        semantic_change=False,
    ),
)

_DEFINITION_BY_ID = {
    definition.capability_id: definition for definition in _DEFINITIONS
}
KNOWN_CAPABILITY_IDS = tuple(
    definition.capability_id for definition in _DEFINITIONS
)


@dataclass(frozen=True)
class CapabilityDeclaration:
    """Explicitly bind a canonical capability to a project surface.

    ``permission`` may make the built-in default stricter.  It cannot turn an
    approval-required or disabled capability into an automatic intervention.
    """

    capability_id: str
    target: Optional[str] = None
    permission: Optional[str] = None

    def __post_init__(self) -> None:
        definition = _get_definition(self.capability_id)
        target = self.target or definition.target
        if target is None:
            raise CapabilityError(
                "declaration for {!r} requires a config target".format(
                    self.capability_id
                )
            )
        _validate_target(target, definition.location)
        if (
            definition.location == "environment"
            and target != definition.target
        ):
            raise CapabilityError(
                "environment capability target is outside the allowlist"
            )
        object.__setattr__(self, "target", target)

        if self.permission is not None:
            if self.permission not in CAPABILITY_PERMISSIONS:
                raise CapabilityError(
                    "permission must be one of {}".format(
                        sorted(CAPABILITY_PERMISSIONS)
                    )
                )
            if _PERMISSION_RANK[self.permission] < _PERMISSION_RANK[
                definition.permission
            ]:
                raise CapabilityError(
                    "declaration cannot weaken {!r} from {} to {}".format(
                        self.capability_id,
                        definition.permission,
                        self.permission,
                    )
                )


@dataclass(frozen=True)
class Capability:
    """One discovered and baseline-bound recovery control."""

    capability_id: str
    location: str
    target: str
    value_type: str
    operations: Tuple[str, ...]
    permission: str
    risk: str
    current_value: Union[str, int, float, bool]
    source: str
    description: str
    expected_effect: str
    semantic_change: bool
    choices: Tuple[Union[str, int, float, bool], ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self) -> None:
        definition = _get_definition(self.capability_id)
        if self.location != definition.location:
            raise CapabilityError(
                "capability location does not match the canonical definition"
            )
        if self.value_type != definition.value_type:
            raise CapabilityError(
                "capability value_type does not match the canonical definition"
            )
        if tuple(self.operations) != definition.operations:
            raise CapabilityError(
                "capability operations do not match the canonical definition"
            )
        if self.risk != definition.risk:
            raise CapabilityError(
                "capability risk does not match the canonical definition"
            )
        if self.description != definition.description:
            raise CapabilityError(
                "capability description does not match the canonical definition"
            )
        if self.expected_effect != definition.expected_effect:
            raise CapabilityError(
                "expected_effect does not match the canonical definition"
            )
        if self.semantic_change != definition.semantic_change:
            raise CapabilityError(
                "semantic_change does not match the canonical definition"
            )
        if tuple(self.choices) != definition.choices:
            raise CapabilityError(
                "capability choices do not match the canonical definition"
            )
        if self.minimum != definition.minimum or self.maximum != definition.maximum:
            raise CapabilityError(
                "capability bounds do not match the canonical definition"
            )
        if _PERMISSION_RANK.get(self.permission, -1) < _PERMISSION_RANK[
            definition.permission
        ]:
            raise CapabilityError(
                "capability permission weakens the canonical default"
            )
        if definition.location == "environment" and self.target != definition.target:
            raise CapabilityError(
                "environment capability target does not match the allowlist"
            )
        _validate_target(self.target, self.location)
        if self.location not in CAPABILITY_LOCATIONS:
            raise CapabilityError("invalid capability location")
        if self.value_type not in CAPABILITY_VALUE_TYPES:
            raise CapabilityError("invalid capability value_type")
        if not self.operations or any(
            operation not in CAPABILITY_OPERATIONS
            for operation in self.operations
        ):
            raise CapabilityError("invalid capability operations")
        if self.permission not in CAPABILITY_PERMISSIONS:
            raise CapabilityError("invalid capability permission")
        if self.risk not in CAPABILITY_RISKS:
            raise CapabilityError("invalid capability risk")
        if self.source not in {"declared", "detected"}:
            raise CapabilityError("capability source must be declared or detected")
        _normalize_value(
            self.current_value,
            self.value_type,
            choices=self.choices,
            minimum=self.minimum,
            maximum=self.maximum,
            field_name="current_value",
        )

    def validate_transition(
        self,
        operation: str,
        proposed_value: Union[str, int, float, bool],
    ) -> Union[str, int, float, bool]:
        """Validate and normalize one proposed value transition.

        This verifies capability mechanics only.  Approval, campaign budgets,
        cross-capability invariants, and evidence requirements remain policy
        concerns.
        """
        if self.permission == "disabled":
            raise CapabilityTransitionError(
                "capability {!r} is disabled".format(self.capability_id)
            )
        if operation not in self.operations:
            raise CapabilityTransitionError(
                "operation {!r} is not allowed for {!r}; expected one of {}".format(
                    operation,
                    self.capability_id,
                    list(self.operations),
                )
            )
        try:
            normalized = _normalize_value(
                proposed_value,
                self.value_type,
                choices=self.choices,
                minimum=self.minimum,
                maximum=self.maximum,
                field_name="proposed_value",
            )
        except CapabilityError as exc:
            raise CapabilityTransitionError(str(exc)) from exc
        current = self.current_value
        if normalized == current:
            raise CapabilityTransitionError(
                "proposed value must differ from the current value"
            )
        if operation == "decrease" and not normalized < current:
            raise CapabilityTransitionError(
                "decrease requires proposed_value < current_value"
            )
        if operation == "increase" and not normalized > current:
            raise CapabilityTransitionError(
                "increase requires proposed_value > current_value"
            )
        if operation == "enable" and not (
            current is False and normalized is True
        ):
            raise CapabilityTransitionError(
                "enable requires a false-to-true transition"
            )
        if operation == "disable" and not (
            current is True and normalized is False
        ):
            raise CapabilityTransitionError(
                "disable requires a true-to-false transition"
            )
        return normalized

    def to_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "location": self.location,
            "target": self.target,
            "value_type": self.value_type,
            "operations": list(self.operations),
            "permission": self.permission,
            "risk": self.risk,
            "current_value": self.current_value,
            "source": self.source,
            "description": self.description,
            "expected_effect": self.expected_effect,
            "semantic_change": self.semantic_change,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Capability":
        if not isinstance(payload, dict):
            raise CapabilityError("capability payload must be an object")
        try:
            return cls(
                capability_id=payload["capability_id"],
                location=payload["location"],
                target=payload["target"],
                value_type=payload["value_type"],
                operations=tuple(payload["operations"]),
                permission=payload["permission"],
                risk=payload["risk"],
                current_value=payload["current_value"],
                source=payload["source"],
                description=payload["description"],
                expected_effect=payload["expected_effect"],
                semantic_change=payload["semantic_change"],
                choices=tuple(payload.get("choices") or ()),
                minimum=payload.get("minimum"),
                maximum=payload.get("maximum"),
            )
        except (KeyError, TypeError) as exc:
            raise CapabilityError(
                "capability payload is missing or contains an invalid field"
            ) from exc


@dataclass(frozen=True)
class CapabilityIssue:
    """A conservative-discovery decision that callers should display."""

    code: str
    capability_id: str
    targets: Tuple[str, ...]
    message: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "capability_id": self.capability_id,
            "targets": list(self.targets),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CapabilityIssue":
        if not isinstance(payload, dict):
            raise CapabilityError("capability issue must be an object")
        try:
            return cls(
                code=payload["code"],
                capability_id=payload["capability_id"],
                targets=tuple(payload.get("targets") or ()),
                message=payload["message"],
            )
        except (KeyError, TypeError) as exc:
            raise CapabilityError("invalid capability issue payload") from exc


@dataclass(frozen=True)
class CapabilityManifest:
    """Stable, serializable capability evidence for one source config."""

    capabilities: Tuple[Capability, ...]
    issues: Tuple[CapabilityIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ids = [capability.capability_id for capability in self.capabilities]
        if len(ids) != len(set(ids)):
            raise CapabilityError("capability ids must be unique")
        targets = [
            (capability.location, capability.target)
            for capability in self.capabilities
        ]
        if len(targets) != len(set(targets)):
            raise CapabilityError("capability targets must be unique")

    def supports(self, capability_id: str) -> bool:
        return any(
            capability.capability_id == capability_id
            for capability in self.capabilities
        )

    def get(self, capability_id: str) -> Optional[Capability]:
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        return None

    def require(self, capability_id: str) -> Capability:
        capability = self.get(capability_id)
        if capability is None:
            raise UnsupportedCapabilityError(
                "capability {!r} was not declared or safely detected".format(
                    capability_id
                )
            )
        return capability

    def to_dict(self) -> dict:
        return {
            "schema": {
                "name": CAPABILITY_SCHEMA_NAME,
                "version": CAPABILITY_SCHEMA_VERSION,
            },
            "capabilities": [
                capability.to_dict() for capability in self.capabilities
            ],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> "CapabilityManifest":
        if not isinstance(payload, dict):
            raise CapabilityError("capability manifest must be an object")
        schema = payload.get("schema") or {}
        if schema.get("name") != CAPABILITY_SCHEMA_NAME:
            raise CapabilityError(
                "capability schema.name must be {!r}".format(
                    CAPABILITY_SCHEMA_NAME
                )
            )
        if schema.get("version") != CAPABILITY_SCHEMA_VERSION:
            raise CapabilityError(
                "capability schema.version must be {!r}".format(
                    CAPABILITY_SCHEMA_VERSION
                )
            )
        capabilities_payload = payload.get("capabilities")
        issues_payload = payload.get("issues") or []
        if not isinstance(capabilities_payload, list):
            raise CapabilityError("capabilities must be an array")
        if not isinstance(issues_payload, list):
            raise CapabilityError("issues must be an array")
        return cls(
            capabilities=tuple(
                Capability.from_dict(item) for item in capabilities_payload
            ),
            issues=tuple(
                CapabilityIssue.from_dict(item) for item in issues_payload
            ),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "CapabilityManifest":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CapabilityError("invalid capability manifest JSON") from exc
        return cls.from_dict(payload)


DeclarationInput = Union[
    Mapping[str, Union[str, CapabilityDeclaration]],
    Iterable[CapabilityDeclaration],
]


def discover_capabilities(
    config: dict,
    *,
    declarations: Optional[DeclarationInput] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> CapabilityManifest:
    """Build a deterministic capability manifest from declared surfaces.

    ``environment`` is never read implicitly.  A caller may pass a filtered or
    complete environment mapping; only built-in, non-secret capability keys are
    examined.  Explicit declarations take precedence over alias detection.
    """
    try:
        normalized_config = validate_config(config)
    except EntrypointError as exc:
        raise CapabilityError(str(exc)) from exc
    normalized_declarations = _normalize_declarations(declarations)
    normalized_environment = _normalize_environment(environment)

    leaves = tuple(_iter_config_leaves(normalized_config))
    capabilities = []
    issues = []
    used_targets = set()

    for definition in _DEFINITIONS:
        declaration = normalized_declarations.get(definition.capability_id)
        if declaration is not None:
            capability = _capability_from_declaration(
                definition,
                declaration,
                normalized_config,
                normalized_environment,
            )
            _claim_target(capability, used_targets)
            capabilities.append(capability)
            continue

        if definition.location == "environment":
            target = definition.target
            if target is not None and target in normalized_environment:
                try:
                    capability = _make_capability(
                        definition,
                        target=target,
                        current_value=normalized_environment[target],
                        permission=definition.permission,
                        source="detected",
                    )
                except CapabilityError as exc:
                    issues.append(
                        CapabilityIssue(
                            code="invalid_detected_value",
                            capability_id=definition.capability_id,
                            targets=(target,),
                            message=str(exc),
                        )
                    )
                else:
                    _claim_target(capability, used_targets)
                    capabilities.append(capability)
            continue

        matches = [
            (path, value)
            for path, leaf_name, value in leaves
            if leaf_name in definition.aliases
        ]
        if len(matches) == 1:
            path, value = matches[0]
            try:
                capability = _make_capability(
                    definition,
                    target=path,
                    current_value=value,
                    permission=definition.permission,
                    source="detected",
                )
            except CapabilityError as exc:
                issues.append(
                    CapabilityIssue(
                        code="invalid_detected_value",
                        capability_id=definition.capability_id,
                        targets=(path,),
                        message=str(exc),
                    )
                )
            else:
                _claim_target(capability, used_targets)
                capabilities.append(capability)
        elif len(matches) > 1:
            targets = tuple(sorted(path for path, _ in matches))
            issues.append(
                CapabilityIssue(
                    code="ambiguous_aliases",
                    capability_id=definition.capability_id,
                    targets=targets,
                    message=(
                        "multiple config paths match {!r}; declare the intended "
                        "target explicitly".format(definition.capability_id)
                    ),
                )
            )

    return CapabilityManifest(
        capabilities=tuple(capabilities),
        issues=tuple(issues),
    )


def known_capabilities() -> Tuple[dict, ...]:
    """Return public metadata for every canonical V1 capability."""
    return tuple(
        {
            "capability_id": definition.capability_id,
            "location": definition.location,
            "default_target": definition.target,
            "aliases": list(definition.aliases),
            "value_type": definition.value_type,
            "operations": list(definition.operations),
            "default_permission": definition.permission,
            "risk": definition.risk,
            "description": definition.description,
            "expected_effect": definition.expected_effect,
            "semantic_change": definition.semantic_change,
            "choices": list(definition.choices),
            "minimum": definition.minimum,
            "maximum": definition.maximum,
        }
        for definition in _DEFINITIONS
    )


def get_config_value(config: dict, target: str):
    """Resolve a validated dotted path from a JSON configuration."""
    _validate_target(target, "config")
    value = config
    for segment in target.split("."):
        if not isinstance(value, dict) or segment not in value:
            raise CapabilityError(
                "config target {!r} does not exist".format(target)
            )
        value = value[segment]
    return value


def _normalize_declarations(
    declarations: Optional[DeclarationInput],
) -> Dict[str, CapabilityDeclaration]:
    if declarations is None:
        return {}
    items = []
    if isinstance(declarations, Mapping):
        for capability_id, value in declarations.items():
            if isinstance(value, CapabilityDeclaration):
                if value.capability_id != capability_id:
                    raise CapabilityError(
                        "declaration mapping key does not match capability_id"
                    )
                items.append(value)
            elif isinstance(value, str):
                items.append(
                    CapabilityDeclaration(
                        capability_id=capability_id,
                        target=value,
                    )
                )
            else:
                raise CapabilityError(
                    "declaration values must be paths or CapabilityDeclaration"
                )
    else:
        try:
            items = list(declarations)
        except TypeError as exc:
            raise CapabilityError("declarations must be a mapping or iterable") from exc
        if any(not isinstance(item, CapabilityDeclaration) for item in items):
            raise CapabilityError(
                "declaration iterable must contain CapabilityDeclaration values"
            )

    normalized = {}
    for declaration in items:
        if declaration.capability_id in normalized:
            raise CapabilityError(
                "duplicate declaration for {!r}".format(
                    declaration.capability_id
                )
            )
        normalized[declaration.capability_id] = declaration
    return normalized


def _normalize_environment(
    environment: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    if environment is None:
        return {}
    if not isinstance(environment, Mapping):
        raise CapabilityError("environment must be a string mapping")
    allowed_keys = {
        definition.target
        for definition in _DEFINITIONS
        if definition.location == "environment"
        and definition.target is not None
    }
    normalized = {}
    for key in allowed_keys:
        if key in environment:
            value = environment[key]
            if not isinstance(value, str):
                raise CapabilityError(
                    "environment capability values must be strings"
                )
            normalized[key] = value
    return normalized


def _capability_from_declaration(
    definition: _CapabilityDefinition,
    declaration: CapabilityDeclaration,
    config: dict,
    environment: Mapping[str, str],
) -> Capability:
    target = declaration.target
    if definition.location == "config":
        current_value = get_config_value(config, target)
    else:
        if target not in environment:
            raise CapabilityError(
                "declared environment target {!r} has no baseline value".format(
                    target
                )
            )
        current_value = environment[target]
    return _make_capability(
        definition,
        target=target,
        current_value=current_value,
        permission=declaration.permission or definition.permission,
        source="declared",
    )


def _make_capability(
    definition: _CapabilityDefinition,
    *,
    target: str,
    current_value,
    permission: str,
    source: str,
) -> Capability:
    normalized_value = _normalize_value(
        current_value,
        definition.value_type,
        choices=definition.choices,
        minimum=definition.minimum,
        maximum=definition.maximum,
        field_name="current value for {}".format(definition.capability_id),
    )
    return Capability(
        capability_id=definition.capability_id,
        location=definition.location,
        target=target,
        value_type=definition.value_type,
        operations=definition.operations,
        permission=permission,
        risk=definition.risk,
        current_value=normalized_value,
        source=source,
        description=definition.description,
        expected_effect=definition.expected_effect,
        semantic_change=definition.semantic_change,
        choices=definition.choices,
        minimum=definition.minimum,
        maximum=definition.maximum,
    )


def _normalize_value(
    value,
    value_type: str,
    *,
    choices: Sequence[Union[str, int, float, bool]],
    minimum: Optional[float],
    maximum: Optional[float],
    field_name: str,
):
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise CapabilityError("{} must be a boolean".format(field_name))
        normalized = value
    elif value_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CapabilityError("{} must be an integer".format(field_name))
        normalized = value
    elif value_type == "number":
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CapabilityError("{} must be a finite number".format(field_name))
        normalized = float(value)
        if not math.isfinite(normalized):
            raise CapabilityError("{} must be a finite number".format(field_name))
    elif value_type == "string":
        if not isinstance(value, str) or not value:
            raise CapabilityError(
                "{} must be a non-empty string".format(field_name)
            )
        normalized = value
    else:
        raise CapabilityError("unknown capability value type")

    if choices and normalized not in choices:
        raise CapabilityError(
            "{} must be one of {}".format(field_name, list(choices))
        )
    if minimum is not None and normalized < minimum:
        raise CapabilityError(
            "{} must be at least {}".format(field_name, minimum)
        )
    if maximum is not None and normalized > maximum:
        raise CapabilityError(
            "{} must be at most {}".format(field_name, maximum)
        )
    return normalized


def _iter_config_leaves(value: dict, prefix: str = ""):
    for key in sorted(value):
        child = value[key]
        path = "{}.{}".format(prefix, key) if prefix else key
        if isinstance(child, dict):
            yield from _iter_config_leaves(child, path)
        else:
            yield path, key, child


def _claim_target(capability: Capability, used_targets: set) -> None:
    target = (capability.location, capability.target)
    if target in used_targets:
        raise CapabilityError(
            "surface {}:{} was bound to more than one capability".format(
                capability.location,
                capability.target,
            )
        )
    used_targets.add(target)


def _get_definition(capability_id: str) -> _CapabilityDefinition:
    if not isinstance(capability_id, str) or capability_id not in _DEFINITION_BY_ID:
        raise CapabilityError(
            "unknown capability_id {!r}; expected one of {}".format(
                capability_id,
                list(KNOWN_CAPABILITY_IDS),
            )
        )
    return _DEFINITION_BY_ID[capability_id]


def _validate_target(target: str, location: str) -> None:
    if not isinstance(target, str) or not target:
        raise CapabilityError("capability target must be a non-empty string")
    if location == "config":
        if not _CONFIG_PATH_PATTERN.fullmatch(target):
            raise CapabilityError(
                "config target must be a dotted path of safe key names"
            )
    elif location == "environment":
        if not _ENVIRONMENT_KEY_PATTERN.fullmatch(target):
            raise CapabilityError("invalid environment capability target")
    else:
        raise CapabilityError("invalid capability location {!r}".format(location))


__all__ = [
    "CAPABILITY_SCHEMA_NAME",
    "CAPABILITY_SCHEMA_VERSION",
    "KNOWN_CAPABILITY_IDS",
    "Capability",
    "CapabilityDeclaration",
    "CapabilityError",
    "CapabilityIssue",
    "CapabilityManifest",
    "CapabilityTransitionError",
    "UnsupportedCapabilityError",
    "discover_capabilities",
    "get_config_value",
    "known_capabilities",
]