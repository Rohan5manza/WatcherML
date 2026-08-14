"""WatcherML's public Python SDK.

WatcherML records ML runs, captures deterministic failure evidence, and runs
bounded CUDA OOM recovery campaigns whose verdicts require independent
confirmation.
"""

from ._version import __version__
from .capsule_schema import CAPSULE_SCHEMA_VERSION
from .entrypoint import TrainingEntrypoint, validate_entrypoint
from .recovery import (
    RecoveryPreparation,
    RecoveryResult,
    prepare_oom_recovery,
    recover_from_oom,
    run_prepared_recovery,
)
from .recovery_contract import (
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    RecoveryContract,
    VerificationRequirements,
    WorkloadIdentity,
)
from .run import Run, init
from .storage import Storage

__all__ = [
    "__version__",
    "CAPSULE_SCHEMA_VERSION",
    "init",
    "Run",
    "Storage",
    "TrainingEntrypoint",
    "validate_entrypoint",
    "MetricGuard",
    "RecoveryBudget",
    "VerificationRequirements",
    "WorkloadIdentity",
    "InterventionPermissions",
    "RecoveryContract",
    "RecoveryPreparation",
    "RecoveryResult",
    "prepare_oom_recovery",
    "run_prepared_recovery",
    "recover_from_oom",
]
