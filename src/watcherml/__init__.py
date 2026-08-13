"""WatcherML: a local-first recovery layer for ML experiments.

    import watcherml as watcher

    with watcher.init(project="tomato-disease", config={"model": "resnet50", "lr": 2e-4}) as run:
        run.set_dataset("./data/tomato")
        ... training loop ...
        run.log_metric("val_accuracy", 0.914)

Every run leaves a receipt. Every failure leaves evidence. WatcherML helps you investigate failures and recover from them.
"""
from .recovery import RecoveryContract, recover_from_oom
from .run import Run, init
from .storage import Storage
from .capsule_schema import CAPSULE_SCHEMA_VERSION
from .entrypoint import TrainingEntrypoint, validate_entrypoint


__version__ = "0.1.0"

__all__ = [
    "init",
    "Run",
    "Storage",
    "recover_from_oom",
    "RecoveryContract",
    "__version__",
    "TrainingEntrypoint",
"validate_entrypoint",
"CAPSULE_SCHEMA_VERSION"
]