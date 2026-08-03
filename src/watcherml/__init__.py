"""WatcherML: a local-first experiment flight recorder for notebook-first ML.

    import watcherml as watcher

    with watcher.init(project="tomato-disease", config={"model": "resnet50", "lr": 2e-4}) as run:
        run.set_dataset("./data/tomato")
        ... training loop ...
        run.log_metric("val_accuracy", 0.914)

Every run leaves a receipt. Every failure leaves evidence.
"""
from .autopilot import autopilot
from .recovery import RecoveryContract, recover_from_oom
from .run import Run, init
from .storage import Storage

__version__ = "0.1.0"
__all__ = ["init", "Run", "Storage", "autopilot", "recover_from_oom", "RecoveryContract", "__version__"]

# IPython looks for these two names directly on the top-level module when you
# run `%load_ext watcherml` — so they must live here, not just in .notebook.
try:
    from .notebook import load_ipython_extension, unload_ipython_extension  # noqa: F401
    __all__ += ["load_ipython_extension", "unload_ipython_extension"]
except ImportError:
    # IPython isn't installed — fine outside notebooks, `%load_ext` just won't apply.
    pass