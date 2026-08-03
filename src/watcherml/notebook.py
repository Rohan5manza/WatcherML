"""Notebook integration: %load_ext watcherml.

    %load_ext watcherml
    %watcher project tomato-disease

    import watcherml as watcher
    run = watcher.init(config={"model": "resnet50", "lr": 2e-4})   # project comes from %watcher
    ...

Records which cells executed, in what order, and whether each succeeded —
attached to whatever run is currently active — without requiring the user
to wrap their whole notebook in a `with` block (which doesn't match how
people actually iterate cell-by-cell in Jupyter).
"""
from __future__ import annotations

import time
from typing import Optional

try:
    from IPython.core.magic import Magics, line_magic, magics_class
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The watcherml notebook extension requires IPython. Install it with "
        "`pip install ipython` (already a dependency inside Jupyter)."
    ) from e

_active_project: dict = {"name": None}
_active_run = None  # set by watcherml.run.init() when created inside a notebook


def get_active_project_default() -> Optional[str]:
    return _active_project.get("name")


def set_active_run(run):
    global _active_run
    _active_run = run
    run._notebook_cells = []


def clear_active_run(run):
    global _active_run
    if _active_run is run:
        _active_run = None


@magics_class
class WatcherMagics(Magics):
    @line_magic
    def watcher(self, line):
        """Usage:
            %watcher project <name>     set the default project for watcher.init()
            %watcher status             show the currently active run, if any
        """
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0] == "project":
            _active_project["name"] = parts[1]
            print(f"WatcherML: default project set to '{parts[1]}'.")
        elif parts and parts[0] == "status":
            if _active_run is None:
                print("WatcherML: no active run.")
            else:
                print(f"WatcherML: active run '{_active_run.run_id}' "
                      f"({len(getattr(_active_run, '_notebook_cells', []))} cells recorded so far).")
        else:
            print("Usage: %watcher project <name>   |   %watcher status")


from . import redact


def _pre_run_cell(info):
    if _active_run is None:
        return
    _active_run._notebook_cells.append({
        "execution_count": None,  # filled in on post_run_cell, once IPython assigns it
        "source": redact.redact(info.raw_cell),
        "start": time.time(),
        "end": None,
        "success": None,
        "error": None,
    })


def _post_run_cell(result):
    if _active_run is None or not getattr(_active_run, "_notebook_cells", None):
        return
    entry = _active_run._notebook_cells[-1]
    entry["end"] = time.time()
    entry["execution_count"] = getattr(result, "execution_count", None)
    entry["success"] = result.success
    if not result.success and result.error_in_exec is not None:
        entry["error"] = repr(result.error_in_exec)
        exc = result.error_in_exec
        run_to_fail = _active_run
        run_to_fail._notebook_auto_fail(type(exc), exc, exc.__traceback__)
        clear_active_run(run_to_fail)


def load_ipython_extension(ipython):
    ipython.register_magics(WatcherMagics)
    ipython.events.register("pre_run_cell", _pre_run_cell)
    ipython.events.register("post_run_cell", _post_run_cell)
    print("WatcherML notebook integration loaded. "
          "Try: %watcher project <name>, then watcher.init(config={...}).")


def unload_ipython_extension(ipython):
    ipython.events.unregister("pre_run_cell", _pre_run_cell)
    ipython.events.unregister("post_run_cell", _post_run_cell)
