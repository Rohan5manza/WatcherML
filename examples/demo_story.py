"""
The WatcherML launch demo story, runnable end to end on any machine
(no real GPU required -- this simulates a CUDA OOM to exercise the failure
capsule and diagnosis engine exactly as they'd behave on real hardware):

  1. Run a training loop with a batch size that blows up GPU memory.
  2. WatcherML captures the executed code state, environment, and the failure.
  3. The failure capsule identifies the likely cause.
  4. Reduce the batch size and rerun successfully.
  5. Compare the failed and successful runs.
  6. Export a reproduction capsule for the successful run.

Run it with:
    python examples/demo_story.py
"""
import random
import time

import watcherml as watcher
from watcherml.diff import compare_runs, format_diff_report
from watcherml.export import export_capsule
from watcherml.storage import Storage


def fake_training_step(batch_size, lr, step):
    """Pretend to train. Raises a CUDA-style OOM if batch_size is too large."""
    if batch_size >= 32:
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 2.10 GiB (GPU 0; 11.00 GiB total capacity; "
            "9.32 GiB already allocated; 1.14 GiB free)"
        )
    time.sleep(0.05)
    # fake but plausible accuracy curve
    return 0.55 + 0.35 * (1 - pow(2.71828, -step / 4)) + random.uniform(-0.01, 0.01)


def run_experiment(run_id_label, batch_size, lr):
    with watcher.init(
        project="tomato-disease",
        config={"model": "resnet50", "lr": lr, "batch_size": batch_size, "seed": 42,
                "_simulated": True},  # flags this run as demo/simulated data in the UI
    ) as run:
        run.set_dataset("./examples")  # stand-in "dataset" for the demo
        for step in range(1, 6):
            acc = fake_training_step(batch_size, lr, step)
            run.log_metric("val_accuracy", acc, step=step)
        return run.run_id


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1-3: run with batch_size=32 -> triggers CUDA OOM, saves failure capsule")
    print("=" * 70)
    failed_run_id = None
    try:
        failed_run_id = run_experiment("oom", batch_size=32, lr=1e-3)
    except RuntimeError:
        pass  # WatcherML already printed + saved the failure capsule

    # We need the run_id even though the exception aborted normal flow.
    # In real usage you'd read this from `watcher runs` -- here we grab the latest.
    storage = Storage()
    latest_failed = storage.list_runs(project="tomato-disease")[0]
    failed_run_id = latest_failed["run_id"]

    print("\n" + "=" * 70)
    print("STEP 4: reduce batch_size to 16, lower learning rate -> succeeds")
    print("=" * 70)
    success_run_id = run_experiment("fixed", batch_size=16, lr=2e-4)

    print("\n" + "=" * 70)
    print("STEP 5: structured comparison")
    print("=" * 70)
    diff = compare_runs(storage, failed_run_id, success_run_id)
    print(format_diff_report(diff))

    print("\n" + "=" * 70)
    print("STEP 6: export a portable reproduction capsule for the successful run")
    print("=" * 70)
    out_path = export_capsule(storage, success_run_id, out_path=f"examples/{success_run_id}.zip")
    print(f"Capsule written to: {out_path}")