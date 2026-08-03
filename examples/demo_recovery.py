"""
Demo: the OOM Recovery Agent, end to end, no GPU or Ollama required.

  1. Run something that fails with a CUDA OOM (simulated here, same as
     examples/demo_story.py).
  2. Hand that failed run to watcher.recover_from_oom() along with a
     training function.
  3. It observes the failure, proposes candidate fixes (deterministic
     fallback candidates if Ollama isn't running, LLM-guided ones if it is),
     probes them cheaply, runs full trials on survivors, and reports which
     fix was actually verified to work.

Run it with:
    python examples/demo_recovery.py

Then inspect the results:
    watcher recoveries
    watcher recovery <campaign_id>
    watcher ui   # see it on the Campaigns and Memory pages
"""
import watcherml as watcher
from watcherml.storage import Storage

storage = Storage()

# Step 1: produce a real OOM failure to recover from.
print("=" * 70)
print("STEP 1: run with batch_size=32 -> triggers CUDA OOM")
print("=" * 70)
try:
    with watcher.init(
        project="tomato-disease",
        config={"model": "resnet50", "lr": 1e-3, "batch_size": 32},
        storage=storage,
    ) as run:
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 2.10 GiB (GPU 0; 11.00 GiB total capacity)"
        )
except RuntimeError:
    pass  # WatcherML already saved the failure capsule

failed_run_id = run.run_id
print(f"\nFailed run: {failed_run_id}\n")


# Step 2: a stand-in training function. Real usage: this is your actual
# training loop, parameterized by config. max_steps is optional -- supporting
# it lets the agent probe cheaply before committing to full runs.
def train(config, max_steps=None):
    if config.get("batch_size", 32) >= 32:
        raise RuntimeError(
            "CUDA out of memory. Tried to allocate 2.10 GiB (GPU 0; 11.00 GiB total capacity)"
        )
    steps = max_steps or 100
    return {
        "val_accuracy": 0.85 + (0.02 if config.get("precision") == "bf16" else 0),
        "throughput_samples_per_sec": 340 if max_steps is None else 120,
    }


# Step 3: run the recovery campaign.
print("=" * 70)
print("STEP 2: recover_from_oom() -- observe, diagnose, plan, probe, evaluate")
print("=" * 70)
report = watcher.recover_from_oom(
    project="tomato-disease",
    failed_run_id=failed_run_id,
    train_fn=train,
    contract=watcher.RecoveryContract(
        goal_metric="val_accuracy",
        throughput_metric="throughput_samples_per_sec",
        max_vram_gb=11.0,
        max_trials=6,
        probe_steps=30,
    ),
    storage=storage,
)

print("\n" + "=" * 70)
print("Next steps to explore this yourself:")
print("=" * 70)
print(f"  watcher recoveries")
print(f"  watcher recovery {report['campaign_id']}")
print(f"  watcher ui   # then open Campaigns and Memory in the sidebar")