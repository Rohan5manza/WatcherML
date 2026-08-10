# WatcherML

A local-first experiment flight recorder for notebook-first ML.

Every experiment leaves a receipt. Every failure leaves evidence.

WatcherML captures the code, environment, data fingerprint, and failure
context needed to reproduce and debug every training run — without
requiring Docker, a database, or an internet connection.

## Install (from this repo, for now)

```bash
pip install -e .
```

(Once published: `pip install watcherml`.)

## Quickstart

```python
import watcherml as watcher

with watcher.init(project="tomato-disease", config={"model": "resnet50", "lr": 2e-4}) as run:
    run.set_dataset("./data/tomato")
    for step in range(epochs):
        acc = train_one_epoch(...)
        run.log_metric("val_accuracy", acc, step=step)
```

That's it — no Git repo setup, no DVC init, no manual instrumentation required
to get your first recorded run. (Those tools remain available for teams that
want stricter provenance; they are just no longer prerequisites for run #1.)

At the end of a run you get a receipt like:

```
WatcherML run completed: tomato-disease-cc0441

val_accuracy       0.797
duration           0m 0s
git_state          clean
dataset_fingerprint 912a76cd9cd0
reproduction_score 9/10

Inspect: watcher inspect tomato-disease-cc0441
```

If the run crashes, you get a **failure capsule** instead — deterministic
diagnosis (CUDA OOM, NaN loss, shape mismatch, missing file, DataLoader
worker crash, device mismatch, or dependency/CUDA incompatibility), plus the
evidence behind it. No LLM or API key required for this to work; an optional
LLM explanation can be layered on top later without becoming the source of
truth.

## Notebook usage

```
%load_ext watcherml
%watcher project tomato-disease

import watcherml as watcher
run = watcher.init(config={"model": "resnet50", "lr": 2e-4})   # project comes from %watcher
run.log_metric("val_accuracy", 0.91, step=1)
run.finish()   # marks success — or just let the notebook keep running
```

No `with` block needed. If a later cell raises instead, WatcherML notices
automatically and saves a failure capsule — that's the point: notebooks are
run cell-by-cell, not inside one tidy `with` block, so recording has to work
the way people actually iterate. Executed cell source and order are attached
to the run and included in any failure capsule. Requires `pip install
watcherml[notebook]` (just IPython).

## Optional: local LLM advisor (Ollama)

Everything above works with zero LLM involvement — the diagnosis and diff
are always deterministic. If you also run [Ollama](https://ollama.com)
locally (`ollama serve`, with a small model pulled — `ollama pull llama3.2`
or similar), you can layer a plain-language explanation on top:

```bash
watcher inspect RUN_ID --advise          # explain a failure capsule
watcher compare RUN_A RUN_B --advise     # "likely explanation" for what changed
watcher advise RUN_ID                    # same as inspect --advise, standalone
```

If Ollama isn't running, these just print a note and fall back to the
deterministic output — nothing breaks, and nothing requires an API key.

## Optional: autopilot (bounded, opt-in iteration)

```python
from watcherml import autopilot

def train(config):
    ...  # your training code, parameterized by config
    return {"val_accuracy": acc}   # or raise, e.g. on CUDA OOM

result = autopilot(
    project="tomato-disease",
    train_fn=train,
    base_config={"model": "resnet50", "lr": 1e-3, "batch_size": 32},
    goal_metric="val_accuracy",
    goal_direction="maximize",
    max_iterations=5,   # hard-capped at 10 regardless of what you pass
)
```

This runs your training function repeatedly, asking Ollama (or a
deterministic fallback if Ollama isn't running — e.g. halving batch_size on
OOM) what to try next after each iteration. **This is explicitly
experimental and bounded, not a background agent:**

- every iteration is logged as a completely normal, independently
  inspectable WatcherML run — nothing is hidden
- it stops and tells you why (converged suggestion, unrecoverable failure,
  or hitting the iteration cap) rather than looping indefinitely
- it never modifies anything outside WatcherML's own storage, and never
  deploys or promotes a result on its own
- a handful of runs is not much evidence — treat the "best run" as a lead
  worth reviewing yourself (`watcher inspect <best_run_id>`), not a decision

## Web UI

```bash
pip install watcherml[ui]
watcher ui                # opens http://localhost:7331
```

A full redesign, built around one idea: **color and labels carry provenance,
not just decoration.** Mint green means "verified" or "live," violet means
"Ollama generated this, unverified," and everything computed deterministically
gets a "calculated" or "rule-based" tag right next to it — so nothing an LLM
said can quietly pass as fact.

Sidebar: **Overview, Projects, Runs, Failures, Campaigns, Memory, Settings.**

- **Overview** — real aggregated stats: total projects/runs, runs needing
  attention (failed + unresolved), active recovery campaigns, GPU/Ollama
  status, recent verified fixes
- **Runs** (global, cross-project) — human-readable names
  (`resnet50 — batch 32`, with inline rename + tagging), filterable by
  status, with hardware, warning count, and failure category as real columns
- **Run detail** — metrics, config, reproduction status, a real GPU/CPU
  utilization trace, a lightweight event timeline (start/warnings/failure —
  not full live monitoring yet, see `ROADMAP.md`), export-capsule download,
  mark-as-resolved
- **Failures** and **Failure capsule** — diagnosis with **evidence IDs**
  (`EV-1, EV-2` linking a diagnosis to the specific evidence category behind
  it), similarity-based nearest-successful-run comparison (not just "the
  most recent success" — see below), action bar (Analyze locally / Compare
  baseline / Create recovery campaign / Export capsule / Mark as resolved)
- **Campaigns** — real trial lineage from `recover_from_oom`: agent
  reasoning steps, an objective sparkline, a trial table with decision pills
  (keep/accept/best/rejected) — nothing here is mockup content, it's your
  actual recovery campaign data
- **Memory** — cross-campaign resolution memory ("this failure signature,
  this fix, verified in N/M matching attempts"), aggregated live from
  `recovery_trials` — not a separately tracked concept, it falls out of data
  you already have the moment more than one campaign has run
- **Settings** — local data directory, Ollama status/host/model, GPU info

Runs created by the bundled demo scripts are tagged `_simulated: true` in
their config and show a **"Simulated OOM Scenario"** badge — real
instrumented runs never get this badge.

## Real nearest-successful-run selection

Comparing a failure against "the last successful run" is misleading —
recency isn't relevance. `compare_to_last_success` now scores every
successful run in the project by a transparent, documented similarity
weighting (dataset fingerprint, model architecture, GPU, Git ancestry,
config distance, framework versions, temporal proximity — see
`similarity.py` for the exact weights) and picks the best match, with a
checklist explaining why: same dataset ✓, same GPU ✓, 11 of 13 config
fields identical, etc. Config differences from that baseline are ranked too
— known memory/throughput-sensitive keys (`batch_size`, `precision`,
`sequence_length`, ...) surface first.

## Optional: OOM Recovery Agent (experimental, narrow by design)

```python
import watcherml as watcher

def train(config, max_steps=None):
    # your training code -- max_steps lets the agent probe cheaply before
    # committing to a full run; if you don't support it, full runs are used
    # for probing too (safe, just not cheap)
    ...
    return {"val_accuracy": acc, "throughput_samples_per_sec": tp}

report = watcher.recover_from_oom(
    project="tomato-disease",
    failed_run_id="tomato-disease-a56b75",   # a run that failed with CUDA OOM
    train_fn=train,
    contract=watcher.RecoveryContract(
        goal_metric="val_accuracy",
        max_trials=6,
        probe_steps=30,
    ),
)
```

This is the first slice of a broader autonomous-recovery design, scoped
deliberately to exactly one failure class. Given a run that failed with a
CUDA OOM, it:

1. **Observes** the failure capsule's evidence (facts only, no LLM)
2. **Diagnoses** likely memory causes via Ollama, ranked with confidence
   (falls back to a generic hypothesis if Ollama isn't running)
3. **Plans** 2-3 candidate patches via Ollama, restricted to exactly six
   keys: `batch_size`, `gradient_accumulation_steps`, `precision`,
   `sequence_length`, `gradient_checkpointing`, `num_workers`
4. **Validates** every proposed patch through a policy engine before
   anything runs — any other key the LLM proposes is silently rejected and
   counted, never executed
5. **Probes** each candidate cheaply (short trials) and eliminates ones that
   still OOM, before running survivors as full trials
6. **Scores** survivors deterministically (success + goal metric +
   throughput + VRAM headroom — a documented heuristic, not the LLM's
   opinion) and reports which fix was actually verified
7. **Remembers** every hypothesis, patch, and outcome — `watcher recoveries`
   and `watcher recovery CAMPAIGN_ID` show the full audit trail, and every
   trial is a completely normal, independently inspectable WatcherML run

Guardrails, on purpose: trial count is hard-capped regardless of what you
ask for; this agent only ever touches config (never code, dependencies, or
datasets — no git worktree isolation is needed yet because there's no code
to isolate); and if you're on a single-GPU box, Ollama calls default to
`keep_alive=0` so the model unloads immediately and a training trial gets
the full card. The metrics your `train_fn` returns should be validation
metrics — this agent picks a "best" by comparing candidates against each
other on whatever you log, so if that's test-set performance you'll be
optimizing against your test set by accident. Evaluate the winner against a
test set once, separately, after the campaign ends.

## CLI

```bash
watcher init                       # set up ./.watcherml (SQLite + local artifact store)
watcher runs                       # list recorded runs
watcher inspect RUN_ID [--advise]  # full detail or failure capsule for one run
watcher failures                   # list every recorded failure
watcher compare RUN_A RUN_B [--advise]   # structured diff: what changed, what improved
watcher advise RUN_ID              # AI (Ollama) explanation for a past failure
watcher export RUN_ID --format capsule   # portable reproduction bundle (.zip)
watcher recoveries                 # list OOM recovery campaigns
watcher recovery CAMPAIGN_ID        # full audit trail for one campaign
```

## See it work end to end

```bash
python examples/demo_story.py
```

This runs the full launch story: a CUDA-OOM-triggering run, a failure
capsule, a fixed rerun, a structured comparison, and a reproduction capsule
export — all in one script, no GPU required (it simulates the OOM condition
so the diagnosis engine can be exercised anywhere).

## What v0.1 is (and isn't)

**In scope:** SDK + `init()`/context-manager run lifecycle, Git/environment/GPU
capture, CPU/RAM/GPU background sampling, metric + artifact logging,
deterministic failure classification, structured run diff, portable
reproduction capsules, and a CLI.



## Two modes (server mode not yet implemented in this scaffold)

- **Local mode** (implemented here): SQLite + local content-addressed
  artifact directory. Fully offline. No Docker.
- **Server mode** (planned): FastAPI + Postgres + S3/MinIO, for team/Buffy
  deployments. See `ROADMAP.md`.

## License

MIT (change if you'd prefer Apache-2.0 — see `ROADMAP.md` security/launch
checklist).