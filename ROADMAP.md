# Roadmap

This repo currently implements: the recorder core (SDK, Git/env/GPU capture,
failure capsules, deterministic diagnosis, structured run diff, reproduction
capsule export, CLI), the **notebook integration** (`%load_ext watcherml`,
automatic failure capture without a `with` block), an **optional Ollama
advisor** (plain-language failure/diff explanations, all clearly labeled and
fully optional), a **deterministic, bounded CUDA OOM recovery workflow** iteration loop, the **OOM
Recovery Agent** (`watcher.recover_from_oom` — Observe/Diagnose/Plan/
Validate/Probe/Evaluate/Remember, policy-engine-gated, full audit trail),
**real similarity-based nearest-successful-run selection** (`similarity.py`
— replaces "just pick the most recent success" with a documented, weighted
match on dataset/model/GPU/git-ancestry/config-distance/framework/
recency), **evidence IDs and provenance labeling** (every diagnosis links to
the specific `EV-N` evidence categories behind it; every UI section is
tagged rule-based / calculated / Ollama-generated / verified), and a
**redesigned web UI** (`watcher ui` — Overview, Runs (global + per-project),
Failures, Campaigns, Memory, Settings; human-readable run names with
rename/tagging/resolution status; a real cross-campaign Resolution Memory
view built from accumulated `recovery_trials` data).

## What's next (Stage 3 of the broader product doc — proactive detection)

The run detail page's "Timeline" currently shows only discrete
already-recorded events (start, warnings, failure) — not live monitoring.
The next real capability gap is **proactive, in-flight detection**: NaN/
exploding-gradient watch, GPU-utilization-below-threshold, DataLoader
stall, throughput regression, validation degradation, no-progress timeout,
imminent-OOM-from-memory-trend, all firing *while a run is still active*,
not after it crashes. This requires `Run`/`SystemSampler` to support live
callbacks/thresholds rather than only post-hoc aggregation, plus a real
event-stream table (not the lightweight timeline that exists now) for the
UI to render against. This is the biggest remaining architectural gap
between what's built and the full product doc.

## Other work still ahead

1. **Dogfood on real workloads** (real GPU, real OOM, real Ollama — see
   prior roadmap notes; still not done as of this writing).
2. **Server mode:** Postgres/S3 `Storage` implementation, auth, multi-tenant
   groundwork — `webapp.py` and the SDK both already go through `Storage`
   as the only seam, so this should mostly be a new implementation of that
   interface rather than a rewrite. Worth confirming that assumption early.
3. **Isolated executor hardening** (git worktree per trial, real
   GPU/wall-time/disk limits) — still only relevant once campaigns do more
   than config-only patches, but cheaper to build now than retrofit later.
4. **Packaging & engineering quality:** PyPI package, versioned Docker
   images, dependency/container scanning, `SECURITY.md`,
   `CONTRIBUTING.md`, license decision, automated release notes.
5. **Private alpha, then public launch.** Same caveats as before on
   autopilot/recovery-agent labeling, plus: the public demo mode needs
   auth from server mode first, and needs every displayed value to come
   from a real instrumented run, not `_simulated: true` demo data.

## Explicitly postponed past v0.1

General workflow orchestration, hosted multi-tenant SaaS, model
deployment/serving, drift monitoring, dataset annotation, full
hyperparameter optimization (comes after semantic diagnosis narrows the
search space, not before), a full model registry, Kubernetes, team
permissions beyond simple project tokens, automated model promotion, and
custom Prometheus/Grafana replacements.

## Before this repo (or any Buffy-derived docs) become public

- Rotate every credential that appeared in any earlier internal Buffy guide
  (Postgres, Grafana, MinIO, n8n, pgAdmin) — treat all of them as
  compromised the moment they existed in a doc you shared or plan to share.
- Scrub secrets from full Git history, not just the latest commit
  (`git filter-repo` or BFG Repo-Cleaner — not just a new commit on top).
- Ship only `.env.example`, never a real `.env`.
- Bind internal services to the Docker network rather than every host
  interface; don't expose the underlying "Buffy" box itself as the public demo.