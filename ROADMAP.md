
WatcherML roadmap

WatcherML is building a local-first recovery and reliability layer for machine-learning workloads. The first proof vertical is deterministic CUDA out-of-memory recovery: preserve the failure, constrain the search, run isolated trials, and verify that a recovery repeats.

This roadmap describes intended direction rather than fixed release dates. Scope may change as the project is tested on real workloads.

Product principles

Evidence remains the durable source of truth.

Deterministic policy and verification remain available without an LLM.

Compute and permissions are declared before a campaign runs.

AI may propose or explain; it cannot issue a recovery verdict.

Every trial remains attributable, inspectable, and exportable.

Integrations extend WatcherML instead of making its core depend on a hosted platform.

v0.1 — Deterministic CUDA OOM recovery

Status: initial public release

The first release establishes one complete, provable recovery path.

Local Python SDK and CLI.

SQLite metadata and local artifact storage under .watcherml/.

Versioned, checksummed failure capsules.

Deterministic CUDA OOM classification.

Explicit training-entrypoint contract.

Capability discovery separated from permission to modify controls.

Typed, allowlisted OOM interventions.

Recovery contracts covering identity, permissions, compute budgets, progress, metrics, regression limits, and optional VRAM limits.

Fresh subprocesses for probes, full trials, and confirmation runs.

Deterministic ranking eligibility and provisional candidate ordering.

Independent verification before a campaign can claim recovery.

Portable capsule and campaign exports.

CLI-first operation for scripts, Jupyter, and Google Colab.

Optional local web UI.

The objective of v0.1 is not broad automation. It is to prove that WatcherML can distinguish a run that happened to finish once from a recovery that satisfied a predeclared contract repeatedly.

v0.2 — Ecosystem integrations and easier adoption

This release will make WatcherML fit into existing ML workflows without replacing their tracking systems.

MLflow sink

Mirror WatcherML run parameters, metrics, tags, and final status into MLflow.

Upload capsules, contracts, campaign reports, and verification artifacts.

Preserve WatcherML run and campaign IDs for two-way traceability.

Support an explicitly configured local or remote tracking URI.

Weights & Biases sink

Mirror configuration, metric history, summaries, and run status into W&B.

Publish recovery trials and confirmations as related runs.

Store exported capsules and campaign reports as artifacts.

Link WatcherML campaign IDs to the corresponding W&B runs.

Integration guarantees

MLflow and W&B support will be optional extras.

No account, API key, or network connection will be required by the core SDK.

WatcherML’s local record will remain authoritative for recovery evidence.

A sink failure will be recorded but will not corrupt a run or change a verification result.

Initial sinks will be one-way exports; bidirectional control is out of scope.

Developer experience

Higher-level PyTorch helpers for common training-loop events.

Clearer Jupyter and Colab setup and progress output.

Campaign summaries designed for terminals without a web UI.

Example projects using real models, datasets, and CUDA OOM failures.

Compatibility testing across supported Python, PyTorch, CUDA, and operating-system versions.

v0.3 — Deeper recovery policies

This release will expand the set of useful OOM interventions while retaining narrow authority and deterministic validation.

Phase-aware policies for training, evaluation, checkpointing, and generation OOMs.

Additional typed controls such as evaluation batch size, sequence length, precision, gradient checkpointing, accumulation, data-loader behavior, and selected optimizer settings.

Framework-aware capability adapters that map user configuration into typed controls.

Better probes for late-stage and phase-specific failures.

Baseline-relative throughput, quality, progress, and memory comparisons.

Smarter intervention ladders based on captured evidence and previous campaign outcomes.

Explicit handling of effective batch-size preservation and permitted semantic changes.

Improved NVIDIA telemetry and allocator evidence when available.

Initial support for bounded multi-GPU and distributed-training recovery.

Adding a discoverable control will not automatically authorize WatcherML to change it. Every new intervention type must define validation, permissions, serialization, execution, and verification behavior.

v0.4 — Optional Autopilot with any-llm

Autopilot will return as an optional planning layer after the deterministic recovery protocol is stable.

Provider-independent AI

Integrate any-llm so users can select supported local or hosted providers.

Keep provider credentials and model configuration outside capsules and exported evidence.

Make the AI dependency an optional package extra.

Continue to support deterministic campaigns when no model is configured.

Autopilot roles

Summarize captured evidence for engineers.

Generate ranked, evidence-linked hypotheses.

Propose typed interventions and explain expected trade-offs.

Learn proposal ordering from verified local campaign history.

Produce human-readable campaign and comparison summaries.

Authority modes

Suggest only: produce proposals without running compute.

Approve each: require confirmation before every trial.

Bounded unattended: execute only preauthorized proposal types within a sealed recovery contract.

All model output will be treated as untrusted input. Proposed interventions must pass the same deterministic schema, capability, permission, identity, and budget checks as non-AI proposals. Autopilot will not be able to rewrite arbitrary training code, install packages, change datasets, expand its own budget, or mark a recovery as verified. Only the deterministic verifier can issue that verdict.

v0.5 — Team workflows and extensibility

Stable plugin interfaces for sinks, capability adapters, intervention policies, storage backends, and reporters.

Shared artifact storage while preserving local-first operation.

Read-only campaign bundles for reviews, incident reports, and pull requests.

CI commands for validating capsules, contracts, campaign reports, and checksums.

Policy presets maintained by teams and repositories.

Cross-project resolution memory built only from verified outcomes.

Optional notifications and webhooks for campaign milestones.

Audit-friendly retention, redaction, and export controls.

v1.0 — Stable recovery protocol

WatcherML 1.0 will focus on stability rather than adding another large feature category.

Stable public SDK and CLI contracts.

Documented compatibility and migration guarantees for stored metadata and artifacts.

Versioned plugin APIs.

Reproducible end-to-end examples and benchmark workloads.

Failure-injection and adversarial tests for the fail-closed trust path.

Clear support policy for Python, PyTorch, CUDA, notebooks, and operating systems.

Production-quality documentation for local, Colab, team, MLflow, and W&B workflows.

Longer-term candidates

These areas require further evidence before being assigned to a release:

Deterministic recovery protocols for NaNs, data-loader stalls, distributed failures, and checkpoint corruption.

Scheduler integrations for Slurm, Kubernetes, and managed training services.

OpenTelemetry and additional experiment-tracker sinks.

Organization-level policy distribution and signed recovery artifacts.

Privacy-preserving sharing of verified recovery patterns.

Explicit non-goals

WatcherML is not intended to:

Replace MLflow, Weights & Biases, or general experiment tracking.

Become a generic hyperparameter-optimization framework.

Let an LLM decide whether a recovery succeeded.

Silently modify arbitrary code, dependencies, datasets, or infrastructure.

Claim that a verified campaign proves every possible root cause.

Claim that a recovery will work forever on every future workload.

How to contribute

The most valuable contributions are currently:

Real CUDA OOM capsules with sensitive data removed.

Reproducible workloads that exercise recovery policies.

Tests for malformed, incomplete, duplicated, or contradictory evidence.

Framework capability adapters with explicit permission semantics.

MLflow and W&B sink implementations that fail independently of core recording.

Documentation and examples for scripts, Jupyter, and Google Colab.

Before proposing a large feature, open an issue describing the failure mode, the evidence WatcherML can capture, the authority it would require, and how success could be verified deterministically.