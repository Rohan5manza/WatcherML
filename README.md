WatcherML

Deterministic CUDA OOM forensics and verified recovery campaigns for ML training.

WatcherML is a local-first Python SDK and CLI that records ML runs, freezes a
structured evidence capsule when training fails with a CUDA out-of-memory
error, proposes bounded interventions, executes candidates in fresh supervised
subprocesses, and calls a recovery verified only after independent confirmation
runs satisfy constraints declared before compute begins.

It is a recovery layer, not another hosted experiment-tracking platform. Use it
on its own with local SQLite storage, or keep using the rest of your ML stack
alongside it.

Manual retries can make a run pass. WatcherML records what failed, what was
allowed to change, what was actually tested, and whether the recovery held up
under confirmation.

Status

WatcherML 0.1.0 is an alpha release with one deliberately narrow, provable
vertical: CUDA OOM capture and verified recovery.

No LLM, API key, hosted account, or internet connection is required.

No Docker container is required. Trials use fresh Python subprocesses on the
current machine and environment.

Recovery compute is launched through the SDK or CLI. The optional web UI is
a local inspection surface.

A successful probe or full trial is not automatically a verified recovery.

Version 1 does not modify source code, datasets, or dependencies.

Install

pip install watcherml

Optional surfaces:

pip install "watcherml[notebook]"  # Jupyter/IPython support
pip install "watcherml[ui]"        # local FastAPI web UI

For development from a clone:

python -m pip install -e ".[dev,notebook,ui]"

WatcherML requires Python 3.10 or newer.

1. Record a training run

import watcherml as watcher

config = {
    "model_name": "resnet50",
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": False,
}

with watcher.init(project="tomato-disease", config=config) as run:
    run.set_dataset("./data/tomato")

    for step in range(100):
        loss = train_step(config)
        run.log_metric("training_loss", loss, step=step)

    validation_loss = evaluate(config)
    run.log_metric("validation_loss", validation_loss)

WatcherML stores run metadata, metrics, artifacts, environment information,
Git state when available, hardware/resource context, and dataset fingerprints
under ./.watcherml/ by default.

You can inspect the result without launching a server:

watcher runs --project tomato-disease
watcher inspect RUN_ID

2. Capture deterministic OOM evidence

When the recorded block raises a CUDA OOM, WatcherML persists a versioned
failure capsule instead of reducing the error to a traceback string.

WatcherML failure capsule v1.0: tomato-disease-a56b75

Exception: RuntimeError: CUDA out of memory
Diagnosis: cuda_out_of_memory (deterministic rule)
Capture completeness: 9/10
Last recorded training state:
  batch_size: 32
  gradient_accumulation_steps: 1
  last_logged_step: 41

The capsule ties a deterministic classification to evidence such as:

the original configuration;

the last recorded training state;

exception type, message, and traceback;

environment and Git information;

GPU and resource state when available;

code, dataset, and run identity information captured by the recorder.

Inspect or export it:

watcher failures --project tomato-disease
watcher inspect RUN_ID --format markdown --output failure.md
watcher export RUN_ID --out failure-capsule.zip

The exported capsule is checksummed and portable. Its purpose is evidence and
reproduction, not an AI-generated explanation.

3. Expose a serializable training entrypoint

Recovery trials must run in fresh processes. WatcherML therefore accepts an
importable module:function entrypoint rather than an in-memory closure.

Create train.py:

def train(config, max_steps=None):
    """Run bounded probe work or the complete configured workload."""
    model, optimizer, train_loader = build_training_objects(config)

    configured_steps = config.get("training_steps", 1_000)
    steps_to_run = max_steps if max_steps is not None else configured_steps

    for step, batch in enumerate(train_loader):
        if step >= steps_to_run:
            break
        train_one_step(model, optimizer, batch, config)

    validation_loss = evaluate(model, config)
    return {
        "validation_loss": validation_loss,
        "steps_completed": steps_to_run,
    }

The entrypoint contract is important:

config is the complete candidate configuration.

max_steps=N means a probe must stop after at most N units of declared
progress.

Omitting max_steps means run the complete configured workload.

The returned mapping must include every guarded metric and an integral
progress metric such as steps_completed.

The callable must be importable from the declared project root.

WatcherML validates the entrypoint before starting campaign compute. It does
not silently fall back to an unbounded in-process function.

4. Declare recovery constraints before compute

import watcherml as watcher

verification = watcher.VerificationRequirements(
    minimum_progress_steps=1_000,
    metric_guards=(
        watcher.MetricGuard(
            name="validation_loss",
            direction="minimize",
            baseline_value=0.42,
            max_regression=0.03,
        ),
    ),
    confirmation_runs=2,
)

budget = watcher.RecoveryBudget(
    max_trials=7,
    max_probe_trials=3,
    max_full_trials=2,
    probe_steps=30,
    trial_timeout_seconds=3_600,
    campaign_timeout_seconds=14_400,
)

result = watcher.recover_from_oom(
    "tomato-disease-a56b75",
    "train:train",
    verification,
    budget=budget,
    project_root=".",
)

if result.verified:
    print("Verified candidate:", result.verified_candidate_id)
    print("Confirmation runs:", result.verified_run_ids)
else:
    print("No verified recovery:", result.campaign.stopped_reason)

The contract seals:

the source OOM run and its original configuration;

the serializable training entrypoint;

probe, full-trial, timeout, and GPU-time budgets;

metric regression boundaries;

minimum required progress;

confirmation-run count;

optional workload-identity and peak-VRAM requirements;

the strongest intervention class the campaign may execute.

The sum of reserved probe, full, and confirmation runs cannot exceed the total
trial budget.

Review the plan before spending compute

For approval-sensitive workflows, separate zero-compute planning from campaign
execution:

preparation = watcher.prepare_oom_recovery(
    "tomato-disease-a56b75",
    "train:train",
    verification,
    budget=budget,
    project_root=".",
)

print(preparation.to_json())

authorizations = {}
for proposal_id in preparation.approval_required_proposal_ids:
    authorization = preparation.authorize(
        proposal_id,
        approved_by="rohan",
        reason="Reviewed configuration and semantic impact.",
    )
    authorizations[proposal_id] = authorization

result = watcher.run_prepared_recovery(
    preparation,
    authorizations=authorizations,
    project_root=".",
)

Automatic low-risk interventions require no approval. Broader interventions
must be explicitly permitted by the campaign contract and authorized for the
specific visible proposal. --yes confirms execution in the CLI; it never
grants proposal authorization.

What happens during a recovery campaign

Validate source evidence. Confirm that the referenced run contains a
valid deterministic CUDA OOM capsule.

Discover capabilities. Determine which typed configuration changes the
entrypoint and workload can represent.

Plan bounded interventions. Generate deterministic proposals backed by
explicit OOM policy rules.

Enforce scope. Reject unknown keys, invalid values, semantic changes, or
higher-risk proposals that exceed the sealed contract and authorization.

Run probes. Launch short trials in fresh supervised subprocesses and
eliminate candidates that still OOM, time out, violate protocol, or fail to
make the declared progress.

Run full trials. Execute surviving candidates against the complete
entrypoint workload.

Rank feasible candidates. Reject constraint violations first, then rank
only candidates that completed and satisfied declared requirements.

Verify independently. Rerun the selected candidate for the required
confirmation count and check progress, metrics, workload identity, resource
limits, process evidence, and artifact integrity.

Persist the audit trail. Store proposals, rejected changes, subprocess
evidence, trial lineage, ranking, verification checks, and the immutable
campaign artifact.

Only step 8 can produce a verified recovery verdict.

Probe, full trial, and confirmation run

Phase

Purpose

Work performed

Can prove recovery?

Probe

Reject obviously bad candidates cheaply

Entrypoint called with max_steps=probe_steps

No

Full trial

Evaluate a surviving candidate

Complete configured workload

No

Confirmation

Independently verify the selected candidate

Complete workload repeated as declared

Yes, collectively

Every phase consumes campaign budget. WatcherML does not make GPU computation
free; it makes the computation bounded, inspectable, comparable, and harder to
misrepresent.

CLI-first workflow

watcher init
watcher doctor
watcher runs --project tomato-disease
watcher inspect RUN_ID
watcher failures --unresolved
watcher compare RUN_A RUN_B
watcher export RUN_ID --out failure-capsule.zip

Prepare a sealed plan without launching trials:

watcher prepare-recovery RUN_ID \
  --entrypoint train:train \
  --project-root . \
  --metric validation_loss:minimize:0.42:0.03 \
  --minimum-progress-steps 1000 \
  --confirmation-runs 2 \
  --max-probe-trials 3 \
  --max-full-trials 2 \
  --probe-steps 30 \
  --out recovery-plan.json

Review and execute it:

watcher recover --plan recovery-plan.json
watcher recoveries --project tomato-disease
watcher recovery CAMPAIGN_ID

Interactive terminals receive progress steps, spinners, tables, sparklines,
colors, and explicit review prompts. Redirected output and CI receive stable
plain text. Most inspection commands support --format json; --no-color and
--quiet are available as top-level CLI flags.

This makes the CLI suitable for SSH sessions and hosted notebooks such as
Google Colab:

!watcher --no-color runs --format json
!watcher --no-color inspect RUN_ID --format json

Optional local web UI

pip install "watcherml[ui]"
watcher ui

The UI runs locally at http://127.0.0.1:7331 and presents runs, metrics,
failure evidence, recovery proposals, isolated trials, confirmation checks,
campaign artifacts, and resolution memory.

Recovery truth remains verifier-owned: the UI cannot manually turn a failed run
into a verified recovery. Launch recovery compute through the SDK or CLI.

Local storage

By default WatcherML creates:

.watcherml/
├── watcher.db       # SQLite metadata, metrics, capsules, and campaign lineage
└── artifacts/       # local content-addressed artifacts

Set WATCHERML_DIR or use the CLI's top-level --data-dir option to choose a
different location. Existing databases are migrated in place without deleting
recorded rows.

Why use this instead of manually changing the batch size?

For a cheap one-off experiment with an obvious fix, manually reducing the batch
size may be faster. WatcherML is useful when the recovery must be reviewable and
repeatable across people, machines, notebooks, CI jobs, or expensive training
runs.

It provides evidence that manual retries usually do not:

the exact failed workload and environment;

an immutable declaration of allowed changes and compute limits;

rejected as well as executed proposals;

fresh-process trial evidence and timeout supervision;

metric-regression and progress constraints chosen before seeing results;

repeated confirmation rather than one lucky successful run;

a machine-readable artifact that another engineer can audit.

WatcherML is not valuable because it knows that smaller batches use less
memory. It is valuable because it turns an informal debugging sequence into a
bounded recovery protocol with a defensible verdict.

WatcherML and experiment trackers

WatcherML 0.1.0 is not trying to replace the dashboards, collaboration,
artifact registries, or hosted services provided by MLflow and Weights &
Biases. Its v1 responsibility is narrower: failure evidence and verified local
recovery.

Native tracker integrations are planned after the core recovery protocol is
stable. Until then, WatcherML can record beside an existing tracker, but the
README does not claim first-class MLflow or W&B synchronization.

Scope and limitations of 0.1.0

Implemented:

local run, metric, artifact, environment, Git, dataset, and resource capture;

deterministic versioned failure capsules;

structured run comparison and portable capsule export;

serializable training-entrypoint validation;

fresh subprocess trial execution with parent-side timeouts;

capability discovery and typed bounded interventions;

deterministic CUDA OOM policy planning;

immutable recovery contracts and explicit authorization boundaries;

probe, full, and confirmation campaign orchestration;

constraint-first candidate ranking;

independent confirmation verification;

SQLite persistence, CLI inspection, and optional local web UI.

Not implemented or deliberately excluded from v1:

automatic source-code, dataset, or dependency modification;

Docker/container isolation for each trial;

distributed multi-node campaign scheduling;

hosted team accounts or a remote control plane;

recovery classes other than CUDA OOM;

LLM diagnosis, autopilot, or open-ended autonomous iteration;

first-class MLflow or Weights & Biases synchronization.

Development

git clone https://github.com/Rohan5manza/WatcherML.git
cd WatcherML
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,ui]"
python -m pytest -q

Build and validate release artifacts:

python -m build
python -m twine check dist/*

Security and bug reports

Please report bugs through the
GitHub issue tracker.
Do not include secrets, proprietary datasets, or sensitive environment values
in public issue attachments.

License

WatcherML is released under the MIT License.