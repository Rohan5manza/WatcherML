"""WatcherML's local-first command-line interface.

The CLI is intentionally useful without the optional web UI.  Human terminals
receive compact colors, tables, progress indicators, metric sparklines, and
review prompts.  Redirected output, CI, and Colab receive stable plain text or
strict JSON without cursor control or ANSI escape sequences.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .capsule import format_capsule_report
from .diff import compare_runs, format_diff_report
from .entrypoint import TrainingEntrypoint
from .export import export_capsule
from .recovery import (
    RecoveryPreparation,
    RecoveryResult,
    prepare_oom_recovery,
    preparation_digest,
    run_prepared_recovery,
)
from .recovery_contract import (
    InterventionPermissions,
    MetricGuard,
    RecoveryBudget,
    VerificationRequirements,
    WorkloadIdentity,
)
from .storage import Storage


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_NOT_VERIFIED = 4
EXIT_DECLINED = 5
EXIT_ERROR = 1

DEFAULT_PORT = 7331
SPARKLINE_GLYPHS = "▁▂▃▄▅▆▇█"


class CLIError(RuntimeError):
    def __init__(self, message: str, exit_code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class Console:
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "cyan": "\033[36m",
    }

    def __init__(self, *, color: Optional[bool] = None, quiet: bool = False) -> None:
        interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        disabled = os.getenv("NO_COLOR") is not None or os.getenv("TERM") == "dumb"
        self.color = interactive and not disabled if color is None else bool(color)
        self.interactive = interactive
        self.quiet = quiet

    def style(self, text: object, *styles: str) -> str:
        value = str(text)
        if not self.color:
            return value
        prefix = "".join(self.COLORS[name] for name in styles)
        return prefix + value + self.COLORS["reset"]

    def print(self, text: object = "", *, file=None) -> None:
        if not self.quiet:
            print(text, file=file or sys.stdout)

    def heading(self, text: str) -> None:
        self.print(self.style(text, "bold", "cyan"))

    def step(self, number: int, total: int, title: str, detail: str = "") -> None:
        prefix = self.style("[{}/{}]".format(number, total), "bold", "blue")
        suffix = "  " + self.style(detail, "dim") if detail else ""
        self.print("{} {}{}".format(prefix, self.style(title, "bold"), suffix))

    def success(self, text: str) -> None:
        self.print("{} {}".format(self.style("✓", "green", "bold"), text))

    def warning(self, text: str) -> None:
        self.print("{} {}".format(self.style("!", "yellow", "bold"), text))

    def error(self, text: str) -> None:
        print("{} {}".format(self.style("error:", "red", "bold"), text), file=sys.stderr)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        if not sys.stdin.isatty():
            raise CLIError(
                "interactive confirmation is unavailable; review the plan and rerun with --yes",
                EXIT_DECLINED,
            )
        marker = "Y/n" if default else "y/N"
        answer = input("{} [{}] ".format(prompt, marker)).strip().lower()
        if not answer:
            return default
        return answer in {"y", "yes"}

    @contextlib.contextmanager
    def spinner(self, label: str):
        if not self.interactive or self.quiet:
            self.print(label + "…")
            yield
            return
        stopped = threading.Event()

        def animate() -> None:
            glyphs = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            index = 0
            while not stopped.wait(0.08):
                frame = self.style(glyphs[index % len(glyphs)], "cyan")
                sys.stdout.write("\r{} {}".format(frame, label))
                sys.stdout.flush()
                index += 1

        thread = threading.Thread(target=animate, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=0.3)
            sys.stdout.write("\r" + " " * (len(label) + 4) + "\r")
            sys.stdout.flush()


def _console(args) -> Console:
    color = False if getattr(args, "no_color", False) else None
    quiet = getattr(args, "quiet", False) or getattr(args, "format", None) == "json"
    return Console(color=color, quiet=quiet)


def _storage(args) -> Storage:
    root = getattr(args, "data_dir", None) or os.getenv("WATCHERML_DIR")
    return Storage(root) if root else Storage()


@contextlib.contextmanager
def _opened_storage(args):
    storage = _storage(args)
    try:
        yield storage
    finally:
        storage.close()


def _row_get(row, name: str, default=None):
    if row is None:
        return default
    try:
        return row[name]
    except (IndexError, KeyError):
        return default


def _safe_json(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _json_print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _atomic_write(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    materialized = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized:
        for index, value in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(value))
    terminal = shutil.get_terminal_size((120, 24)).columns
    if sum(widths) + max(0, len(widths) - 1) * 3 > terminal:
        excess = sum(widths) + max(0, len(widths) - 1) * 3 - terminal
        widest = sorted(range(len(widths)), key=widths.__getitem__, reverse=True)
        for index in widest:
            reducible = max(0, widths[index] - 12)
            reduction = min(reducible, excess)
            widths[index] -= reduction
            excess -= reduction
            if excess <= 0:
                break

    def shorten(value: str, width: int) -> str:
        if len(value) <= width:
            return value
        return value[: max(1, width - 1)] + "…"

    lines = [
        "   ".join(shorten(header, widths[i]).ljust(widths[i]) for i, header in enumerate(headers)),
        "   ".join("─" * width for width in widths),
    ]
    lines.extend(
        "   ".join(shorten(value, widths[i]).ljust(widths[i]) for i, value in enumerate(row))
        for row in materialized
    )
    return "\n".join(lines)


def _duration(value) -> str:
    if value is None:
        return "—"
    seconds = float(value)
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return "{}m {:02d}s".format(minutes, seconds)
    hours, minutes = divmod(minutes, 60)
    return "{}h {:02d}m".format(hours, minutes)


def _sparkline(values: Sequence[float], width: int = 36) -> str:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return ""
    if len(finite) > width:
        indexes = [round(index * (len(finite) - 1) / (width - 1)) for index in range(width)]
        finite = [finite[index] for index in indexes]
    low, high = min(finite), max(finite)
    if high == low:
        return SPARKLINE_GLYPHS[len(SPARKLINE_GLYPHS) // 2] * len(finite)
    return "".join(
        SPARKLINE_GLYPHS[
            min(
                len(SPARKLINE_GLYPHS) - 1,
                int((value - low) / (high - low) * len(SPARKLINE_GLYPHS)),
            )
        ]
        for value in finite
    )


def _progress_bar(current: int, maximum: int, width: int = 24) -> str:
    maximum = max(1, maximum)
    fraction = min(1.0, max(0.0, current / maximum))
    filled = round(width * fraction)
    return "[{}{}] {}/{}".format("█" * filled, "░" * (width - filled), current, maximum)


def _status_label(value: str, console: Console) -> str:
    if value in {"success", "verified"}:
        return console.style(value, "green", "bold")
    if value in {"failed", "not_recovered", "integration_error", "stopped"}:
        return console.style(value, "red" if value in {"failed", "integration_error"} else "yellow", "bold")
    return console.style(value, "cyan")


def cmd_init(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        if args.format == "json":
            _json_print(
                {
                    "data_directory": storage.root,
                    "database": storage.db_path,
                    "storage_schema_version": storage.schema_version,
                }
            )
            return EXIT_OK
        console.success("WatcherML initialized")
        console.print("  data       {}".format(storage.root))
        console.print("  database   {}".format(storage.db_path))
        console.print("\nNext: wrap training with `watcherml.init(...)`, or inspect `watcher --help`.")
    return EXIT_OK


def cmd_doctor(args) -> int:
    console = _console(args)
    checks = []
    with _opened_storage(args) as storage:
        checks.append(("Storage", True, storage.root))
        checks.append(("SQLite schema", True, storage.schema_version))
        try:
            from . import _trial_worker  # noqa: F401

            checks.append(("Trial worker", True, "importable"))
        except Exception as exc:
            checks.append(("Trial worker", False, str(exc)))
        try:
            import torch

            available = bool(torch.cuda.is_available())
            detail = (
                "{} device(s): {}".format(
                    torch.cuda.device_count(), torch.cuda.get_device_name(0)
                )
                if available
                else "PyTorch installed; CUDA unavailable"
            )
            checks.append(("CUDA", available, detail))
        except ImportError:
            checks.append(("CUDA", False, "PyTorch is not installed"))

    if args.format == "json":
        _json_print(
            {
                "checks": [
                    {"name": name, "ok": ok, "detail": detail}
                    for name, ok, detail in checks
                ],
                "ready_for_cpu_features": all(ok for name, ok, _ in checks if name != "CUDA"),
                "ready_for_real_cuda_recovery": all(ok for _, ok, _ in checks),
            }
        )
    else:
        console.heading("WatcherML doctor")
        for name, ok, detail in checks:
            marker = console.style("✓", "green") if ok else console.style("!", "yellow")
            console.print("{} {:<16} {}".format(marker, name, detail))
        if not checks[-1][1]:
            console.warning("Recorder and CPU smoke tests can work, but real CUDA recovery cannot be validated here.")
    return EXIT_OK if all(ok for name, ok, _ in checks if name != "CUDA") else EXIT_ERROR


def cmd_runs(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        rows = storage.list_runs(project=args.project)
        if args.limit is not None:
            rows = rows[: args.limit]
        if args.status:
            rows = [row for row in rows if row["exit_status"] == args.status]
        if args.format == "json":
            _json_print([_run_payload(storage, row) for row in rows])
            return EXIT_OK
        if not rows:
            console.print("No runs matched.")
            return EXIT_OK
        console.heading("Recorded runs")
        console.print(
            _table(
                ("RUN ID", "PROJECT", "STATUS", "DURATION", "RESOLVED"),
                (
                    (
                        row["run_id"],
                        row["project"] or "—",
                        row["exit_status"] or "—",
                        _duration(row["duration_seconds"]),
                        "yes" if bool(_row_get(row, "resolved", 0)) else "no",
                    )
                    for row in rows
                ),
            )
        )
    return EXIT_OK


def _run_payload(storage: Storage, row) -> dict:
    payload = dict(row)
    for field in (
        "config_json",
        "git_json",
        "env_json",
        "gpu_json",
        "resource_json",
        "warnings_json",
        "tags_json",
    ):
        if field in payload:
            payload[field.removesuffix("_json")] = _safe_json(payload.pop(field), {})
    payload["final_metrics"] = storage.final_metrics(row["run_id"])
    return payload


def cmd_inspect(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        row = storage.get_run(args.run_id)
        if row is None:
            raise CLIError("run {!r} was not found".format(args.run_id), EXIT_NOT_FOUND)
        capsule = storage.get_failure_capsule(args.run_id)
        if capsule is not None:
            if args.format == "json":
                _write_or_print(json.dumps(capsule, indent=2, ensure_ascii=False), args.output)
            elif args.format == "markdown":
                _write_or_print(_capsule_markdown(capsule), args.output)
            else:
                _write_or_print(format_capsule_report(capsule), args.output)
            return EXIT_OK

        payload = _run_payload(storage, row)
        if args.format == "json":
            _write_or_print(json.dumps(payload, indent=2, ensure_ascii=False), args.output)
            return EXIT_OK
        if args.format == "markdown":
            _write_or_print(_run_markdown(payload), args.output)
            return EXIT_OK
        console.heading("Run {}".format(row["run_id"]))
        console.print("project     {}".format(row["project"]))
        console.print("status      {}".format(_status_label(row["exit_status"] or "unknown", console)))
        console.print("duration    {}".format(_duration(row["duration_seconds"])))
        console.print("resolved    {}".format("yes" if bool(_row_get(row, "resolved", 0)) else "no"))
        console.print("\nConfig")
        console.print(json.dumps(payload.get("config") or {}, indent=2, ensure_ascii=False))
        _render_metric_history(console, storage.get_metrics(args.run_id))
        completeness = _row_get(row, "capture_completeness")
        if completeness is None:
            completeness = _row_get(row, "reproduction_score")
        if completeness is not None:
            console.print("\nCapture completeness: {}/10".format(int(completeness)))
    return EXIT_OK


def _render_metric_history(console: Console, rows) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row["value"])
    console.print("\nMetrics")
    if not grouped:
        console.print("  none")
        return
    rendered = []
    for name, values in grouped.items():
        rendered.append(
            (
                name,
                "{:.6g}".format(values[-1]),
                str(len(values)),
                _sparkline(values),
            )
        )
    console.print(_table(("METRIC", "FINAL", "POINTS", "HISTORY"), rendered))


def _capsule_markdown(capsule: dict) -> str:
    failure = capsule.get("failure") or {}
    return "\n".join(
        [
            "# WatcherML failure capsule: `{}`".format(capsule.get("run_id")),
            "",
            "- **Failure class:** `{}`".format(
                capsule.get("failure_class") or failure.get("class") or "unknown"
            ),
            "- **Exception:** `{}`".format(
                failure.get("exception_type") or capsule.get("exception_type") or "unknown"
            ),
            "- **Message:** {}".format(failure.get("message") or capsule.get("message") or ""),
            "",
            "## Evidence",
            "",
            "```json",
            json.dumps(capsule.get("evidence") or {}, indent=2, ensure_ascii=False),
            "```",
        ]
    )


def _run_markdown(payload: dict) -> str:
    metrics = payload.get("final_metrics") or {}
    lines = [
        "# WatcherML run: `{}`".format(payload["run_id"]),
        "",
        "- **Project:** {}".format(payload.get("project")),
        "- **Status:** {}".format(payload.get("exit_status")),
        "- **Duration:** {}".format(_duration(payload.get("duration_seconds"))),
        "",
        "## Config",
        "",
        "```json",
        json.dumps(payload.get("config") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Final metrics",
        "",
    ]
    lines.extend("- `{}`: {}".format(name, value) for name, value in metrics.items())
    if not metrics:
        lines.append("No metrics recorded.")
    return "\n".join(lines)


def _write_or_print(text: str, output: Optional[str]) -> None:
    if output:
        _atomic_write(Path(output), text.rstrip() + "\n")
        print(output)
    else:
        print(text)


def cmd_failures(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        rows = storage.list_failures(project=args.project)
        if args.unresolved:
            rows = [row for row in rows if not bool(_row_get(storage.get_run(row["run_id"]), "resolved", 0))]
        payload = []
        for row in rows:
            diagnosis = _safe_json(row["diagnosis_json"], {})
            payload.append(
                {
                    "run_id": row["run_id"],
                    "project": _row_get(row, "project"),
                    "failure_class": _row_get(row, "failure_class") or diagnosis.get("rule", "unknown"),
                    "message": row["message"],
                    "resolved": bool(_row_get(storage.get_run(row["run_id"]), "resolved", 0)),
                }
            )
        if args.format == "json":
            _json_print(payload)
        elif not payload:
            console.print("No failures matched.")
        else:
            console.heading("Failure capsules")
            console.print(
                _table(
                    ("RUN ID", "CLASS", "RESOLVED", "MESSAGE"),
                    (
                        (item["run_id"], item["failure_class"], "yes" if item["resolved"] else "no", item["message"])
                        for item in payload
                    ),
                )
            )
    return EXIT_OK


def cmd_compare(args) -> int:
    with _opened_storage(args) as storage:
        diff = compare_runs(storage, args.run_a, args.run_b)
    if args.format == "json":
        _json_print(diff)
    else:
        print(format_diff_report(diff))
    return EXIT_OK


def cmd_export(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        path = export_capsule(storage, args.run_id, args.out)
    if args.format == "json":
        _json_print({"run_id": args.run_id, "path": path})
    else:
        console.success("Portable capsule written to {}".format(path))
    return EXIT_OK


def _metric_guard(value: str) -> MetricGuard:
    parts = value.split(":")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "metric must be NAME:DIRECTION:BASELINE:MAX_REGRESSION[:TARGET]"
        )
    name, direction = parts[:2]
    try:
        baseline = float(parts[2])
        regression = float(parts[3])
        target = float(parts[4]) if len(parts) == 5 else None
        return MetricGuard(name, direction, baseline, regression, target)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _verification(args) -> VerificationRequirements:
    if not args.metric:
        raise CLIError("at least one --metric guard is required", EXIT_USAGE)
    peak = int(args.max_peak_vram_gib * 1024**3) if args.max_peak_vram_gib else None
    identity = WorkloadIdentity(
        dataset_fingerprint=args.dataset_fingerprint,
        environment_fingerprint=args.environment_fingerprint,
        git_commit=args.git_commit,
        model_identifier=args.model_identifier,
    )
    return VerificationRequirements(
        minimum_progress_steps=args.minimum_progress_steps,
        metric_guards=tuple(args.metric),
        confirmation_runs=args.confirmation_runs,
        max_peak_vram_bytes=peak,
        workload_identity=identity,
    )


def _budget(args, confirmations: int) -> RecoveryBudget:
    max_trials = args.max_trials
    reserved = args.max_probe_trials + args.max_full_trials + confirmations
    if max_trials is None:
        max_trials = reserved
    return RecoveryBudget(
        max_trials=max_trials,
        max_probe_trials=args.max_probe_trials,
        max_full_trials=args.max_full_trials,
        probe_steps=args.probe_steps,
        trial_timeout_seconds=args.trial_timeout,
        campaign_timeout_seconds=args.campaign_timeout,
        max_gpu_seconds=args.max_gpu_seconds,
    )


def _permissions(args) -> InterventionPermissions:
    allow_approval = bool(
        args.allow_approval_required or args.allow_semantic_changes or args.allow_high_risk
    )
    return InterventionPermissions(
        allow_approval_required=allow_approval,
        allow_semantic_changes=args.allow_semantic_changes,
        allow_high_risk=args.allow_high_risk,
    )


def _declarations(path: Optional[str]):
    if not path:
        return None
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CLIError("could not read capability declarations: {}".format(exc)) from exc
    if not isinstance(payload, list):
        raise CLIError("capability declarations file must contain a JSON array")
    return payload


def _prepare_from_args(args, storage: Storage) -> RecoveryPreparation:
    verification = _verification(args)
    entrypoint = TrainingEntrypoint(args.entrypoint, args.working_directory)
    return prepare_oom_recovery(
        args.source_run_id,
        entrypoint,
        verification,
        budget=_budget(args, verification.confirmation_runs),
        permissions=_permissions(args),
        storage=storage,
        project_root=args.project_root,
        capability_declarations=_declarations(args.capabilities),
        max_proposals=args.max_proposals,
        include_approval_required=not args.automatic_only,
    )


def _render_preparation(console: Console, preparation: RecoveryPreparation) -> None:
    contract = preparation.contract
    console.heading("OOM recovery plan")
    console.print("source run    {}".format(contract.source_run_id))
    console.print("entrypoint    {}".format(contract.entrypoint.target))
    console.print("plan digest   {}".format(preparation_digest(preparation)[:16]))
    console.print(
        "budget        {} trials: {} probe + {} full + {} confirmation".format(
            contract.budget.max_trials,
            contract.budget.max_probe_trials,
            contract.budget.max_full_trials,
            contract.verification.confirmation_runs,
        )
    )
    console.print("\nDiscovered controls")
    capabilities = []
    for item in preparation.capability_manifest.capabilities:
        capabilities.append(
            (
                item.capability_id,
                item.location,
                item.target,
                str(item.current_value),
                item.risk,
            )
        )
    console.print(
        _table(("CAPABILITY", "LOCATION", "TARGET", "CURRENT", "RISK"), capabilities)
        if capabilities
        else "  none"
    )
    console.print("\nBounded proposals")
    automatic = set(preparation.automatic_proposal_ids)
    proposals = []
    for index, proposal in enumerate(preparation.policy_plan.proposals, 1):
        changes = ", ".join(
            "{}→{}".format(change.capability_id, change.proposed_value)
            for change in proposal.changes
        )
        proposals.append(
            (
                index,
                proposal.proposal_id,
                "automatic" if proposal.proposal_id in automatic else "approval",
                proposal.policy_rule,
                changes,
            )
        )
    console.print(
        _table(("#", "PROPOSAL ID", "AUTH", "RULE", "CHANGES"), proposals)
        if proposals
        else "  none justified by the captured evidence"
    )
    console.print(
        "\nPreparation performs no trial compute. Approval-required proposals remain inert until explicitly authorized."
    )


def cmd_prepare_recovery(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        with console.spinner("Inspecting capsule and building deterministic plan"):
            preparation = _prepare_from_args(args, storage)
    encoded = preparation.to_json()
    if args.out:
        _atomic_write(Path(args.out), encoded + "\n")
    if args.format == "json":
        if args.out:
            _json_print(
                {
                    "path": str(Path(args.out).expanduser().resolve()),
                    "preparation_digest": preparation_digest(preparation),
                    "preparation": preparation.to_dict(),
                }
            )
        else:
            print(json.dumps(preparation.to_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _render_preparation(console, preparation)
        if args.out:
            console.success("Sealed plan written to {}".format(Path(args.out).expanduser().resolve()))
        else:
            console.print("\nSave it with `--out recovery-plan.json`, or run the same arguments with `watcher recover`.")
    return EXIT_OK


def _load_preparation(path: str) -> RecoveryPreparation:
    try:
        return RecoveryPreparation.from_json(
            Path(path).expanduser().read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise CLIError("could not read recovery plan: {}".format(exc)) from exc


def _authorize_proposals(args, console: Console, preparation: RecoveryPreparation):
    requested = set(args.authorize or [])
    approval_ids = set(preparation.approval_required_proposal_ids)
    unknown = sorted(requested - approval_ids)
    if unknown:
        raise CLIError(
            "--authorize contains proposal ids that are not approval-required: {}".format(
                ", ".join(unknown)
            ),
            EXIT_USAGE,
        )
    if requested and not preparation.contract.permissions.allow_approval_required:
        raise CLIError("the sealed contract does not permit approval-required interventions")
    if requested and not args.approved_by:
        raise CLIError("--approved-by is required with --authorize", EXIT_USAGE)
    authorizations = {}
    for proposal_id in requested:
        authorizations[proposal_id] = preparation.authorize(
            proposal_id,
            approved_by=args.approved_by,
            reason=args.approval_reason,
        )

    if (
        console.interactive
        and args.format != "json"
        and not args.no_approval_prompts
        and preparation.contract.permissions.allow_approval_required
    ):
        for proposal_id in preparation.approval_required_proposal_ids:
            if proposal_id in authorizations:
                continue
            proposal = preparation.proposal(proposal_id)
            console.print("\nApproval review: {}".format(proposal.policy_rule))
            console.print("  {}".format(proposal.rationale))
            console.print("  proposal id: {}".format(proposal_id))
            if console.confirm("Authorize this proposal for this campaign?", default=False):
                approved_by = args.approved_by or input("Approver identity: ").strip()
                if not approved_by:
                    raise CLIError("approver identity cannot be empty")
                reason = args.approval_reason
                if reason == "Reviewed and authorized through the WatcherML CLI.":
                    entered = input("Approval reason [reviewed in CLI]: ").strip()
                    if entered:
                        reason = entered
                authorizations[proposal_id] = preparation.authorize(
                    proposal_id,
                    approved_by=approved_by,
                    reason=reason,
                )
    return authorizations


def cmd_recover(args) -> int:
    console = _console(args)
    if args.format == "json" and not args.yes:
        raise CLIError("--format json requires --yes for non-interactive execution", EXIT_USAGE)
    with _opened_storage(args) as storage:
        console.step(1, 5, "Observe", "load and seal deterministic OOM evidence")
        if args.plan:
            preparation = _load_preparation(args.plan)
        else:
            if not args.source_run_id or not args.entrypoint:
                raise CLIError(
                    "provide SOURCE_RUN_ID and --entrypoint, or use --plan FILE",
                    EXIT_USAGE,
                )
            with console.spinner("Building zero-compute recovery preparation"):
                preparation = _prepare_from_args(args, storage)

        if args.save_plan:
            _atomic_write(Path(args.save_plan), preparation.to_json() + "\n")

        if args.format != "json":
            console.step(2, 5, "Plan", "review deterministic proposals and budget")
            _render_preparation(console, preparation)
            console.step(3, 5, "Authorize", "broad changes require proposal-specific approval")
        authorizations = _authorize_proposals(args, console, preparation)
        if args.format != "json":
            console.print(
                "  {} explicitly authorized; {} will remain skipped".format(
                    len(authorizations),
                    len(preparation.approval_required_proposal_ids) - len(authorizations),
                )
            )
        if not args.yes:
            if not console.confirm(
                "Start an isolated campaign with a hard budget of {} trials?".format(
                    preparation.contract.budget.max_trials
                ),
                default=False,
            ):
                console.warning("Campaign not started; preparation remains zero-compute.")
                return EXIT_DECLINED

        if args.format != "json":
            console.step(4, 5, "Execute", "fresh subprocesses; hard trial and time budgets")
        with console.spinner("Running isolated probes, full trials, and confirmations"):
            result = run_prepared_recovery(
                preparation,
                authorizations=authorizations,
                storage=storage,
                project_root=args.project_root,
                trials_root=args.trials_root,
                progress_metric=args.progress_metric,
                python_executable=args.python_executable,
                termination_grace_seconds=args.termination_grace,
                campaign_id=args.campaign_id,
                print_summary=False,
            )
    if args.format == "json":
        _json_print(result.to_dict())
    else:
        console.step(5, 5, "Verify", "only independent confirmations can promote recovery")
        _render_recovery_result(console, result)
    return EXIT_OK if result.verified else EXIT_NOT_VERIFIED


def _render_recovery_result(console: Console, result: RecoveryResult) -> None:
    campaign = result.campaign
    console.print("")
    if result.verified:
        console.success("Verified recovery: {}".format(result.verified_candidate_id))
    else:
        console.warning("No verified recovery was produced.")
    console.print("campaign     {}".format(campaign.campaign_id))
    console.print("status       {}".format(_status_label(campaign.status, console)))
    console.print("reason       {}".format(campaign.stopped_reason))
    console.print(
        "trial budget {}".format(
            _progress_bar(
                campaign.usage.attempted_trials,
                result.preparation.contract.budget.max_trials,
            )
        )
    )
    console.print(
        "phases       probe {} · full {} · confirmation {}".format(
            campaign.usage.probe_trials,
            campaign.usage.full_trials,
            campaign.usage.confirmation_trials,
        )
    )
    console.print("\nTrial evidence")
    console.print(
        _table(
            ("PHASE", "RUN ID", "CANDIDATE", "STATUS", "PROGRESS", "DURATION"),
            (
                (
                    trial.phase,
                    trial.run_id,
                    trial.candidate_id,
                    trial.status,
                    trial.progress_steps if trial.progress_steps is not None else "—",
                    _duration(trial.duration_seconds),
                )
                for trial in campaign.trials
            ),
        )
        if campaign.trials
        else "  no trial evidence"
    )
    if result.verified:
        console.print("\nIndependent confirmations")
        for run_id in result.verified_run_ids:
            console.print("  ✓ {}".format(run_id))
    if result.skipped_proposals:
        console.print("\nSkipped proposals")
        for item in result.skipped_proposals:
            console.print("  {}  {} ({})".format(item.proposal_id, item.policy_rule, item.code))
    console.print("\nInspect later: watcher recovery {}".format(campaign.campaign_id))


def cmd_recoveries(args) -> int:
    console = _console(args)
    verified = True if args.verified else None
    with _opened_storage(args) as storage:
        rows = storage.list_recovery_campaigns(
            project=args.project,
            status=args.status,
            verified=verified,
        )
        if args.limit is not None:
            rows = rows[: args.limit]
        payload = [_campaign_row_payload(row) for row in rows]
    if args.format == "json":
        _json_print(payload)
    elif not payload:
        console.print("No recovery campaigns matched.")
        console.print("Start with `watcher prepare-recovery --help`.")
    else:
        console.heading("OOM recovery campaigns")
        console.print(
            _table(
                ("CAMPAIGN", "PROJECT", "STATUS", "VERIFIED", "REASON", "STARTED"),
                (
                    (
                        item["campaign_id"],
                        item["project"],
                        item["status"],
                        "yes" if item["verified"] else "no",
                        item["stopped_reason"] or "running",
                        _timestamp(item["started_at"]),
                    )
                    for item in payload
                ),
            )
        )
    return EXIT_OK


def _campaign_row_payload(row) -> dict:
    return {
        "campaign_id": row["campaign_id"],
        "project": row["project"],
        "source_run_id": row["source_run_id"],
        "status": _row_get(row, "status", "running") or "running",
        "verified": bool(_row_get(row, "verified", 0)),
        "verified_candidate_id": _row_get(row, "verified_candidate_id"),
        "stopped_reason": row["stopped_reason"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "artifact_path": _row_get(row, "artifact_path"),
    }


def _timestamp(value) -> str:
    if value is None:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))


def cmd_recovery(args) -> int:
    console = _console(args)
    with _opened_storage(args) as storage:
        row = storage.get_recovery_campaign(args.campaign_id)
        if row is None:
            raise CLIError(
                "campaign {!r} was not found".format(args.campaign_id), EXIT_NOT_FOUND
            )
        report = storage.get_recovery_campaign_report(args.campaign_id)
        trials = storage.list_recovery_trials(args.campaign_id)
        proposals = storage.list_recovery_proposals(args.campaign_id)
        verifications = storage.list_recovery_verifications(args.campaign_id)
    if args.format == "json":
        _json_print(
            {
                "campaign": _campaign_row_payload(row),
                "report": report,
                "trials": [_stored_trial_payload(item) for item in trials],
                "proposals": [_stored_proposal_payload(item) for item in proposals],
                "verifications": [_stored_verification_payload(item) for item in verifications],
            }
        )
        return EXIT_OK

    console.heading("Recovery campaign {}".format(row["campaign_id"]))
    console.print("project      {}".format(row["project"]))
    console.print("source run   {}".format(row["source_run_id"]))
    console.print("status       {}".format(_status_label(_row_get(row, "status", "running"), console)))
    console.print("verified     {}".format("yes" if bool(_row_get(row, "verified", 0)) else "no"))
    console.print("reason       {}".format(row["stopped_reason"] or "still running"))
    if _row_get(row, "verified_candidate_id"):
        console.print("candidate    {}".format(row["verified_candidate_id"]))
    usage = _safe_json(_row_get(row, "usage_json"), {})
    if usage:
        contract = _safe_json(row["contract_json"], {})
        maximum = (contract.get("budget") or {}).get("max_trials", usage.get("attempted_trials", 0))
        console.print(
            "trial budget  {}".format(
                _progress_bar(usage.get("attempted_trials", 0), maximum)
            )
        )
    console.print("\nTrials")
    console.print(
        _table(
            ("PHASE", "RUN ID", "CANDIDATE", "STATUS", "PROGRESS", "VERIFIED"),
            (
                (
                    item["phase"],
                    item["run_id"],
                    _row_get(item, "candidate_id") or _row_get(item, "proposal_id") or "—",
                    _row_get(item, "status") or item["outcome"] or "—",
                    _row_get(item, "progress_steps") if _row_get(item, "progress_steps") is not None else "—",
                    "yes" if bool(item["verified"]) else "no",
                )
                for item in trials
            ),
        )
        if trials
        else "  none"
    )
    if proposals:
        console.print("\nProposals")
        console.print(
            _table(
                ("PROPOSAL ID", "AUTH", "STATE", "RULE", "SKIP"),
                (
                    (
                        item["proposal_id"], item["authorization_mode"], item["state"],
                        item["policy_rule"], item["skip_code"] or "—"
                    )
                    for item in proposals
                ),
            )
        )
    if verifications:
        console.print("\nVerification")
        for item in verifications:
            run_ids = _safe_json(item["confirmation_run_ids_json"], [])
            marker = "✓" if bool(item["verified"]) else "×"
            console.print("  {} {}  {}".format(marker, item["candidate_id"], ", ".join(run_ids)))
    if _row_get(row, "artifact_path"):
        console.print("\nArtifact: {}".format(row["artifact_path"]))
    return EXIT_OK


def _stored_trial_payload(row) -> dict:
    payload = dict(row)
    for field in (
        "hypothesis_json", "patch_json", "metrics_json", "workload_identity_json",
        "environment_patch_json", "trial_json",
    ):
        payload[field.removesuffix("_json")] = _safe_json(payload.pop(field, None), {})
    payload["verified"] = bool(payload["verified"])
    return payload


def _stored_proposal_payload(row) -> dict:
    payload = dict(row)
    payload["proposal"] = _safe_json(payload.pop("proposal_json"), {})
    return payload


def _stored_verification_payload(row) -> dict:
    payload = dict(row)
    payload["verified"] = bool(payload["verified"])
    payload["confirmation_run_ids"] = _safe_json(
        payload.pop("confirmation_run_ids_json"), []
    )
    payload["report"] = _safe_json(payload.pop("report_json"), {})
    return payload


def cmd_ui(args) -> int:
    console = _console(args)
    try:
        import uvicorn
    except ImportError as exc:
        raise CLIError(
            "the web UI extra is not installed; run `pip install 'watcherml[ui]'`"
        ) from exc
    from .webapp import create_app

    storage = _storage(args)
    app = create_app(storage)
    url = "http://{}:{}".format(args.host, args.port)
    console.success("WatcherML UI running at {} (Ctrl+C to stop)".format(url))
    if not args.no_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        storage.close()
    return EXIT_OK


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default=None,
        help="WatcherML data directory (default: ./.watcherml or WATCHERML_DIR)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--quiet", action="store_true", help="Suppress human progress output")


def _add_format(parser: argparse.ArgumentParser, choices=("text", "json")) -> None:
    parser.add_argument("--format", choices=choices, default=choices[0])


def _add_recovery_contract_options(parser: argparse.ArgumentParser, *, source: bool) -> None:
    if source:
        parser.add_argument("source_run_id", help="Recorded deterministic CUDA OOM run")
        parser.add_argument(
            "--entrypoint",
            required=True,
            help="Importable training callable, for example train:main",
        )
    parser.add_argument("--project-root", default=".", help="Root used to import the entrypoint")
    parser.add_argument("--working-directory", default=".", help="Entrypoint working directory relative to project root")
    parser.add_argument(
        "--metric",
        action="append",
        type=_metric_guard,
        metavar="NAME:DIRECTION:BASELINE:MAX_REGRESSION[:TARGET]",
        help="Verification metric guard; repeat for multiple metrics",
    )
    parser.add_argument("--minimum-progress-steps", type=int, default=100)
    parser.add_argument("--confirmation-runs", type=int, default=2)
    parser.add_argument("--max-peak-vram-gib", type=float, default=None)
    parser.add_argument("--dataset-fingerprint", default=None)
    parser.add_argument("--environment-fingerprint", default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--model-identifier", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-probe-trials", type=int, default=3)
    parser.add_argument("--max-full-trials", type=int, default=2)
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--trial-timeout", type=float, default=3600.0)
    parser.add_argument("--campaign-timeout", type=float, default=14400.0)
    parser.add_argument("--max-gpu-seconds", type=float, default=None)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--automatic-only", action="store_true", help="Exclude all approval-required proposals from the plan")
    parser.add_argument("--allow-approval-required", action="store_true", help="Permit proposal-specific broader interventions")
    parser.add_argument("--allow-semantic-changes", action="store_true", help="Permit explicitly authorized semantic changes")
    parser.add_argument("--allow-high-risk", action="store_true", help="Permit explicitly authorized high-risk changes")
    parser.add_argument("--capabilities", default=None, help="JSON array of custom capability declarations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watcher",
        description="Deterministic CUDA OOM forensics and verified recovery",
    )
    _add_common_options(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("init", help="Initialize local WatcherML storage")
    _add_format(command)
    command.set_defaults(func=cmd_init)

    command = sub.add_parser("doctor", help="Check local recorder and CUDA readiness")
    _add_format(command)
    command.set_defaults(func=cmd_doctor)

    command = sub.add_parser("runs", help="List recorded runs")
    command.add_argument("--project")
    command.add_argument("--status", choices=("running", "success", "failed"))
    command.add_argument("--limit", type=int, default=50)
    _add_format(command)
    command.set_defaults(func=cmd_runs)

    command = sub.add_parser("inspect", help="Inspect one run or failure capsule")
    command.add_argument("run_id")
    command.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    command.add_argument("--output", default=None)
    command.set_defaults(func=cmd_inspect)

    command = sub.add_parser("failures", help="List deterministic failure capsules")
    command.add_argument("--project")
    command.add_argument("--unresolved", action="store_true")
    _add_format(command)
    command.set_defaults(func=cmd_failures)

    command = sub.add_parser("compare", help="Compare two recorded runs")
    command.add_argument("run_a")
    command.add_argument("run_b")
    _add_format(command)
    command.set_defaults(func=cmd_compare)

    command = sub.add_parser("export", help="Export a portable failure capsule")
    command.add_argument("run_id")
    command.add_argument("--out")
    _add_format(command)
    command.set_defaults(func=cmd_export)

    command = sub.add_parser(
        "prepare-recovery",
        help="Build and review a sealed zero-compute OOM recovery plan",
    )
    _add_recovery_contract_options(command, source=True)
    command.add_argument("--out", default=None, help="Write the sealed plan JSON")
    _add_format(command)
    command.set_defaults(func=cmd_prepare_recovery)

    command = sub.add_parser(
        "recover",
        help="Run an interactive bounded recovery campaign",
    )
    command.add_argument("source_run_id", nargs="?", help="OOM run; omit when using --plan")
    command.add_argument("--plan", help="Previously sealed preparation JSON")
    command.add_argument("--entrypoint", help="Importable callable when not using --plan")
    # These are used only when a plan is built in this command.
    _add_recovery_contract_options(command, source=False)
    command.add_argument("--save-plan", default=None)
    command.add_argument("--trials-root", default=None)
    command.add_argument("--progress-metric", default="steps_completed")
    command.add_argument("--python-executable", default=None)
    command.add_argument("--termination-grace", type=float, default=5.0)
    command.add_argument("--campaign-id", default=None)
    command.add_argument("--authorize", action="append", default=[], metavar="PROPOSAL_ID")
    command.add_argument("--approved-by", default=None)
    command.add_argument(
        "--approval-reason",
        default="Reviewed and authorized through the WatcherML CLI.",
    )
    command.add_argument("--no-approval-prompts", action="store_true")
    command.add_argument("--yes", "-y", action="store_true", help="Confirm campaign execution; never approves proposals")
    _add_format(command)
    command.set_defaults(func=cmd_recover)

    command = sub.add_parser("recoveries", help="List persisted OOM campaigns")
    command.add_argument("--project")
    command.add_argument("--status")
    command.add_argument("--verified", action="store_true")
    command.add_argument("--limit", type=int, default=50)
    _add_format(command)
    command.set_defaults(func=cmd_recoveries)

    command = sub.add_parser("recovery", help="Inspect one recovery audit trail")
    command.add_argument("campaign_id")
    _add_format(command)
    command.set_defaults(func=cmd_recovery)

    command = sub.add_parser("ui", help="Launch the optional local web UI")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=DEFAULT_PORT)
    command.add_argument("--no-browser", action="store_true")
    command.set_defaults(func=cmd_ui)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or EXIT_OK)
    except CLIError as exc:
        _console(args).error(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        _console(args).error("interrupted")
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        _console(args).error(str(exc))
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())