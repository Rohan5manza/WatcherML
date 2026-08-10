from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from . import collectors
from .capsule import (
    build_evidence_index,
    compare_to_last_success,
    find_similar_failures,
    format_capsule_report,
)
from .diff import compare_runs, format_diff_report
from .export import export_capsule
from .storage import Storage


def _safe_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _row_get(row, key: str, default=None):
    try:
        return row[key] if key in row.keys() else default
    except (AttributeError, KeyError, TypeError):
        return default


def _close_storage(storage: Storage) -> None:
    close = getattr(storage, "close", None)
    if callable(close):
        close()
        return
    connection = getattr(storage, "_conn", None)
    if connection is not None:
        connection.close()


def _write_or_print(text: str, output: str | None = None) -> None:
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        print(f"Written to {path}")
        return
    print(text)


def _is_colab() -> bool:
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_GPU")
        or "google.colab" in sys.modules
    )


def _load_capsule(storage: Storage, run_id: str) -> dict | None:
    """Load a complete v1 capsule when available, with legacy DB fallback."""
    get_capsule = getattr(storage, "get_capsule", None)
    if callable(get_capsule):
        stored = get_capsule(run_id)
        if stored is not None:
            if isinstance(stored, dict):
                return stored
            capsule_json = _row_get(stored, "capsule_json")
            if capsule_json:
                return _safe_json(capsule_json, None)

    failure = storage.get_failure(run_id)
    if failure is None:
        return None

    row = storage.get_run(run_id)
    diagnosis = _safe_json(failure["diagnosis_json"], {})
    evidence = _safe_json(failure["evidence_json"], {})
    project = _row_get(row, "project") if row is not None else None
    rule = diagnosis.get("rule", "unclassified")

    similar = []
    comparison = None
    if project:
        similar = find_similar_failures(storage, project, run_id, rule)
        comparison = compare_to_last_success(storage, project, run_id)

    return {
        "schema_version": 0,
        "run_id": run_id,
        "project": project,
        "exception_type": failure["exception_type"],
        "message": failure["message"],
        "traceback": failure["traceback"],
        "diagnosis": diagnosis,
        "evidence": evidence,
        "evidence_index": build_evidence_index(evidence),
        "similar_previous_failures": similar,
        "comparison_to_last_success": comparison,
        "provenance": {
            "diagnosis": "rule-based",
            "comparison": "calculated",
        },
    }


def _format_capsule_markdown(capsule: dict) -> str:
    diagnosis = capsule.get("diagnosis") or capsule.get("classification") or {}
    evidence = capsule.get("evidence") or {}
    comparison = (
        capsule.get("comparison_to_last_success")
        or capsule.get("nearest_successful_run")
    )

    lines = [
        f"# WatcherML failure capsule: `{capsule.get('run_id', 'unknown')}`",
        "",
        f"- **Project:** {capsule.get('project') or 'unknown'}",
        f"- **Failure:** `{diagnosis.get('rule', 'unclassified')}`",
        f"- **Exception:** `{capsule.get('exception_type', 'unknown')}`",
        f"- **Message:** {capsule.get('message') or 'No message captured'}",
        f"- **Schema version:** {capsule.get('schema_version', 0)}",
        "",
        "## Deterministic diagnosis",
        "",
        diagnosis.get("summary") or "No deterministic summary available.",
    ]

    if diagnosis.get("likely_cause"):
        lines.extend(["", f"**Likely cause:** {diagnosis['likely_cause']}"])

    actions = diagnosis.get("suggested_actions") or []
    if actions:
        lines.extend(["", "## Suggested interventions", ""])
        lines.extend(f"- {action}" for action in actions)

    if comparison:
        lines.extend([
            "",
            "## Nearest successful run",
            "",
            f"- **Run:** `{comparison.get('run_id')}`",
            f"- **Similarity:** {comparison.get('similarity_score', 'unknown')}",
        ])

    recent_metrics = evidence.get("recent_metrics") or []
    if recent_metrics:
        lines.extend(["", "## Recent metrics", ""])
        for metric in recent_metrics:
            lines.append(
                f"- `{metric.get('name')}` = {metric.get('value')} "
                f"(step {metric.get('step')})"
            )

    evidence_index = capsule.get("evidence_index") or []
    if evidence_index:
        lines.extend(["", "## Evidence index", ""])
        lines.extend(
            f"- **{item.get('id')}** — {item.get('label', item.get('category'))}"
            for item in evidence_index
        )

    lines.extend([
        "",
        "## Traceback",
        "",
        "```text",
        capsule.get("traceback") or "No traceback captured.",
        "```",
    ])
    return "\n".join(lines)


def _render_capsule(capsule: dict, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(capsule, indent=2, ensure_ascii=False)
    if output_format == "markdown":
        return _format_capsule_markdown(capsule)
    return format_capsule_report(capsule)


def cmd_init(args) -> None:
    storage = Storage()
    try:
        print(f"Initialized WatcherML in {storage.root}")
        print(
            "Next: `import watcherml as watcher` and wrap your training code "
            "with `watcher.init(...)`."
        )
    finally:
        _close_storage(storage)


def cmd_runs(args) -> None:
    storage = Storage()
    try:
        rows = storage.list_runs(project=args.project)
        if args.format == "json":
            payload = []
            for row in rows:
                item = dict(row)
                item["final_metrics"] = storage.final_metrics(row["run_id"])
                payload.append(item)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        if not rows:
            print("No runs recorded yet.")
            return

        print(f"{'RUN_ID':<26} {'PROJECT':<20} {'STATUS':<12} {'DURATION':<10}")
        for row in rows:
            duration = row["duration_seconds"]
            duration_text = f"{duration:.1f}s" if duration is not None else "-"
            print(
                f"{row['run_id']:<26} {row['project']:<20} "
                f"{row['exit_status'] or '-':<12} {duration_text:<10}"
            )
    finally:
        _close_storage(storage)


def cmd_inspect(args) -> None:
    storage = Storage()
    try:
        row = storage.get_run(args.run_id)
        if row is None:
            raise ValueError(f"Run '{args.run_id}' not found.")

        capsule = _load_capsule(storage, args.run_id)
        if capsule is not None:
            _write_or_print(_render_capsule(capsule, args.format), args.output)
            return

        config = _safe_json(row["config_json"], {})
        final_metrics = storage.final_metrics(args.run_id)
        if args.format == "json":
            payload = dict(row)
            payload["config"] = config
            payload["final_metrics"] = final_metrics
            _write_or_print(
                json.dumps(payload, indent=2, ensure_ascii=False), args.output
            )
            return

        if args.format == "markdown":
            lines = [
                f"# WatcherML run: `{row['run_id']}`",
                "",
                f"- **Project:** {row['project']}",
                f"- **Status:** {row['exit_status']}",
                f"- **Duration:** {row['duration_seconds'] or 0:.1f}s",
                "",
                "## Config",
                "",
                "```json",
                json.dumps(config, indent=2, ensure_ascii=False),
                "```",
                "",
                "## Final metrics",
                "",
            ]
            lines.extend(f"- `{name}`: {value}" for name, value in final_metrics.items())
            _write_or_print("\n".join(lines), args.output)
            return

        print(f"Run: {row['run_id']}  ({row['project']})")
        print(f"Status: {row['exit_status']}")
        if row["duration_seconds"] is not None:
            print(f"Duration: {row['duration_seconds']:.1f}s")
        print(f"Config: {json.dumps(config, ensure_ascii=False)}")
        print("Final metrics:")
        if final_metrics:
            for name, value in final_metrics.items():
                print(f"  {name}: {value}")
        else:
            print("  none")

        completeness = _row_get(row, "capture_completeness")
        if completeness is None:
            completeness = _row_get(row, "reproduction_score")
        if completeness is not None:
            print(f"Capture completeness: {int(completeness)}/10")
    finally:
        _close_storage(storage)


def cmd_capsule(args) -> None:
    storage = Storage()
    try:
        capsule = _load_capsule(storage, args.run_id)
        if capsule is None:
            raise ValueError(f"No failure capsule found for run '{args.run_id}'.")
        _write_or_print(_render_capsule(capsule, args.format), args.output)
    finally:
        _close_storage(storage)


def cmd_failures(args) -> None:
    storage = Storage()
    try:
        rows = storage.list_failures(project=args.project)
        if args.format == "json":
            payload = []
            for row in rows:
                item = dict(row)
                item["diagnosis"] = _safe_json(row["diagnosis_json"], {})
                payload.append(item)
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        if not rows:
            print("No failures recorded.")
            return

        print(f"{'RUN_ID':<26} {'FAILURE':<30} MESSAGE")
        for row in rows:
            diagnosis = _safe_json(row["diagnosis_json"], {})
            rule = diagnosis.get("rule", "unclassified")
            message = (row["message"] or "")[:70]
            print(f"{row['run_id']:<26} {rule:<30} {message}")
    finally:
        _close_storage(storage)


def cmd_compare(args) -> None:
    storage = Storage()
    try:
        diff = compare_runs(storage, args.run_a, args.run_b)
        if args.format == "json":
            _write_or_print(
                json.dumps(diff, indent=2, ensure_ascii=False), args.output
            )
        else:
            _write_or_print(format_diff_report(diff), args.output)
    finally:
        _close_storage(storage)


def cmd_export(args) -> None:
    storage = Storage()
    try:
        out_path = export_capsule(storage, args.run_id, args.out)
        print(f"Evidence capsule written to {out_path}")
    finally:
        _close_storage(storage)


def cmd_recover_list(args) -> None:
    storage = Storage()
    try:
        rows = storage.list_recovery_campaigns(project=args.project)
        if args.format == "json":
            print(json.dumps([dict(row) for row in rows], indent=2, ensure_ascii=False))
            return

        if not rows:
            print("No OOM recovery campaigns recorded yet.")
            return

        print(f"{'CAMPAIGN_ID':<22} {'PROJECT':<20} {'STATUS':<38} {'SELECTED_RUN':<26}")
        for row in rows:
            status = (row["stopped_reason"] or "running")[:36]
            print(
                f"{row['campaign_id']:<22} {row['project']:<20} "
                f"{status:<38} {row['best_run_id'] or '-':<26}"
            )
    finally:
        _close_storage(storage)


def cmd_recover_show(args) -> None:
    storage = Storage()
    try:
        campaign = storage.get_recovery_campaign(args.campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign '{args.campaign_id}' not found.")

        trials = storage.list_recovery_trials(args.campaign_id)
        if args.format == "json":
            payload = dict(campaign)
            payload["trials"] = [dict(trial) for trial in trials]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        print(f"Campaign: {campaign['campaign_id']}  ({campaign['project']})")
        print(f"Source OOM run: {campaign['source_run_id']}")
        print(f"Status: {campaign['stopped_reason'] or 'running'}")
        if campaign["best_run_id"]:
            print(f"Selected full trial: {campaign['best_run_id']}")
        print("\nTrials:")
        if not trials:
            print("  none")
            return
        for trial in trials:
            patch = _safe_json(trial["patch_json"], {})
            print(
                f"  [{trial['phase']:<8}] {trial['run_id']:<26} "
                f"outcome={trial['outcome']:<22} patch={patch}"
            )
    finally:
        _close_storage(storage)


def cmd_doctor(args) -> None:
    storage = Storage()
    try:
        connection = getattr(storage, "_conn", None)
        journal_mode = "unknown"
        if connection is not None:
            try:
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            except Exception:
                journal_mode = "unavailable"

        try:
            package_version = importlib.metadata.version("watcherml")
        except importlib.metadata.PackageNotFoundError:
            package_version = "development checkout"

        torch_report: dict[str, Any]
        try:
            import torch

            torch_report = {
                "available": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_build": torch.version.cuda,
                "device": (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
            }
        except ImportError:
            torch_report = {
                "available": False,
                "version": None,
                "cuda_available": False,
                "cuda_build": None,
                "device": None,
            }

        try:
            isolated_runner = importlib.util.find_spec(
                "watcherml.recovery.executor"
            ) is not None
        except (ImportError, ModuleNotFoundError):
            isolated_runner = False

        gpu = collectors.collect_gpu_info()
        report = {
            "watcherml_version": package_version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "environment": "google_colab" if _is_colab() else "local",
            "storage_root": str(storage.root),
            "database_path": str(storage.db_path),
            "database_exists": Path(storage.db_path).exists(),
            "sqlite_journal_mode": journal_mode,
            "pytorch": torch_report,
            "gpu": gpu,
            "isolated_trial_runner": isolated_runner,
            "ui_dependencies": importlib.util.find_spec("uvicorn") is not None,
        }

        if args.format == "json":
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return

        rows = [
            ("WatcherML", report["watcherml_version"]),
            ("Python", report["python_version"]),
            ("Environment", report["environment"]),
            ("Storage", report["storage_root"]),
            ("Database", "healthy" if report["database_exists"] else "missing"),
            ("SQLite journal", report["sqlite_journal_mode"]),
            ("PyTorch", torch_report["version"] or "not installed"),
            ("CUDA available", "yes" if torch_report["cuda_available"] else "no"),
            ("CUDA build", torch_report["cuda_build"] or "unknown"),
            ("GPU", torch_report["device"] or "not detected"),
            ("Isolated trials", "ready" if isolated_runner else "not implemented"),
            ("Local UI", "available" if report["ui_dependencies"] else "not installed"),
        ]
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"{label:<{width}}  {value}")

        if _is_colab():
            print(
                "\nColab mode: use CLI/JSON/Markdown reports. To preserve run history, "
                "set WATCHERML_HOME to a mounted Google Drive directory."
            )
    finally:
        _close_storage(storage)


def cmd_ui(args) -> None:
    if _is_colab():
        raise ValueError(
            "The local web UI is not supported in Google Colab. Use `watcherml runs`, "
            "`watcherml inspect`, and `watcherml capsule --format markdown` instead."
        )

    try:
        import uvicorn
    except ImportError:
        raise ValueError(
            "The web UI needs optional dependencies. Install them with "
            "`pip install watcherml[ui]`."
        ) from None

    from .webapp import create_app

    app = create_app(Storage())
    url = f"http://{args.host}:{args.port}"
    print(f"WatcherML UI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _add_format_argument(parser, choices=("table", "json")) -> None:
    parser.add_argument(
        "--format",
        choices=choices,
        default=choices[0],
        help=f"Output format (default: {choices[0]})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watcherml",
        description="Your open-source reliability and recovery layer for ML training runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("init", help="Initialize WatcherML in the current directory")
    command.set_defaults(func=cmd_init)

    command = sub.add_parser("runs", help="List recorded runs")
    command.add_argument("--project")
    _add_format_argument(command)
    command.set_defaults(func=cmd_runs)

    command = sub.add_parser("inspect", help="Inspect a run or its failure capsule")
    command.add_argument("run_id")
    command.add_argument("--output", help="Write the report to a file")
    _add_format_argument(command, ("table", "json", "markdown"))
    command.set_defaults(func=cmd_inspect)

    command = sub.add_parser(
        "capsule", help="Show or export a deterministic failure capsule"
    )
    command.add_argument("run_id")
    command.add_argument("--output", help="Write the capsule report to a file")
    _add_format_argument(command, ("table", "json", "markdown"))
    command.set_defaults(func=cmd_capsule)

    command = sub.add_parser("failures", help="List recorded failures")
    command.add_argument("--project")
    _add_format_argument(command)
    command.set_defaults(func=cmd_failures)

    command = sub.add_parser("compare", help="Compare two recorded runs")
    command.add_argument("run_a")
    command.add_argument("run_b")
    command.add_argument("--output", help="Write the comparison to a file")
    _add_format_argument(command)
    command.set_defaults(func=cmd_compare)

    command = sub.add_parser("export", help="Export a portable evidence capsule")
    command.add_argument("run_id")
    command.add_argument("--out", help="Destination ZIP path")
    command.set_defaults(func=cmd_export)

    recover = sub.add_parser("recover", help="Inspect bounded OOM recovery campaigns")
    recover_sub = recover.add_subparsers(dest="recover_command", required=True)

    command = recover_sub.add_parser("list", help="List OOM recovery campaigns")
    command.add_argument("--project")
    _add_format_argument(command)
    command.set_defaults(func=cmd_recover_list)

    command = recover_sub.add_parser("show", help="Show an OOM recovery campaign")
    command.add_argument("campaign_id")
    _add_format_argument(command)
    command.set_defaults(func=cmd_recover_show)

    command = sub.add_parser(
        "doctor", help="Check storage, PyTorch, CUDA, Colab, UI, and trial support"
    )
    _add_format_argument(command)
    command.set_defaults(func=cmd_doctor)

    command = sub.add_parser("ui", help="Launch the optional local web UI")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=7331)
    command.add_argument("--no-browser", action="store_true")
    command.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        print(f"watcherml: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())