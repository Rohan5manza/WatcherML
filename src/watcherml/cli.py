from __future__ import annotations

import argparse
import json
import sys

from . import advisor
from .capsule import format_capsule_report
from .diff import compare_runs, format_diff_report
from .export import export_capsule
from .storage import Storage


def _print_ai_section(title: str, text: str | None, model: str):
    if text is None:
        print(f"\n[{title}: unavailable -- Ollama isn't running, or model "
              f"'{model}' isn't pulled. `ollama pull {model}` and retry. "
              f"Everything above this line is deterministic and didn't need it.]")
        return
    print(f"\n--- {title} (AI-generated, model: {model} -- verify before trusting) ---")
    print(text)


def cmd_init(args):
    Storage()  # creates .watcherml/ in the current directory
    print("Initialized WatcherML in ./.watcherml (SQLite metadata + local artifact store).")
    print("Next: `import watcherml as watcher` and wrap your training code with `watcher.init(...)`.")


def cmd_runs(args):
    storage = Storage()
    rows = storage.list_runs(project=args.project)
    if not rows:
        print("No runs recorded yet.")
        return
    print(f"{'RUN_ID':<20} {'PROJECT':<20} {'STATUS':<10} {'DURATION':<10}")
    for r in rows:
        dur = f"{r['duration_seconds']:.1f}s" if r["duration_seconds"] else "-"
        print(f"{r['run_id']:<20} {r['project']:<20} {r['exit_status'] or '-':<10} {dur:<10}")


def cmd_inspect(args):
    storage = Storage()
    row = storage.get_run(args.run_id)
    if row is None:
        print(f"Run {args.run_id} not found.", file=sys.stderr)
        sys.exit(1)

    failure = storage.get_failure(args.run_id)
    if failure is not None:
        capsule = {
            "run_id": args.run_id,
            "exception_type": failure["exception_type"],
            "message": failure["message"],
            "traceback": failure["traceback"],
            "diagnosis": json.loads(failure["diagnosis_json"]),
            "evidence": json.loads(failure["evidence_json"]),
            "similar_previous_failures": [],
            "comparison_to_last_success": None,
        }
        print(format_capsule_report(capsule))
        if getattr(args, "advise", False):
            _print_ai_section("AI explanation", advisor.explain_failure(capsule, model=args.model), args.model)
        return

    print(f"Run: {row['run_id']}  ({row['project']})")
    print(f"Status: {row['exit_status']}")
    if row["duration_seconds"]:
        print(f"Duration: {row['duration_seconds']:.1f}s")
    print(f"Config: {row['config_json']}")
    print("Final metrics:")
    for name, value in storage.final_metrics(args.run_id).items():
        print(f"  {name}: {value}")
    if row["reproduction_score"] is not None:
        print(f"Reproduction score: {int(row['reproduction_score'])}/10")


def cmd_failures(args):
    storage = Storage()
    rows = storage.list_failures(project=args.project)
    if not rows:
        print("No failures recorded. 🎉")
        return
    for r in rows:
        diag = json.loads(r["diagnosis_json"])
        print(f"{r['run_id']:<20} [{diag['rule']:<28}] {r['message'][:60]}")


def cmd_compare(args):
    storage = Storage()
    diff = compare_runs(storage, args.run_a, args.run_b)
    print(format_diff_report(diff))
    if getattr(args, "advise", False):
        _print_ai_section("Likely explanation", advisor.explain_diff(diff, model=args.model), args.model)


def cmd_advise(args):
    """Standalone advisor command: run it against any past failure or comparison."""
    storage = Storage()
    failure = storage.get_failure(args.run_id)
    if failure is None:
        print(f"No failure recorded for {args.run_id}. `watcher advise` currently "
              f"only explains failures -- use `watcher compare A B --advise` for comparisons.",
              file=sys.stderr)
        sys.exit(1)
    capsule = {
        "run_id": args.run_id,
        "exception_type": failure["exception_type"],
        "message": failure["message"],
        "diagnosis": json.loads(failure["diagnosis_json"]),
        "evidence": json.loads(failure["evidence_json"]),
    }
    if not advisor.is_available(host=args.host):
        print(f"Ollama isn't reachable at {args.host}. Start it (`ollama serve`) "
              f"and make sure the model is pulled (`ollama pull {args.model}`).")
        sys.exit(1)
    text = advisor.explain_failure(capsule, model=args.model, host=args.host)
    _print_ai_section("AI explanation", text, args.model)


def cmd_export(args):
    storage = Storage()
    out_path = export_capsule(storage, args.run_id, args.out)
    print(f"Reproduction capsule written to {out_path}")


def cmd_recoveries(args):
    storage = Storage()
    rows = storage.list_recovery_campaigns(project=args.project)
    if not rows:
        print("No recovery campaigns recorded yet. Launch one from Python: "
              "watcher.recover_from_oom(project=..., failed_run_id=..., train_fn=...)")
        return
    print(f"{'CAMPAIGN_ID':<20} {'PROJECT':<20} {'STOPPED_REASON':<38} {'BEST_RUN':<24}")
    for r in rows:
        reason = (r["stopped_reason"] or "running")[:36]
        print(f"{r['campaign_id']:<20} {r['project']:<20} {reason:<38} {r['best_run_id'] or '-':<24}")


def cmd_recovery(args):
    storage = Storage()
    campaign = storage.get_recovery_campaign(args.campaign_id)
    if campaign is None:
        print(f"Campaign {args.campaign_id} not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Campaign: {campaign['campaign_id']}  ({campaign['project']})")
    print(f"Source run (the OOM failure this campaign recovers from): {campaign['source_run_id']}")
    print(f"Stopped: {campaign['stopped_reason'] or 'still running'}")
    if campaign["best_run_id"]:
        print(f"Best verified run: {campaign['best_run_id']}")
    print("\nTrials:")
    for t in storage.list_recovery_trials(args.campaign_id):
        patch = json.loads(t["patch_json"] or "{}")
        print(f"  [{t['phase']:<5}] {t['run_id']:<24} outcome={t['outcome']:<20} "
              f"score={t['score']}  patch={patch}")


def cmd_ui(args):
    try:
        import uvicorn
    except ImportError:
        print("The web UI needs extra dependencies. Install them with:\n"
              "  pip install watcherml[ui]", file=sys.stderr)
        sys.exit(1)
    from .webapp import create_app

    app = create_app(Storage())
    url = f"http://{args.host}:{args.port}"
    print(f"WatcherML UI running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def build_parser():
    p = argparse.ArgumentParser(prog="watcher", description="WatcherML CLI")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="Initialize WatcherML in the current directory")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("runs", help="List recorded runs")
    sp.add_argument("--project", default=None)
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("inspect", help="Show details for a single run")
    sp.add_argument("run_id")
    sp.add_argument("--advise", action="store_true",
                     help="Add an AI-generated explanation via Ollama (optional; deterministic capsule works without it)")
    sp.add_argument("--model", default=advisor.DEFAULT_MODEL)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("failures", help="List all recorded failures")
    sp.add_argument("--project", default=None)
    sp.set_defaults(func=cmd_failures)

    sp = sub.add_parser("compare", help="Compare two runs")
    sp.add_argument("run_a")
    sp.add_argument("run_b")
    sp.add_argument("--advise", action="store_true",
                     help="Add an AI-generated 'likely explanation' via Ollama")
    sp.add_argument("--model", default=advisor.DEFAULT_MODEL)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("advise", help="Get an AI (Ollama) explanation for a past failure")
    sp.add_argument("run_id")
    sp.add_argument("--model", default=advisor.DEFAULT_MODEL)
    sp.add_argument("--host", default=advisor.DEFAULT_HOST)
    sp.set_defaults(func=cmd_advise)

    sp = sub.add_parser("export", help="Export a portable reproduction capsule")
    sp.add_argument("run_id")
    sp.add_argument("--format", default="capsule", choices=["capsule"])
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("recoveries", help="List OOM recovery campaigns")
    sp.add_argument("--project", default=None)
    sp.set_defaults(func=cmd_recoveries)

    sp = sub.add_parser("recovery", help="Show detail for one recovery campaign")
    sp.add_argument("campaign_id")
    sp.set_defaults(func=cmd_recovery)

    sp = sub.add_parser("ui", help="Launch the local web UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=7331)
    sp.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    sp.set_defaults(func=cmd_ui)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()