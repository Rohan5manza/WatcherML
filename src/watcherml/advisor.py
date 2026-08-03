"""Optional local-LLM advisor, powered by Ollama.

Design constraint from the product spec: the deterministic rule engine in
`failures.py` and the deterministic diff in `diff.py` are always the source
of truth. This module only ever adds a clearly-labeled natural-language
layer on top of facts that were already computed without it. If Ollama
isn't running, every function here returns None and the rest of the product
works exactly as if this file didn't exist.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"  # any small local model works: qwen2.5:3b, phi3, etc.
TIMEOUT_SECONDS = 30


def is_available(host: str = DEFAULT_HOST) -> bool:
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _chat(messages: list, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
          keep_alive: Optional[int] = None) -> Optional[str]:
    body = {"model": model, "messages": messages, "stream": False}
    if keep_alive is not None:
        # Buffy-style single-GPU setups: unload the model immediately after
        # this response (keep_alive=0) so a training trial gets the full card.
        body["keep_alive"] = keep_alive
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode())
            return body.get("message", {}).get("content", "").strip() or None
    except urllib.error.URLError:
        return None  # Ollama not running, or model not pulled -- degrade silently
    except Exception:
        return None


_SYSTEM_PROMPT = (
    "You are a terse, practical ML failure advisor embedded in a developer tool. "
    "You are always given a deterministic diagnosis that has ALREADY been computed "
    "by rules, plus evidence. Do not contradict the deterministic diagnosis or invent "
    "a different root cause. Your job is only to: (1) explain it in plain language "
    "specific to the evidence given, and (2) suggest 1-3 concrete, specific next "
    "actions, ranked by how likely they are to help. Be concise -- 5 sentences max. "
    "Never claim certainty; you are reasoning from limited evidence."
)


def explain_failure(capsule: dict, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> Optional[str]:
    """Add a natural-language explanation on top of an already-computed failure capsule."""
    prompt = (
        f"Deterministic diagnosis: {capsule['diagnosis']['rule']} -- {capsule['diagnosis']['summary']}\n"
        f"Exception: {capsule['exception_type']}: {capsule['message']}\n"
        f"Config: {json.dumps(capsule['evidence'].get('config', {}))}\n"
        f"Recent metrics: {json.dumps(capsule['evidence'].get('recent_metrics', []))}\n"
        f"Resource state at failure: {json.dumps(capsule['evidence'].get('resource_state_at_failure', {}))}\n"
        f"Git dirty: {capsule['evidence'].get('git', {}).get('dirty')}\n\n"
        "Explain what likely happened here and suggest next actions."
    )
    return _chat(
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        model=model, host=host,
    )


_DIFF_SYSTEM_PROMPT = (
    "You are a terse ML experiment analyst embedded in a developer tool. You are given "
    "a deterministic diff between two runs (already computed -- config changes, metric "
    "changes, resource changes). Write ONE short paragraph (2-4 sentences) connecting "
    "the changes to the results, in the style of: 'The smaller batch eliminated the OOM "
    "condition, while the lower learning rate stabilized training after epoch 3.' "
    "Only reason from the facts given. Do not invent numbers that weren't provided."
)


def explain_diff(diff: dict, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> Optional[str]:
    """Add a 'likely explanation' narrative on top of an already-computed run diff."""
    prompt = (
        f"Config changes: {json.dumps(diff['config_diff'])}\n"
        f"Package changes: {json.dumps(diff['package_diff'][:10])}\n"
        f"Metric changes: {json.dumps(diff['metric_diff'])}\n"
        f"Resource changes: {json.dumps(diff['resource_diff'])}\n"
        f"Exit status: {diff['exit_status_a']} -> {diff['exit_status_b']}\n\n"
        "What likely explains the change in outcome?"
    )
    return _chat(
        [{"role": "system", "content": _DIFF_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        model=model, host=host,
    )


_SUGGEST_SYSTEM_PROMPT = (
    "You are an ML experimentation assistant embedded in a developer tool. You are given "
    "the full history of past runs for a project (configs, final metrics, and whether each "
    "run failed and why). Propose ONE next config to try, as a JSON object with two keys: "
    "\"config\" (an object with the same keys as past configs, changed where you think it "
    "will help) and \"rationale\" (1-2 sentences). Change as few keys as possible. "
    "Respond with ONLY the JSON object, no other text. "
    "Be aware you are working from very few data points -- prefer small, explainable, "
    "reversible changes over large speculative ones."
)


def _parse_json_object(raw: Optional[str], required_key: Optional[str] = None) -> Optional[dict]:
    """Shared defensive JSON parsing: models sometimes wrap JSON in ```json fences
    despite instructions not to. Returns None (never raises) on anything unparseable."""
    if not raw:
        return None
    cleaned = raw.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if required_key is not None and required_key not in parsed:
        return None
    return parsed


MAX_JSON_ATTEMPTS = 2  # confirmed via real testing: llama3.2:1b failed to produce
# parseable JSON on 2 of 5 identical calls -- small models are inconsistent about
# following "respond with ONLY JSON" instructions call-to-call, not just occasionally.


def _chat_json(messages: list, required_key: str, model: str = DEFAULT_MODEL,
               host: str = DEFAULT_HOST, keep_alive: Optional[int] = None,
               max_attempts: int = MAX_JSON_ATTEMPTS) -> Optional[dict]:
    """Calls the model and parses JSON, retrying if the response didn't parse.
    The model is kept warm (keep_alive left at Ollama's default) across retries
    within this call -- the caller-supplied keep_alive is only applied on the
    final attempt, so a retry doesn't pay a second full reload cost on top of
    the first, while the GPU still gets freed by the time this function returns.
    """
    conversation = list(messages)
    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        raw = _chat(conversation, model=model, host=host,
                    keep_alive=keep_alive if is_last else None)
        parsed = _parse_json_object(raw, required_key=required_key)
        if parsed is not None:
            return parsed
        if not is_last:
            conversation = messages + [
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content": (
                    f"That wasn't valid JSON, or was missing the required \"{required_key}\" "
                    "key. Respond again with ONLY a JSON object, no other text."
                )},
            ]
    return None


def suggest_next_config(run_history: list, goal_metric: str, goal_direction: str = "maximize",
                         model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> Optional[dict]:
    """Suggest the next config to try, given run history. Returns None if unavailable
    or unparseable after retrying -- callers must have a non-LLM fallback and must
    never block on this.
    """
    prompt = (
        f"Goal: {goal_direction} '{goal_metric}'.\n"
        f"Run history (oldest first): {json.dumps(run_history)}\n\n"
        "Propose the next config as a JSON object: {\"config\": {...}, \"rationale\": \"...\"}"
    )
    return _chat_json(
        [{"role": "system", "content": _SUGGEST_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        required_key="config", model=model, host=host,
    )


# ============================================================================
# OOM Recovery Agent roles: Diagnostician (ranks hypotheses) and Planner
# (proposes candidate patches). Both are advisory only -- recovery.py's policy
# engine validates every proposed patch before anything executes, and both
# roles have deterministic fallbacks so the agent still functions with no LLM.
# ============================================================================

_HYPOTHESIS_SYSTEM_PROMPT = (
    "You are a memory-diagnosis specialist for ML training failures, embedded in a "
    "developer tool. You are given a factual observation report about a CUDA "
    "out-of-memory failure -- it was already computed deterministically, not by you. "
    "Produce 1-3 ranked hypotheses for what specifically drove memory usage over the "
    "limit, using ONLY the evidence given -- do not invent numbers or facts not present. "
    "Respond with ONLY a JSON object: "
    "{\"hypotheses\": [{\"cause\": \"short label\", \"explanation\": \"1 sentence\", "
    "\"confidence\": 0.0-1.0}, ...]}, ordered most confident first."
)


def rank_memory_hypotheses(observation: dict, model: str = DEFAULT_MODEL,
                            host: str = DEFAULT_HOST, keep_alive: Optional[int] = 0) -> Optional[list]:
    """Diagnostician role. Returns a list of hypothesis dicts, or None if
    Ollama is unavailable or the response didn't parse after retrying --
    caller must fall back."""
    prompt = f"Observation report:\n{json.dumps(observation, indent=2)}\n\nRank the likely causes."
    parsed = _chat_json(
        [{"role": "system", "content": _HYPOTHESIS_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        required_key="hypotheses", model=model, host=host, keep_alive=keep_alive,
    )
    if parsed is None or not isinstance(parsed.get("hypotheses"), list):
        return None
    return parsed["hypotheses"]


_RECOVERY_PLANNER_SYSTEM_PROMPT = (
    "You are an ML memory-recovery planner embedded in a developer tool. A training run "
    "failed with a CUDA out-of-memory error. You may ONLY propose changes to these exact "
    "keys, and no others: batch_size, gradient_accumulation_steps, precision, "
    "sequence_length, gradient_checkpointing, num_workers. Any other key will be rejected "
    "by a validator you cannot see. Given the current config and ranked hypotheses about "
    "the memory cause, propose 2-3 DISTINCT candidate patches, each targeting a different "
    "mechanism (e.g. one lowers batch_size, one changes precision, one enables gradient "
    "checkpointing) so they can be tested independently and compared. Keep changes small "
    "and explainable. Respond with ONLY a JSON object: "
    "{\"candidates\": [{\"patch\": {\"key\": value, ...}, \"rationale\": \"1 sentence\", "
    "\"confidence\": 0.0-1.0}, ...]}."
)


def propose_recovery_patches(observation: dict, hypotheses: list, base_config: dict,
                              model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                              keep_alive: Optional[int] = 0) -> Optional[list]:
    """Planner role. Returns a list of {"patch", "rationale", "confidence"} dicts,
    UNVALIDATED -- recovery.py's policy engine must filter these before use.
    Returns None if Ollama is unavailable or the response didn't parse after
    retrying."""
    prompt = (
        f"Current config: {json.dumps(base_config)}\n"
        f"Observation: {json.dumps(observation, indent=2)}\n"
        f"Ranked hypotheses: {json.dumps(hypotheses, indent=2)}\n\n"
        "Propose 2-3 candidate patches."
    )
    parsed = _chat_json(
        [{"role": "system", "content": _RECOVERY_PLANNER_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        required_key="candidates", model=model, host=host, keep_alive=keep_alive,
    )
    if parsed is None or not isinstance(parsed.get("candidates"), list):
        return None
    return parsed["candidates"]