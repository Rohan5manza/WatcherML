"""Lightweight redaction for secrets that might appear in captured notebook cell
source (e.g. an API key pasted directly into a cell instead of loaded from env).

This is intentionally simple pattern-matching, not a security boundary --
it catches common accidental-paste patterns, not determined secret exfiltration.
Users can add their own patterns via `add_pattern()`.
"""
from __future__ import annotations

import re

_DEFAULT_PATTERNS = [
    # key="value" / token='value' / password=value / secret=value assignments
    re.compile(r"""(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*["']?[^\s"']{4,}["']?"""),
    # common cloud/provider key prefixes
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),          # OpenAI/Anthropic-style keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),               # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),     # GitHub tokens
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),   # Slack tokens
]

_custom_patterns: list = []


def add_pattern(compiled_regex: "re.Pattern"):
    """Register an additional redaction pattern, e.g. for an internal secret format."""
    _custom_patterns.append(compiled_regex)


def redact(text: str) -> str:
    if not text:
        return text
    for pattern in _DEFAULT_PATTERNS + _custom_patterns:
        text = pattern.sub("[REDACTED]", text)
    return text
