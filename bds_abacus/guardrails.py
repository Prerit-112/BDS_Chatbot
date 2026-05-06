"""Input/output checks: length, injection heuristics, optional OpenAI moderation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Common jailbreak / instruction-override phrasing (heuristic, not complete).
_INJECTION_PATTERNS = re.compile(
    r"|".join(
        re.escape(p)
        for p in (
            "ignore previous instructions",
            "ignore the above",
            "disregard the above",
            "system prompt",
            "you are now",
            "developer mode",
            "dan mode",
            "jailbreak",
            "bypass safety",
            "reveal your prompt",
            "output the",
            "begin base64",
            "### new instructions",
        )
    ),
    re.IGNORECASE,
)

# Suspicious long delimiter blocks sometimes used in attacks.
_BLOB_PATTERN = re.compile(r"```[\s\S]{2000,}```")


@dataclass
class InputGuardResult:
    ok: bool
    text: str
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class OutputGuardResult:
    ok: bool
    text: str
    reason: str | None = None
    flags: list[str] = field(default_factory=list)


def normalize_user_text(raw: str) -> str:
    t = raw.strip()
    t = unicodedata.normalize("NFC", t)
    return t


def check_input(
    text: str,
    *,
    max_chars: int,
) -> InputGuardResult:
    n = len(text)
    if n == 0:
        return InputGuardResult(
            ok=False, text="", reason="empty_message", flags=["empty"]
        )
    if n > max_chars:
        return InputGuardResult(
            ok=False,
            text="",
            reason=f"message_too_long (max {max_chars} characters)",
            flags=["length"],
        )
    if _BLOB_PATTERN.search(text):
        return InputGuardResult(
            ok=False,
            text="",
            reason="suspicious_content",
            flags=["suspicious_blob"],
        )
    if _INJECTION_PATTERNS.search(text):
        return InputGuardResult(
            ok=False,
            text="",
            reason="message_blocked_safety",
            flags=["injection_heuristic"],
        )
    return InputGuardResult(ok=True, text=text, flags=[])


def moderate_text_sync(client: Any, text: str, model: str) -> tuple[bool, list[str]]:
    """
    OpenAI Moderation API. Returns (flagged, category_names that triggered).
    If the API errors, returns (True, ['moderation_error']) to fail closed.
    """
    if not (text and text.strip()):
        return False, []
    try:
        r = client.moderations.create(model=model, input=text[:32000])
    except Exception:
        return True, ["moderation_error"]
    if not r.results:
        return True, ["moderation_error"]
    res = r.results[0]
    if not res.flagged:
        return False, []
    cats = getattr(res, "categories", None)
    if cats is None:
        return True, ["moderation"]
    out: list[str] = []
    d = cats.model_dump() if hasattr(cats, "model_dump") else dict(cats)
    for k, v in d.items():
        if v:
            out.append(k)
    return True, (out or ["moderation"])


def check_output_sanity(answer: str) -> OutputGuardResult:
    """Post-check for empty model output."""
    a = (answer or "").strip()
    if not a:
        return OutputGuardResult(
            ok=False, text="", reason="empty_model_output", flags=["empty"]
        )
    return OutputGuardResult(ok=True, text=answer, flags=[])
