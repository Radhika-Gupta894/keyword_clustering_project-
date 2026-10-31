"""OPTIONAL (A7): transparent heuristic intent *suggestions*. EXPERIMENTAL / not validated."""
import re

_RULES = [
    ("transactional", r"\b(buy|price|prices|cheap|discount|coupon|order|deal|deals|for sale|near me|hire|subscribe)\b"),
    ("commercial investigation", r"\b(best|top|review|reviews|vs|versus|compare|comparison|alternatives?|rated)\b"),
    ("informational", r"\b(how|what|why|when|who|guide|tutorial|tips|ideas|meaning|examples?|learn)\b"),
    ("navigational", r"\b(login|log in|sign in|official site|website|app download|\.com)\b"),
]


def suggest_intent(keyword: str) -> str:
    """Exactly one rule matched -> that label; none or conflicting matches -> 'uncertain' (needs human review)."""
    hits = [label for label, pat in _RULES if re.search(pat, keyword.lower())]
    return hits[0] if len(hits) == 1 else "uncertain"
