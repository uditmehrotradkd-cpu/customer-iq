"""Built-in data agent: turns short natural-language requests into training runs and datasets.

It is a deterministic intent parser (no external AI service): every answer is traceable to
the rules below, and it never executes user text as code or queries.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .datasets import SUMMARY_TABLES, DatasetError, build_dataset
from .knowledge import MIN_SIMILARITY, MIN_SIMILARITY_WITH_AI, answer_question
from .registry import ModelRegistry
from .services import SegmentationService

MAX_SYNTHETIC = 20_000
PREVIEW_ROWS = 6

FEATURE_ALIASES = {
    "average order value": "Avg_Order_Value",
    "avg order value": "Avg_Order_Value",
    "order value": "Avg_Order_Value",
    "aov": "Avg_Order_Value",
    "total spend": "Total_Spend",
    "spending": "Total_Spend",
    "spend": "Total_Spend",
    "income": "Income",
    "salary": "Income",
    "total purchases": "Total_Purchases",
    "purchases": "Total_Purchases",
    "orders": "Total_Purchases",
    "frequency": "Total_Purchases",
    "days since last purchase": "Recency",
    "recency": "Recency",
    "web visits": "NumWebVisitsMonth",
    "visits": "NumWebVisitsMonth",
    "deal ratio": "Deal_Ratio",
    "deals": "Deal_Ratio",
    "web share": "Web_Share",
    "catalog share": "Catalog_Share",
    "store share": "Store_Share",
    "campaigns accepted": "Campaigns_Accepted",
    "campaigns": "Campaigns_Accepted",
    "tenure": "Tenure_Days",
    "children": "Children",
    "kids": "Children",
    "age": "Age",
    "wine": "MntWines",
    "wines": "MntWines",
    "meat": "MntMeatProducts",
    "fish": "MntFishProducts",
    "fruit": "MntFruits",
    "fruits": "MntFruits",
    "sweets": "MntSweetProducts",
    "gold": "MntGoldProds",
}
WORD_OPERATORS = {
    "greater than or equal to": ">=",
    "at least": ">=",
    "minimum": ">=",
    "min": ">=",
    "less than or equal to": "<=",
    "at most": "<=",
    "maximum": "<=",
    "max": "<=",
    "more than": ">",
    "greater than": ">",
    "above": ">",
    "over": ">",
    "less than": "<",
    "below": "<",
    "under": "<",
    "equal to": "==",
    "equals": "==",
    "is": "==",
}
SYMBOL_OPERATORS = {">=": ">=", "<=": "<=", "=>": ">=", "=<": "<=", ">": ">", "<": "<", "==": "==", "=": "==", "!=": "!="}
TABLE_KEYWORDS = {
    "recommendation": "recommendations",
    "action": "recommendations",
    "playbook": "recommendations",
    "channel": "channel_mix",
    "category": "category_mix",
    "product": "category_mix",
    "z-score": "zscores",
    "zscore": "zscores",
    "fingerprint": "zscores",
    "index": "index",
    "k selection": "k_selection",
    "metrics": "k_selection",
    "elbow": "k_selection",
    "cleaning": "cleaning",
    "quality": "cleaning",
    "profile": "profiles",
    "summary": "profiles",
}
SEGMENT_STOPWORDS = {"customers", "customer", "segment", "segments", "shoppers", "the", "and"}

_feature_pattern = "|".join(sorted((re.escape(k) for k in FEATURE_ALIASES), key=len, reverse=True))
_op_pattern = "|".join(
    sorted([*(re.escape(k) for k in WORD_OPERATORS), *(re.escape(k) for k in SYMBOL_OPERATORS)], key=len, reverse=True)
)
_number = r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m|%)?"
CONDITION_RE = re.compile(rf"\b(?P<f>{_feature_pattern})\b\s*(?:is\s+)?(?P<op>{_op_pattern})\s*{_number}", re.I)
BETWEEN_RE = re.compile(rf"\b(?P<f>{_feature_pattern})\b\s*(?:is\s+)?between\s*{_number}\s*(?:and|to|-)\s*{_number}", re.I)
K_RE = re.compile(r"\bk\s*(?:=|of|to)?\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:segments|clusters|groups)\b", re.I)
COUNT_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", re.I)


def _to_number(raw: str, suffix: str | None) -> float:
    value = float(raw.replace(",", ""))
    suffix = (suffix or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


def parse_conditions(text: str) -> tuple[list[dict], str]:
    """Extract numeric filters; returns conditions and the text with those phrases removed."""
    conditions: list[dict] = []
    remaining = text
    for match in BETWEEN_RE.finditer(text):
        feature = FEATURE_ALIASES[match.group("f").lower()]
        low = _to_number(match.group(2), match.group(3))
        high = _to_number(match.group(4), match.group(5))
        conditions += [
            {"feature": feature, "op": ">=", "value": min(low, high)},
            {"feature": feature, "op": "<=", "value": max(low, high)},
        ]
        remaining = remaining.replace(match.group(0), " ")
    for match in CONDITION_RE.finditer(remaining):
        feature = FEATURE_ALIASES[match.group("f").lower()]
        op_text = match.group("op").lower()
        op = SYMBOL_OPERATORS.get(op_text) or WORD_OPERATORS[op_text]
        conditions.append({"feature": feature, "op": op, "value": _to_number(match.group(3), match.group(4))})
    remaining = CONDITION_RE.sub(" ", remaining)
    return conditions, remaining


def parse_segments(text: str, service: SegmentationService) -> list[int]:
    lowered = text.lower()
    found = {int(m.group(1)) for m in re.finditer(r"\b(?:segment|cluster|group)\s*#?\s*(\d{1,2})\b", lowered)}
    for seg_id, name in service.names.items():
        tokens = [t for t in re.split(r"[^a-z]+", name.lower()) if len(t) > 3 and t not in SEGMENT_STOPWORDS]
        stems = {t.rstrip("s") for t in tokens}
        if name.lower() in lowered or any(re.search(rf"\b{re.escape(s)}s?\b", lowered) for s in stems):
            found.add(seg_id)
    return sorted(s for s in found if s in service.names)


def parse_k(text: str) -> int | None:
    match = K_RE.search(text)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def parse_count(text: str, default: int = 500) -> int:
    for match in COUNT_RE.finditer(text):
        value = int(_to_number(match.group(1), match.group(2)))
        if value >= 10:
            return min(value, MAX_SYNTHETIC)
    return default


def _has(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(w)}", text) for w in words)


def _preview(frame: pd.DataFrame, columns: list[str]) -> dict:
    cols = [c for c in columns if c in frame.columns]
    head = frame[cols].head(PREVIEW_ROWS)
    return {"columns": cols, "rows": head.astype(object).where(head.notna(), None).values.tolist(), "total": int(len(frame))}


def _describe_conditions(conditions: list[dict]) -> str:
    return " and ".join(f"{c['feature']} {c['op']} {c['value']:,.0f}" if isinstance(c["value"], (int, float)) else f"{c['feature']} {c['op']} {c['value']}" for c in conditions)


def draft_summary(registry: ModelRegistry, draft_id: str) -> dict:
    service = registry.draft(draft_id)
    meta = registry.draft_meta(draft_id)
    live = registry.live()
    profiles = service.segmenter.profiles_.reset_index()
    return {
        **meta,
        "silhouette": round(service.segmenter.silhouette_, 4),
        "rationale": service.segmenter.k_selection_.rationale,
        "segments": [
            {
                "id": int(p["Segment"]),
                "name": p["Segment_Name"],
                "customers_pct": float(p["Customer_Share_%"]),
                "revenue_pct": float(p["Revenue_Share_%"]),
                "avg_spend": float(p["Avg_Total_Spend"]),
            }
            for _, p in profiles.iterrows()
        ],
        "live": {"k": live.segmenter.n_segments, "silhouette": round(live.segmenter.silhouette_, 4)},
    }


HELP_TEXT = (
    "Ask me anything about this site, the segments, the data, the methods or the metrics, for example:\n"
    "• \"What is the silhouette score?\" · \"Why 4 segments?\" · \"Tell me about digital deal-seekers\"\n"
    "• \"Which segment has the highest income?\" · \"Compare premium and affluent\"\n"
    "I can also do things:\n"
    "• \"give me premium customers with income over 70k\" · \"generate 1000 synthetic customers\"\n"
    "• \"download the recommendations table\" · \"upload template\"\n"
    "• attach a CSV, Excel or JSON file to analyse it, assign segments, train a model\n"
    "  or find segments in ANY dataset using all of its columns\n"
    "• \"train with 5 segments\" · \"status\" · \"publish\" · \"reset to original\""
)
SUGGESTIONS = [
    "Why 4 segments?",
    "Tell me about digital deal-seekers",
    "Which segment has the highest income?",
    "Give me premium customers with income over 70k",
    "Generate 500 synthetic customers",
]
QUESTION_START = re.compile(
    r"^(what|what's|whats|why|how|which|who|whom|when|where|is|are|does|do|did|can|could|should|would|will|explain|describe|define|"
    r"tell me|meaning|difference|compare|list the|i want to know|help me understand)\b"
)
POLITE_PREFIX = re.compile(r"^((please|kindly|can you|could you|would you|will you|i want you to|i need you to|i want to|i need to|i'd like to|let's|lets)\s+)+")
ACTION_START = re.compile(r"^(give|show|export|download|generate|create|list|find|get|send|fetch|pull|make|train|retrain|re-train|rebuild|publish|reset|restore|revert)\b")
GREETING = re.compile(r"^(hi|hello|hey|help|good (morning|afternoon|evening)|what can you do|what can you help)\b")
# Open-ended requests the AI should handle even when they mention customers or a segment.
AI_START = re.compile(
    r"^(now\s+|also\s+|then\s+)?(suggest|write|draft|compose|explain|summari[sz]e|brainstorm|recommend|translate|plan|advise|teach|tell me a|"
    r"help me (write|plan|think|understand|decide)|give me (ideas|advice|tips|a plan|an idea|some ideas)|ideas? for|how (can|should|do) i)\b"
)


def reply(text: str, **extra: Any) -> dict:
    return {"reply": text, "intent": extra.pop("intent", "info"), "dataset": None, "preview": None, "draft": None, "suggestions": [], "links": [], **extra}


def handle_message(message: str, registry: ModelRegistry, draft_id: str | None, ai_enabled: bool = False) -> dict:
    """Route one chat message: questions go to the knowledge base, commands to actions.

    With ``ai_enabled``, anything the built-in rules cannot answer returns intent ``general_question``
    so the caller can hand it to the AI model.
    """
    text = " ".join(message.strip().split())
    lowered = text.lower().rstrip("?!. ")
    service = registry.resolve(draft_id)
    is_question = text.rstrip().endswith("?") or bool(QUESTION_START.match(lowered))
    if service.mode == "generic":
        from .agent_generic import handle_generic  # imported here: agent_generic builds on this module

        return handle_generic(POLITE_PREFIX.sub("", lowered).strip(), registry, service, is_question=is_question, ai_enabled=ai_enabled)
    if not lowered or GREETING.match(lowered):
        return reply(HELP_TEXT, intent="help", suggestions=SUGGESTIONS)

    command = POLITE_PREFIX.sub("", lowered).strip()
    is_action = bool(ACTION_START.match(command))
    conditions, remaining = parse_conditions(lowered)
    min_score = MIN_SIMILARITY_WITH_AI if ai_enabled else MIN_SIMILARITY
    if ai_enabled and AI_START.match(command) and not conditions:
        return reply("", intent="general_question")

    if is_question and not is_action and not conditions:
        answer = answer_question(lowered, service, parse_segments(remaining, service), min_score)
        if answer:
            return reply(answer["reply"], intent="answer", links=answer["links"], suggestions=answer["suggestions"])
        if ai_enabled:
            return reply("", intent="general_question")

    result = _handle_command(command, registry, draft_id, service)
    if result["intent"] == "unknown":
        if ai_enabled:
            return reply("", intent="general_question")
        answer = answer_question(lowered, service, parse_segments(remaining, service))
        if answer:
            return reply(answer["reply"], intent="answer", links=answer["links"], suggestions=answer["suggestions"])
    return result


def _handle_command(lowered: str, registry: ModelRegistry, draft_id: str | None, service: SegmentationService) -> dict:
    using = "your draft model" if draft_id else "the live model"

    if _has(lowered, "status", "which model", "live model", "current model"):
        status = registry.status()
        source = "a model trained on your data" if status["live_source"] == "published" else "the original demo model"
        return reply(
            f"Your workspace uses {source}: {status['n_segments']} segments, silhouette {status['silhouette']:.3f}. "
            + ("You also have an unpublished draft open." if draft_id else "No draft is open."),
            intent="status",
        )

    if _has(lowered, "reset", "restore", "rollback", "roll back", "revert", "original model"):
        return reply("Resetting the live site to the original model.", intent="reset")

    if _has(lowered, "publish", "make it live", "go live", "make live"):
        if not draft_id:
            return reply("There is no draft to publish yet. Train a model first (e.g. \"train with 5 segments\").", intent="publish_missing")
        return reply("Publishing your draft model to the live site.", intent="publish")

    if _has(lowered, "cluster my", "auto-segment", "autosegment", "all columns", "any dataset", "any file", "segment my", "my own dataset"):
        return reply(
            "Attach any CSV, Excel or JSON file with the 📎 button (or drop it on the chat) and press Send (\"Segment & apply\"). "
            "I'll use every usable column, pick the number of segments from the data and switch every page of the site to it.",
            intent="cluster_upload",
        )

    if _has(lowered, "train", "retrain", "re-train", "rebuild", "build a model", "build model"):
        k = parse_k(lowered)
        if k is not None and not 2 <= k <= 10:
            return reply("Please choose between 2 and 10 segments.", intent="error")
        if _has(lowered, "upload", "my file", "my data", "csv", "excel", "json", "attached"):
            return reply(
                "Attach your CSV, Excel or JSON file with the 📎 button and choose \"Train model\" (optionally the number of segments). "
                "If it isn't in the customer format, choose \"Find segments (all columns)\" and I'll segment it on its own columns.",
                intent="train_upload",
            )
        return reply("Training on the built-in dataset" + (f" with {k} segments" if k else " (k chosen automatically)") + "…", intent="train", k=k)

    if _has(lowered, "template", "blank", "schema", "format", "sample file"):
        dataset = {"kind": "template"}
        frame, _, _ = build_dataset(service, dataset)
        return reply(
            "Here is a blank upload template in the exact format I need for training or batch scoring (2 example rows).",
            intent="template",
            dataset=dataset,
            preview=_preview(frame, list(frame.columns)[:8]),
        )

    if _has(lowered, "synthetic", "generate", "fake", "simulate", "dummy", "mock"):
        segments = parse_segments(lowered, service)
        n = parse_count(lowered)
        dataset = {"kind": "synthetic", "n": n, "segments": segments, "seed": 42}
        frame, _, info = build_dataset(service, dataset)
        target = ", ".join(service.names[s] for s in segments) if segments else "all segments (in their real proportions)"
        return reply(
            f"Generated {info['rows']:,} synthetic customers for {target} using {using}. "
            f"{info['fidelity']:.0%} of them are assigned back to their intended segment, so they follow the real patterns. "
            "Rows are sampled from each segment's statistical profile, not copied from real customers.",
            intent="synthetic",
            dataset=dataset,
            preview=_preview(frame, ["ID", "Intended_Segment", "Income", "MntWines", "NumWebPurchases", "NumStorePurchases", "Recency"]),
        )

    if _has(lowered, "report") or (_has(lowered, "chart", "graph", "visual") and _has(lowered, "download", "give", "get", "send", "export", "make", "create")):
        return reply(
            f"Here is an Excel report of {using} with charts: segment sizes, a chart per key measure, "
            "a plain-language profile and next steps for every segment, and all customers with their segment.",
            intent="report",
            dataset={"kind": "report", "filename": "segment_report.xlsx"},
        )

    table = next((t for key, t in TABLE_KEYWORDS.items() if key in lowered), None)
    if table and not CONDITION_RE.search(lowered):
        dataset = {"kind": "summary", "table": table}
        frame, _, _ = build_dataset(service, dataset)
        return reply(
            f"Here is the {table.replace('_', ' ')} table from {using}.",
            intent="summary",
            dataset=dataset,
            preview=_preview(frame, list(frame.columns)[:7]),
            suggestions=[f"Download the {t.replace('_', ' ')} table" for t in SUMMARY_TABLES if t != table][:3],
        )

    conditions, remaining = parse_conditions(lowered)
    segments = parse_segments(remaining, service)
    wants_data = conditions or segments or _has(lowered, "customer", "give", "show", "export", "list", "find", "download", "dataset", "data", "who")
    if wants_data:
        if _has(lowered, "parent", "with children", "with kids"):
            conditions.append({"feature": "Is_Parent", "op": "==", "value": "Parent"})
        elif _has(lowered, "no children", "without children", "no kids"):
            conditions.append({"feature": "Is_Parent", "op": "==", "value": "No children"})
        dataset = {"kind": "filtered", "segments": segments, "conditions": conditions}
        frame, _, info = build_dataset(service, dataset)
        where = f" in {', '.join(service.names[s] for s in segments)}" if segments else ""
        rule = f" with {_describe_conditions(conditions)}" if conditions else ""
        text_reply = f"Found {info['rows']:,} customers{where}{rule} ({using})."
        if info["rows"] == 0:
            text_reply += " Try loosening the filters."
        return reply(
            text_reply,
            intent="filtered",
            dataset=dataset,
            preview=_preview(frame, ["ID", "Segment_Name", "Income", "Total_Spend", "Total_Purchases", "Recency", "Deal_Ratio"]),
        )

    return reply("Sorry, I didn't understand that. " + HELP_TEXT, intent="unknown", suggestions=SUGGESTIONS)


__all__ = ["DatasetError", "draft_summary", "handle_message", "parse_conditions", "parse_count", "parse_k", "parse_segments"]
