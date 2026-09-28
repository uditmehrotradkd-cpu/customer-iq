"""Chat commands for workspaces built from the user's own dataset (any columns)."""
from __future__ import annotations

import re

from segmentation.auto_segment import fmt_value, pretty

from .agent import AI_START, _has, _preview, parse_segments, reply
from .generic_service import GenericService
from .registry import ModelRegistry

GLOSSARY = {
    "silhouette": "The silhouette score (−1 to 1) measures how much closer each row is to its own segment than to the nearest other one. Higher is better; 0.15–0.35 is typical for real business data.",
    "k-means": "K-Means groups rows so that each row is as close as possible to the centre of its segment. I scale every column first so no single column dominates.",
    "segment": "A segment is a group of rows that look alike across the columns of your file. Each one is named after what makes it most different from the average.",
}


def _help(service: GenericService) -> str:
    first = service.names[service.segment_ids[0]]
    metric = pretty(service.metrics[0]) if service.metrics else "a column"
    return (
        f"Your workspace runs on {service.source} ({len(service.customers):,} rows, {service.result.k} segments). Ask me, for example:\n"
        f"• \"Why this number of segments?\" · \"Tell me about {first}\" · \"Which segment has the highest {metric.lower()}?\"\n"
        f"• \"Compare all segments\" · \"Which columns were used?\" · \"Download the segment report with charts\"\n"
        f"• \"Give me the rows in {first}\" · \"Download the profiles table\" · \"Status\" · \"Reset to original\"\n"
        "• attach another file and choose \"Assign segments\" to place new rows into these segments, or \"Find segments\" to rebuild the workspace."
    )


def suggestions_for(service: GenericService) -> list[str]:
    first = service.names[service.segment_ids[0]]
    items = ["Why this number of segments?", f"Tell me about {first}", "Download the segment report with charts"]
    if service.metrics:
        items.insert(2, f"Which segment has the highest {pretty(service.metrics[0]).lower()}?")
    return items


def _find_feature(text: str, service: GenericService) -> str | None:
    matches = []
    for col in service.numeric_features:
        for alias in {col.lower(), pretty(col).lower(), col.lower().replace("_", " ")}:
            if alias and alias in text:
                matches.append((len(alias), col))
    return max(matches)[1] if matches else None


def _describe(service: GenericService, seg: int) -> str:
    rec = next(r for r in service.recommendations if r["segment"] == seg)
    profile = service.profiles.loc[seg]
    lines = [f"{service.names[seg]}: {int(profile['Customers']):,} rows ({profile['Customer_Share_%']:.1f}%)."]
    lines += [f"• {t}" for t in rec["traits"]] or ["• Close to the overall average on every measure."]
    lines.append("Next step: " + rec["actions"][0])
    return "\n".join(lines)


def _compare(service: GenericService, segs: list[int]) -> str:
    segs = segs or service.segment_ids
    lines = ["Averages per segment:"]
    for m in service.metrics[:5]:
        lines.append(f"• {pretty(m)}: " + " | ".join(f"{service.names[s]} {fmt_value(service.profiles.loc[s, f'Avg {m}'])}" for s in segs))
    return "\n".join(lines)


def handle_generic(lowered: str, registry: ModelRegistry, service: GenericService, is_question: bool = False, ai_enabled: bool = False) -> dict:
    suggestions = suggestions_for(service)
    if not lowered or re.match(r"^(hi|hello|hey|help|what can you do)\b", lowered):
        return reply(_help(service), intent="help", suggestions=suggestions)
    if ai_enabled and AI_START.match(lowered):
        return reply("", intent="general_question")

    if _has(lowered, "reset", "restore", "rollback", "revert", "original model", "demo data"):
        return reply("Resetting your workspace to the original demo data.", intent="reset")

    if _has(lowered, "status", "which model", "current model", "my workspace"):
        return reply(
            f"Your workspace runs on {service.source}: {len(service.customers):,} rows in {service.result.k} segments (silhouette {service.result.silhouette:.3f}). "
            "Every page of the site shows this dataset.",
            intent="status",
            suggestions=suggestions,
        )

    if _has(lowered, "report") or (_has(lowered, "chart", "graph", "visual") and _has(lowered, "download", "give", "get", "send", "export", "make", "create")):
        return reply(
            f"Here is an Excel report of {service.source} with charts: segment sizes, a chart per key measure, what makes each segment different and every row with its segment.",
            intent="report",
            dataset={"kind": "report", "filename": "segment_report.xlsx"},
        )

    if _has(lowered, "train", "retrain", "rebuild", "re-segment", "resegment"):
        return reply(
            "To rebuild the segments, attach your file again with 📎 and choose \"Find segments (all columns)\" (pick a number of segments if you like). "
            "Customer-format files can also use \"Train customer model\".",
            intent="train_upload",
        )

    for term, text in GLOSSARY.items():
        if term in lowered and _has(lowered, "what", "mean", "explain", "define"):
            return reply(text, intent="answer", suggestions=suggestions)

    about_segments = _has(lowered, "segment", "cluster", "group", "k=", "silhouette")
    if about_segments and _has(lowered, "why", "how many", "number of", "how was", "method", "how did", "silhouette", "k="):
        return reply(service.rationale, intent="answer", links=[["Model & methodology", "#/diagnostics"]], suggestions=suggestions)

    if _has(lowered, "column", "feature", "field", "variable") and _has(lowered, "used", "which", "my", "dataset", "file", "list"):
        used = ", ".join(pretty(c) for c in [*service.numeric_features, *service.categorical_features][:25])
        dropped = ", ".join(f"{c} ({why})" for c, why in service.result.encoder.dropped.items()) or "none"
        return reply(f"Columns used to form segments: {used}.\nNot used: {dropped}.", intent="answer", suggestions=suggestions)

    segments = parse_segments(lowered, service)
    feature = _find_feature(lowered, service)
    low = _has(lowered, "lowest", "least", "smallest", "minimum", "fewest")
    if feature and (low or _has(lowered, "highest", "most", "largest", "biggest", "maximum", "top")):
        means = service.profiles[f"Avg {feature}"] if f"Avg {feature}" in service.profiles else service.customers.groupby("Segment")[feature].mean()
        ranked = means.sort_values(ascending=low)
        best = int(ranked.index[0])
        order = "; ".join(f"{service.names[int(s)]} {fmt_value(v)}" for s, v in ranked.items())
        return reply(f"{service.names[best]} has the {'lowest' if low else 'highest'} average {pretty(feature)}: {fmt_value(ranked.iloc[0])}. Ranking: {order}.", intent="answer", suggestions=suggestions)

    if _has(lowered, "compare", "difference", "versus", "vs") and (segments or about_segments):
        return reply(_compare(service, segments), intent="answer", links=[["Segment profiles", "#/segments"]], suggestions=suggestions)

    if _has(lowered, "profile table", "profiles", "summary", "table"):
        dataset = {"kind": "summary", "table": "profiles"}
        frame = service.profiles.reset_index()
        return reply("Here is the profile of every segment (averages of the key columns).", intent="summary", dataset=dataset, preview=_preview(frame, list(frame.columns)[:7]))

    wants_rows = _has(lowered, "give", "show", "list", "rows", "export", "download", "data", "records", "who")
    if ai_enabled and is_question and not segments:
        return reply("", intent="general_question")
    if segments and not wants_rows:
        return reply(_describe(service, segments[0]), intent="answer", links=[["Open profile", f"#/segments/{segments[0]}"]], suggestions=suggestions)

    if wants_rows or segments:
        dataset = {"kind": "filtered", "segments": segments, "conditions": []}
        rows = service.customers[service.customers["Segment"].isin(segments)] if segments else service.customers
        where = f" in {', '.join(service.names[s] for s in segments)}" if segments else ""
        return reply(
            f"Found {len(rows):,} rows{where}.",
            intent="filtered",
            dataset=dataset,
            preview=_preview(rows, ["ID", "Segment_Name", *service.metrics[:5]]),
        )

    if ai_enabled:
        return reply("", intent="general_question")
    return reply("Sorry, I didn't understand that. " + _help(service), intent="unknown", suggestions=suggestions)
