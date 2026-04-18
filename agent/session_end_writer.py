from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_HARD_TOOL_NAMES = {
    "patch",
    "write_file",
    "terminal",
    "process",
    "delegate_task",
    "cronjob",
    "skill_manage",
}

_WRITE_TOOL_NAMES = {"patch", "write_file"}
_SUPPORTING_TOOL_NAMES = {
    "read_file",
    "search_files",
    "web_extract",
    "web_search",
    "session_search",
    "skill_view",
    "browser_snapshot",
    "browser_console",
}

_DECISION_MARKERS = (
    "we should",
    "we will",
    "decided",
    "decision",
    "adopt",
    "switch to",
    "keep ",
    "remove ",
    "use ",
    "决定",
    "改成",
    "采用",
    "保留",
    "删除",
    "不再",
    "结论",
)

_DETAIL_MARKERS = (
    "config",
    "parameter",
    "flag",
    "path",
    "function",
    "class",
    "symbol",
    "workflow",
    "参数",
    "配置",
    "路径",
    "函数",
    "类",
    "符号",
    "环境变量",
)

_CORRECTION_MARKERS = (
    "not this",
    "that's not",
    "instead",
    "actually",
    "不是",
    "重点是",
    "我要的是",
    "不对",
)

_FOLLOW_UP_MARKERS = (
    "follow-up",
    "next step",
    "remaining",
    "todo",
    "need to",
    "later",
    "下次",
    "后续",
    "下一步",
    "未决",
)


@dataclass
class SessionContext:
    session_id: str
    title: str
    topic_slug: str
    user_goal: str
    final_outcome: str
    key_decisions: List[str]
    stable_conclusions: List[str]
    important_details: List[str]
    user_corrections: List[str]
    files_touched: List[str]
    tool_events: List[Dict[str, Any]]
    read_sources: List[str]
    follow_ups: List[str]
    transcript_snippets: List[str]
    doc_type: str
    score: int
    score_reasons: List[str]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif item:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def _slugify(value: str, fallback: str = "session") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"\[\[|\]\]", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or fallback


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        kept.append(value)
    return kept


def _normalize_tool_events(tool_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: list[Dict[str, Any]] = []
    for event in tool_events:
        name = str((event or {}).get("name") or "").strip()
        if not name:
            continue
        if name not in _HARD_TOOL_NAMES and name not in _SUPPORTING_TOOL_NAMES:
            continue
        kept.append(
            {
                "event_type": str((event or {}).get("event_type") or "tool.event"),
                "name": name,
                "args": dict((event or {}).get("args") or {}),
                "preview": (event or {}).get("preview"),
            }
        )
    return kept


def _extract_files_touched(tool_events: Iterable[Dict[str, Any]]) -> List[str]:
    paths: list[str] = []
    for event in tool_events:
        args = event.get("args") or {}
        for key in ("path", "file_path"):
            value = args.get(key)
            if isinstance(value, str) and value:
                paths.append(value)
    return _unique_keep_order(paths)


def _extract_read_sources(tool_events: Iterable[Dict[str, Any]]) -> List[str]:
    sources: list[str] = []
    for event in tool_events:
        name = event.get("name")
        args = event.get("args") or {}
        if name in {"read_file", "search_files", "skill_view"}:
            for key in ("path", "file_path", "pattern"):
                value = args.get(key)
                if isinstance(value, str) and value:
                    sources.append(value)
        elif name in {"web_extract", "web_search", "session_search"}:
            for key in ("urls", "query"):
                value = args.get(key)
                if isinstance(value, str) and value:
                    sources.append(value)
                elif isinstance(value, list):
                    sources.extend(str(v) for v in value if v)
    return _unique_keep_order(sources)


def _title_from_messages(messages: List[Dict[str, Any]], session_title: str | None) -> str:
    if session_title:
        return session_title.strip()
    for msg in messages:
        if msg.get("role") == "user":
            text = _content_to_text(msg.get("content"))
            if text:
                return text.splitlines()[0][:80].strip()
    return "session log"


def _extract_user_goal(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = _content_to_text(msg.get("content"))
            if text:
                return text
    return ""


def _extract_final_outcome(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            text = _content_to_text(msg.get("content"))
            if text:
                return text
    return ""


def _extract_snippets(messages: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    snippets: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        text = _content_to_text(msg.get("content"))
        if role == "assistant" and msg.get("tool_calls"):
            names = []
            for item in (msg.get("tool_calls") or [])[:5]:
                fn = ((item or {}).get("function") or {}).get("name")
                if fn:
                    names.append(fn)
            if names:
                text = f"tool calls: {', '.join(names)}"
        if not text and role != "assistant":
            continue
        label = "User" if role == "user" else "Assistant" if role == "assistant" else "Tool"
        snippets.append(f"- {label}: {text[:400].strip()}" if text else f"- {label}: (empty)")
        if len(snippets) >= limit:
            break
    return snippets or ["- No retained transcript snippets"]


def _extract_marked_lines(messages: List[Dict[str, Any]], markers: tuple[str, ...], *, roles: set[str]) -> List[str]:
    hits: list[str] = []
    for msg in messages:
        if msg.get("role") not in roles:
            continue
        text = _content_to_text(msg.get("content"))
        lowered = text.lower()
        if text and any(marker in lowered for marker in markers):
            hits.append(text[:240])
    return _unique_keep_order(hits)


def _extract_important_details(messages: List[Dict[str, Any]], tool_events: List[Dict[str, Any]]) -> List[str]:
    details: list[str] = []
    for msg in messages:
        text = _content_to_text(msg.get("content"))
        if not text:
            continue
        if any(marker in text.lower() for marker in _DETAIL_MARKERS):
            details.append(text[:240])
        for match in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_./:-]{2,}\b", text):
            if any(ch in match for ch in ("/", ".", "_")):
                details.append(match)
    for path in _extract_files_touched(tool_events):
        details.append(path)
    for event in tool_events:
        args = event.get("args") or {}
        for key in ("command", "query"):
            value = args.get(key)
            if isinstance(value, str) and value:
                details.append(value[:180])
    return _unique_keep_order(details)[:12]


def _has_hard_signal(messages: List[Dict[str, Any]], tool_events: List[Dict[str, Any]]) -> bool:
    if any(event.get("name") in _HARD_TOOL_NAMES for event in tool_events):
        return True
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    joined = "\n".join(_content_to_text(m.get("content")) for m in messages).lower()
    if any(marker in joined for marker in _DECISION_MARKERS):
        return True
    if any(marker in joined for marker in _DETAIL_MARKERS):
        return True
    if any(marker in joined for marker in _CORRECTION_MARKERS):
        return True
    return False


def _score_session(messages: List[Dict[str, Any]], tool_events: List[Dict[str, Any]]) -> tuple[int, List[str]]:
    score = 0
    reasons: list[str] = []
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    decisions = _extract_marked_lines(messages, _DECISION_MARKERS, roles={"assistant", "user"})
    corrections = _extract_marked_lines(messages, _CORRECTION_MARKERS, roles={"user"})
    details = _extract_important_details(messages, tool_events)
    follow_ups = _extract_marked_lines(messages, _FOLLOW_UP_MARKERS, roles={"assistant", "user"})

    if len(tool_events) >= 3:
        score += 2
        reasons.append("3+ key tool events")
    elif len(tool_events) >= 1:
        score += 1
        reasons.append("at least 1 key tool event")

    if decisions:
        score += 2
        reasons.append("explicit decision language")
    if corrections:
        score += 2
        reasons.append("user correction captured")
    if details:
        score += 2
        reasons.append("implementation details captured")
    if follow_ups:
        score += 1
        reasons.append("follow-up items captured")
    if len(user_messages) >= 2:
        score += 1
        reasons.append("multi-turn topic")
    if any(len(_content_to_text(m.get("content"))) > 240 for m in assistant_messages):
        score += 1
        reasons.append("substantial assistant summary")

    if len(user_messages) <= 1 and not tool_events:
        score -= 2
        reasons.append("trivial/no-tool interaction")
    if not decisions and not details and not follow_ups and len(messages) <= 2:
        score -= 2
        reasons.append("little durable information")
    return score, reasons


def worth_storing(messages: List[Dict[str, Any]], tool_events: Optional[List[Dict[str, Any]]] = None) -> bool:
    normalized_events = _normalize_tool_events(tool_events or [])
    filtered_messages = [m for m in messages if m.get("role") not in {"system", "session_meta"}]
    if not filtered_messages:
        return False
    if _has_hard_signal(filtered_messages, normalized_events):
        return True
    score, _ = _score_session(filtered_messages, normalized_events)
    return score >= 3


def _choose_doc_type(
    messages: List[Dict[str, Any]],
    tool_events: List[Dict[str, Any]],
    key_decisions: List[str],
    files_touched: List[str],
) -> str:
    if files_touched or any(event.get("name") in _WRITE_TOOL_NAMES for event in tool_events):
        return "task-log"
    if any(event.get("name") in {"terminal", "process"} for event in tool_events):
        return "task-log"
    if key_decisions and len(key_decisions) >= 2:
        return "decision-log"
    return "session-log"


def collect_session_context(
    *,
    session_id: str,
    messages: List[Dict[str, Any]],
    tool_events: Optional[List[Dict[str, Any]]] = None,
    session_title: str | None = None,
) -> SessionContext:
    filtered_messages = [m for m in messages if m.get("role") not in {"system", "session_meta"}]
    normalized_events = _normalize_tool_events(tool_events or [])
    title = _title_from_messages(filtered_messages, session_title)
    topic_slug = _slugify(session_title or title or session_id, fallback="session")
    key_decisions = _extract_marked_lines(filtered_messages, _DECISION_MARKERS, roles={"assistant", "user"})[:8]
    stable_conclusions = _unique_keep_order([_extract_final_outcome(filtered_messages)] + key_decisions)[:8]
    important_details = _extract_important_details(filtered_messages, normalized_events)
    user_corrections = _extract_marked_lines(filtered_messages, _CORRECTION_MARKERS, roles={"user"})[:8]
    files_touched = _extract_files_touched(normalized_events)
    read_sources = _extract_read_sources(normalized_events)[:12]
    follow_ups = _extract_marked_lines(filtered_messages, _FOLLOW_UP_MARKERS, roles={"assistant", "user"})[:8]
    score, score_reasons = _score_session(filtered_messages, normalized_events)
    doc_type = _choose_doc_type(filtered_messages, normalized_events, key_decisions, files_touched)

    return SessionContext(
        session_id=session_id,
        title=title,
        topic_slug=topic_slug,
        user_goal=_extract_user_goal(filtered_messages),
        final_outcome=_extract_final_outcome(filtered_messages),
        key_decisions=key_decisions,
        stable_conclusions=stable_conclusions,
        important_details=important_details,
        user_corrections=user_corrections,
        files_touched=files_touched,
        tool_events=normalized_events,
        read_sources=read_sources,
        follow_ups=follow_ups,
        transcript_snippets=_extract_snippets(filtered_messages),
        doc_type=doc_type,
        score=score,
        score_reasons=score_reasons,
    )


def _resolve_wiki_root(config: Dict[str, Any]) -> Optional[Path]:
    memory_cfg = (config or {}).get("memory") or {}
    enabled = bool(memory_cfg.get("session_end_auto_write", False))
    mode = str(memory_cfg.get("session_end_write_mode", "off") or "off").strip().lower()
    if not enabled or mode == "off":
        return None
    configured = str(memory_cfg.get("session_end_write_path", "") or "").strip()
    raw_path = configured or os.getenv("WIKI_PATH", "").strip() or str(Path.home() / "wiki")
    return Path(raw_path).expanduser().resolve()


def _ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(2, 1000):
        candidate = parent / f"{stem}-{idx}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}-{datetime.now().strftime('%H%M%S%f')}{suffix}"


def _append_log_entry(wiki_root: Path, doc_path: Path) -> None:
    log_path = wiki_root / "log.md"
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
    else:
        existing = "# Wiki Log\n\n> 结构化操作日志。追加写入。\n> 格式: `## [YYYY-MM-DD] action | subject`\n"
    rel = doc_path.relative_to(wiki_root)
    entry = (
        f"\n## [{datetime.now().strftime('%Y-%m-%d')}] create | session-end raw write\n"
        f"- Created: `{rel.as_posix()}`\n"
    )
    if entry.strip() not in existing:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")


def _render_bullets(items: Iterable[str], fallback: str) -> str:
    values = _unique_keep_order(items)
    if not values:
        return f"- {fallback}"
    return "\n".join(f"- {value}" for value in values)


def _render_frontmatter_list(items: Iterable[str], indent: str = "  ") -> str:
    values = _unique_keep_order(items)
    if not values:
        return f"{indent}- (none)"
    return "\n".join(f"{indent}- {value}" for value in values)


def _render_markdown(
    *,
    context: SessionContext,
    end_reason: str,
    model: str | None,
    platform: str,
) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"""---
title: {context.title}
created: {date_str}
updated: {date_str}
type: {context.doc_type}
status: completed
topic: {context.topic_slug}
tags: [memory, session-end, {platform}]
session_ids: [{context.session_id}]
source_sessions: [{context.session_id}]
related_wiki: []
files_touched:
{_render_frontmatter_list(context.files_touched)}
branch:
---

# Background
- Auto-written at session end from Hermes CLI.
- End reason: {end_reason}
- Model: {model or ''}
- Score: {context.score}
- Score reasons: {', '.join(context.score_reasons) if context.score_reasons else 'hard signal'}

# Session Focus
- {context.user_goal or context.title}

# Key Decisions / Realizations
{_render_bullets(context.key_decisions or context.stable_conclusions, 'No explicit decisions extracted')}

# Important Details
{_render_bullets(context.important_details, 'No important implementation details extracted')}

# Actions / Evidence
{_render_bullets(context.transcript_snippets, 'No retained transcript snippets')}

# Outcome
- {context.final_outcome or 'Session ended without a retained assistant summary.'}

# Follow-up
{_render_bullets(context.follow_ups, 'Review whether this RAW record should later be compiled into Wiki pages.')}

# Evidence Snippets
{_render_bullets([
    f"End reason: `{end_reason}`",
    f"Topic slug: `{context.topic_slug}`",
    *[f"Read source: {src}" for src in context.read_sources[:6]],
    *[f"User correction: {item}" for item in context.user_corrections[:4]],
], 'No additional evidence snippets')}
"""


def maybe_write_session_raw(
    *,
    session_id: str,
    messages: List[Dict[str, Any]],
    tool_events: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
    session_title: str | None = None,
    end_reason: str = "session_end",
    model: str | None = None,
    platform: str = "cli",
) -> Optional[str]:
    wiki_root = _resolve_wiki_root(config or {})
    if wiki_root is None:
        return None

    context = collect_session_context(
        session_id=session_id,
        messages=messages or [],
        tool_events=tool_events or [],
        session_title=session_title,
    )
    if not worth_storing(messages or [], tool_events or []):
        return None

    now = datetime.now()
    if context.doc_type == "task-log":
        out_path = wiki_root / "RAW" / "articles" / "tasks" / f"{context.topic_slug}.md"
    elif context.doc_type == "decision-log":
        out_path = wiki_root / "RAW" / "logs" / f"decision-{now.strftime('%Y-%m-%d')}-{context.topic_slug}.md"
    else:
        out_path = wiki_root / "RAW" / "logs" / f"session-{now.strftime('%Y-%m-%d')}-{context.topic_slug}.md"
    out_path = _ensure_unique_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = _render_markdown(
        context=context,
        end_reason=end_reason,
        model=model,
        platform=platform,
    )
    out_path.write_text(content, encoding="utf-8")
    _append_log_entry(wiki_root, out_path)
    return str(out_path)
