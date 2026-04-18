#!/usr/bin/env python3
"""Task tracking tool for Obsidian notes."""

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from tools.registry import registry

OBSIDIAN_AGENT_ROOT = Path("~/Documents/ObsidianNotes/计算机/Agent/").expanduser()
_VALID_PREFIXES = ("research-", "task-", "feat-", "fix-", "refactor-", "chore-")


def _today() -> str:
    return date.today().isoformat()


def _ok(message: str, **extra: Any) -> str:
    payload: Dict[str, Any] = {"success": True, "message": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _err(message: str, **extra: Any) -> str:
    payload: Dict[str, Any] = {"success": False, "error": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    raw = content[4:end]
    body = content[end + 5 :]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data, body


def _render_frontmatter(data: dict[str, str]) -> str:
    ordered_keys = ["title", "topic", "tags", "created", "updated", "type", "summary", "status"]
    lines: List[str] = []
    used = set()
    for key in ordered_keys:
        if key in data:
            lines.append(f"{key}: {data[key]}")
            used.add(key)
    for key, value in data.items():
        if key not in used:
            lines.append(f"{key}: {value}")
    return "---\n" + "\n".join(lines) + "\n---\n"


def _replace_or_append_section(body: str, heading: str, section_content: str) -> str:
    pattern = re.compile(rf"(?ms)^##\s+{re.escape(heading)}\n.*?(?=^##\s+|\Z)")
    replacement = f"## {heading}\n{section_content.strip()}\n\n"
    if pattern.search(body):
        return pattern.sub(replacement, body, count=1)
    body = body.rstrip() + "\n\n" if body.strip() else ""
    return body + replacement


def _append_progress(body: str, progress: str) -> str:
    today = _today()
    pattern = re.compile(r"(?ms)^##\s+进展\n(.*?)(?=^##\s+|\Z)")
    m = pattern.search(body)
    entry = f"- {today}: {progress.strip()}"
    if m:
        existing = m.group(1).rstrip()
        new_section = f"## 进展\n{existing}\n{entry}\n\n" if existing else f"## 进展\n{entry}\n\n"
        return body[: m.start()] + new_section + body[m.end() :]
    body = body.rstrip() + "\n\n" if body.strip() else ""
    return body + f"## 进展\n{entry}\n\n"


def find_task_note(topic: str, keywords: List[str] | None = None, task_id: str | None = None) -> str:
    del task_id
    keywords = [k.strip() for k in (keywords or []) if str(k).strip()]
    topic_lower = topic.lower().strip()
    scored: list[tuple[int, Path, str]] = []

    if not OBSIDIAN_AGENT_ROOT.exists():
        return _err(f"Task tracking root not found: {OBSIDIAN_AGENT_ROOT}")

    for path in OBSIDIAN_AGENT_ROOT.rglob("*.md"):
        if not path.name.startswith(_VALID_PREFIXES):
            continue
        try:
            text = _read_text(path)
        except Exception:
            continue

        frontmatter, _body = _parse_frontmatter(text)
        haystacks = [
            path.name.lower(),
            frontmatter.get("topic", "").lower(),
            frontmatter.get("summary", "").lower(),
            text[:400].lower(),
        ]
        score = 0
        if any(topic_lower and topic_lower in h for h in haystacks):
            score += 10
        for kw in keywords:
            kw_lower = kw.lower()
            if any(kw_lower in h for h in haystacks):
                score += 3
        if score <= 0:
            continue
        preview = "\n".join(text.splitlines()[:20])
        scored.append((score, path, preview))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    candidates = [
        {"path": str(path), "filename": path.name, "score": score, "preview": preview}
        for score, path, preview in scored[:5]
    ]
    return _ok(f"Found {len(candidates)} potential candidates.", candidates=candidates)


def update_task_note(path: str, progress: str, summary: str | None = None, status: str | None = None, task_id: str | None = None) -> str:
    del task_id
    note_path = Path(path).expanduser()
    if not note_path.exists():
        return _err(f"Task note not found: {note_path}")
    try:
        content = _read_text(note_path)
        frontmatter, body = _parse_frontmatter(content)
        if not frontmatter:
            return _err(f"Task note missing frontmatter: {note_path}")

        frontmatter["updated"] = _today()
        if summary:
            frontmatter["summary"] = summary
        if status:
            frontmatter["status"] = status

        body = _append_progress(body, progress)
        new_content = _render_frontmatter(frontmatter) + "\n" + body.lstrip("\n")
        _write_text(note_path, new_content)
        return _ok(f"Updated task note {note_path.name}: {progress[:40]}", path=str(note_path))
    except Exception as e:
        return _err(str(e), path=str(note_path))


def create_task_note(
    path: str,
    title: str,
    topic: str,
    summary: str,
    background: str,
    todo: str,
    status: str = "active",
    task_id: str | None = None,
) -> str:
    del task_id
    note_path = Path(path).expanduser()
    today = _today()
    frontmatter = {
        "title": title,
        "topic": topic,
        "tags": "[task-tracking]",
        "created": today,
        "updated": today,
        "type": "task",
        "summary": summary,
        "status": status,
    }
    body = (
        f"## 背景\n{background.strip()}\n\n"
        f"## 进展\n- {today}: Note created\n\n"
        "## 关键发现\n\n"
        f"## 待办\n{todo.strip()}\n"
    )
    try:
        _write_text(note_path, _render_frontmatter(frontmatter) + "\n" + body)
        return _ok(f"Created task note {note_path.name}", path=str(note_path))
    except Exception as e:
        return _err(str(e), path=str(note_path))


def task_tracking_manage(action: str, **kwargs: Any) -> str:
    if action == "find":
        topic = kwargs.get("topic")
        if not topic:
            return _err("topic is required for 'find'")
        return find_task_note(topic=topic, keywords=kwargs.get("keywords") or [], task_id=kwargs.get("task_id"))

    if action == "update":
        path = kwargs.get("path")
        progress = kwargs.get("progress")
        if not path or not progress:
            return _err("path and progress are required for 'update'")
        return update_task_note(
            path=path,
            progress=progress,
            summary=kwargs.get("summary"),
            status=kwargs.get("status"),
            task_id=kwargs.get("task_id"),
        )

    if action == "create":
        required = ["path", "title", "topic", "summary", "background", "todo"]
        missing = [key for key in required if not kwargs.get(key)]
        if missing:
            return _err(f"Missing required fields for 'create': {', '.join(missing)}")
        return create_task_note(
            path=kwargs["path"],
            title=kwargs["title"],
            topic=kwargs["topic"],
            summary=kwargs["summary"],
            background=kwargs["background"],
            todo=kwargs["todo"],
            status=kwargs.get("status", "active"),
            task_id=kwargs.get("task_id"),
        )

    return _err(f"Unknown action '{action}'. Use: find, update, create")


registry.register(
    name="task_tracking_manage",
    toolset="task_tracking",
    schema={
        "name": "task_tracking_manage",
        "description": (
            "Manage Obsidian task tracking notes in ~/Documents/ObsidianNotes/计算机/Agent/. "
            "Actions: find existing notes, update an existing note, or create a new note."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["find", "update", "create"]},
                "topic": {"type": "string", "description": "Stable task slug/name for search or creation."},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "High-signal search keywords."},
                "path": {"type": "string", "description": "Absolute path to the task note."},
                "progress": {"type": "string", "description": "Progress entry to append to the note."},
                "summary": {"type": "string", "description": "Updated one-line summary."},
                "status": {"type": "string", "description": "Task status, e.g. active or completed."},
                "title": {"type": "string", "description": "Human-readable task title."},
                "background": {"type": "string", "description": "Task background/context."},
                "todo": {"type": "string", "description": "Markdown todo list body."},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: task_tracking_manage(**args),
)
