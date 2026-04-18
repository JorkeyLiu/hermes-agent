from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from agent.session_end_writer import collect_session_context, worth_storing


class _FakeAgent:
    def __init__(self, session_id: str, session_start):
        self.session_id = session_id
        self.session_start = session_start
        self.model = "gpt-5"
        self.platform = "cli"
        self.flush_memories = MagicMock()
        self.commit_memory_session = MagicMock()


def _make_cli(config_overrides=None, env_overrides=None, **kwargs):
    _clean_config = {
        "model": {
            "default": "gpt-5",
            "base_url": "https://example.invalid/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "provider": "",
            "session_end_auto_write": True,
            "session_end_write_mode": "raw-only",
            "session_end_write_path": "",
        },
    }
    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict) and isinstance(_clean_config.get(key), dict):
                _clean_config[key].update(value)
            else:
                _clean_config[key] = value

    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)

    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }

    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI(**kwargs)


def _prepare_cli(tmp_path):
    wiki_root = tmp_path / "wiki"
    (wiki_root / "RAW" / "logs").mkdir(parents=True, exist_ok=True)
    (wiki_root / "RAW" / "articles" / "tasks").mkdir(parents=True, exist_ok=True)
    (wiki_root / "log.md").write_text("# Wiki Log\n", encoding="utf-8")

    cli = _make_cli(
        config_overrides={
            "memory": {
                "session_end_auto_write": True,
                "session_end_write_mode": "raw-only",
                "session_end_write_path": str(wiki_root),
            }
        },
        env_overrides={"WIKI_PATH": str(wiki_root)},
    )
    cli._session_db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [
        {"role": "user", "content": "Please implement the session-end raw writer and keep the config path stable."},
        {
            "role": "assistant",
            "content": "Decision: we should add a CLI finalize hook and keep config under memory.session_end_write_path.",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok", "tool_name": "write_file"},
        {"role": "user", "content": "不是改成摘要页，重点是保留 RAW 细节和 follow-up。"},
        {
            "role": "assistant",
            "content": "Done. The hook now writes a RAW session log, captures config/path details, and leaves follow-up for wiki compilation.",
        },
    ]
    cli._session_tool_events = [
        {"event_type": "tool.started", "name": "write_file", "args": {"path": "/tmp/example.py"}},
        {"event_type": "tool.completed", "name": "write_file", "args": {"path": "/tmp/example.py"}},
        {"event_type": "tool.started", "name": "terminal", "args": {"command": "python -m pytest"}},
        {"event_type": "tool.started", "name": "read_file", "args": {"path": "/tmp/spec.md"}},
    ]
    return cli, wiki_root


def test_worth_storing_false_for_trivial_conversation():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert worth_storing(messages, []) is False


def test_collect_session_context_extracts_decisions_details_and_corrections():
    messages = [
        {"role": "user", "content": "Please change config path handling."},
        {"role": "assistant", "content": "Decision: adopt memory.session_end_write_path and keep WIKI_PATH fallback."},
        {"role": "user", "content": "不是摘要系统，重点是保留 RAW 细节。"},
        {"role": "assistant", "content": "Follow-up: add wiki compilation later after raw-only stabilizes."},
    ]
    tool_events = [
        {"event_type": "tool.started", "name": "write_file", "args": {"path": "/tmp/example.py"}},
        {"event_type": "tool.started", "name": "read_file", "args": {"path": "/tmp/spec.md"}},
    ]

    ctx = collect_session_context(
        session_id="s1",
        messages=messages,
        tool_events=tool_events,
        session_title="Session End Writer",
    )

    assert ctx.topic_slug == "session-end-writer"
    assert ctx.doc_type == "task-log"
    assert any("Decision:" in item for item in ctx.key_decisions)
    assert any("/tmp/example.py" in item for item in ctx.important_details)
    assert any("不是摘要系统" in item for item in ctx.user_corrections)
    assert any("Follow-up" in item for item in ctx.follow_ups)
    assert any("/tmp/spec.md" in item for item in ctx.read_sources)
    assert ctx.score >= 3


def test_finalize_current_session_writes_raw_log(tmp_path):
    cli, wiki_root = _prepare_cli(tmp_path)

    cli._finalize_current_session("cli_close", notify_plugin=False)

    written = list((wiki_root / "RAW" / "articles" / "tasks").glob("*.md"))
    if not written:
        written = list((wiki_root / "RAW" / "logs").glob("*.md"))
    assert written, "expected a RAW session file to be written"
    content = written[0].read_text(encoding="utf-8")
    assert "config/path details" in content or "config under memory.session_end_write_path" in content
    assert "cli_close" in content
    assert "write_file" in content
    assert "不是改成摘要页" in content
    assert cli._session_tool_events == []


def test_finalize_current_session_skips_when_disabled(tmp_path):
    wiki_root = tmp_path / "wiki"
    (wiki_root / "RAW" / "logs").mkdir(parents=True, exist_ok=True)
    (wiki_root / "RAW" / "articles" / "tasks").mkdir(parents=True, exist_ok=True)
    (wiki_root / "log.md").write_text("# Wiki Log\n", encoding="utf-8")

    cli = _make_cli(
        config_overrides={
            "memory": {
                "session_end_auto_write": False,
                "session_end_write_mode": "off",
            }
        },
        env_overrides={"WIKI_PATH": str(wiki_root)},
    )
    cli.agent = _FakeAgent(cli.session_id, cli.session_start)
    cli.conversation_history = [{"role": "user", "content": "hi"}]
    cli._session_tool_events = [{"event_type": "tool.started", "name": "write_file", "args": {"path": "/tmp/x.py"}}]

    cli._finalize_current_session("cli_close", notify_plugin=False)

    assert list((wiki_root / "RAW" / "logs").glob("*.md")) == []
    assert list((wiki_root / "RAW" / "articles" / "tasks").glob("*.md")) == []
