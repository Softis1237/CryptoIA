import os
import sys
import types

# Add src to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pytest  # noqa: E402
from pipeline.trading import publish_telegram as pt  # noqa: E402


@pytest.mark.parametrize(
    "text,limit,expected",
    [
        ("short", 10, ["short"]),
        ("a" * 4095 + "\n" + "b", 4096, ["a" * 4095 + "\n", "b"]),
        ("line1\nline2\nline3", 12, ["line1\nline2\n", "line3"]),
    ],
)
def test_chunk_text_various(text, limit, expected):
    assert pt._chunk_text(text, limit) == expected


def test_dm_user_ids(monkeypatch):
    monkeypatch.delenv("TELEGRAM_DM_USER_IDS", raising=False)
    assert pt._dm_user_ids() == []
    monkeypatch.setenv("TELEGRAM_DM_USER_IDS", "1, 2\n3,")
    assert pt._dm_user_ids() == ["1", "2", "3"]


def test_append_aff_footer(monkeypatch):
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)
    assert pt._append_aff_footer("hello") == "hello"
    monkeypatch.setenv("EXTERNAL_AFF_LINK_URL", "https://ex")
    monkeypatch.setenv("EXTERNAL_AFF_FOOTER_EN", "Visit")
    out = pt._append_aff_footer("hi")
    assert out.endswith("Visit https://ex")
    assert pt._append_aff_footer(out) == out


def _setup_bot(monkeypatch):
    sent: list[tuple[str, str]] = []

    class FakeBot:
        def __init__(self, token: str):
            self.token = token

        def send_message(self, chat_id: str, text: str, **kwargs):
            sent.append((chat_id, text))

    telegram_mod = types.SimpleNamespace(Bot=FakeBot)
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)

    subs_mod = types.ModuleType("pipeline.trading.subscriptions")
    subs_mod.sweep_and_revoke_channel_access = lambda: None
    monkeypatch.setitem(
        sys.modules,
        "pipeline.trading.subscriptions",
        subs_mod,
    )

    return sent


def test_publish_message_channel(monkeypatch):
    sent = _setup_bot(monkeypatch)
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "t")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", "chan")
    monkeypatch.delenv("TELEGRAM_DM_USER_IDS", raising=False)
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)

    pt.publish_message("hi")
    assert sent == [("chan", "hi")]


def test_publish_message_dm_list(monkeypatch):
    sent = _setup_bot(monkeypatch)
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "t")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", None)
    monkeypatch.setenv("TELEGRAM_DM_USER_IDS", "1,2")
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)

    def _no_db():
        raise AssertionError("no db")

    monkeypatch.setattr(pt, "list_active_subscriber_ids", _no_db)

    pt.publish_message("hi")
    assert sent == [("1", "hi"), ("2", "hi")]


def test_publish_message_db_fallback(monkeypatch):
    sent = _setup_bot(monkeypatch)
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "t")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", None)
    monkeypatch.delenv("TELEGRAM_DM_USER_IDS", raising=False)
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)
    monkeypatch.setattr(pt, "list_active_subscriber_ids", lambda: ["3", "4"])

    pt.publish_message("hi")
    assert sent == [("3", "hi"), ("4", "hi")]
