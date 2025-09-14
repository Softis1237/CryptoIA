import os
import sys
import types

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from pipeline.trading import publish_telegram as pt  # noqa: E402


# Fixture to provide dummy telegram.Bot
@pytest.fixture
def dummy_bot(monkeypatch):
    calls = []

    class DummyBot:
        def __init__(self, token):
            self.token = token

        def send_message(
            self, chat_id, text, parse_mode=None, disable_web_page_preview=None
        ):
            calls.append(chat_id)

    # Inject dummy module into sys.modules
    module = types.SimpleNamespace(Bot=DummyBot)
    monkeypatch.setitem(sys.modules, "telegram", module)
    return calls


def test_chunk_text_splits():
    text = "12345\n6789\n10"
    assert pt._chunk_text(text, limit=10) == ["12345\n", "6789\n10"]


def test_append_aff_footer(monkeypatch):
    monkeypatch.setenv("EXTERNAL_AFF_LINK_URL", "http://ex.com")
    monkeypatch.setenv("EXTERNAL_AFF_FOOTER_EN", "Check")
    base = "Hello"
    out = pt._append_aff_footer(base)
    assert out.endswith("Check http://ex.com")
    # calling twice doesn't duplicate footer
    assert pt._append_aff_footer(out) == out


def test_publish_channel_branch(monkeypatch, dummy_bot):
    # ensure no affiliate footer
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_DM_USER_IDS", "")
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", "chan")
    monkeypatch.setattr(pt, "list_active_subscriber_ids", lambda: [])
    pt.publish_message("hi")
    assert dummy_bot == ["chan"]


def test_publish_dm_branch(monkeypatch, dummy_bot):
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_DM_USER_IDS", "1,2")
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", None)
    monkeypatch.setattr(pt, "list_active_subscriber_ids", lambda: [])
    pt.publish_message("hi")
    assert sorted(dummy_bot) == ["1", "2"]


def test_publish_fallback_branch(monkeypatch, dummy_bot):
    monkeypatch.delenv("EXTERNAL_AFF_LINK_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_DM_USER_IDS", "")
    monkeypatch.setattr(pt.settings, "telegram_bot_token", "tok")
    monkeypatch.setattr(pt.settings, "telegram_chat_id", None)
    monkeypatch.setattr(pt, "list_active_subscriber_ids", lambda: ["3", "4"])
    pt.publish_message("hi")
    assert sorted(dummy_bot) == ["3", "4"]
