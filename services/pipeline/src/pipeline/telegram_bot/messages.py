from __future__ import annotations

import gettext
import os
from pathlib import Path

_LOCALE_DIR = Path(__file__).resolve().parent / "locale"
_DEFAULT_LANG = "en"


def _get_translator() -> gettext.NullTranslations:
    lang = os.getenv("BOT_LOCALE", _DEFAULT_LANG)
    return gettext.translation(
        "messages", localedir=_LOCALE_DIR, languages=[lang], fallback=True
    )


_translator = _get_translator()
_ = _translator.gettext

TEMPLATES: dict[str, str] = {
    "paper_account_not_found": "Paper: account not found",
    "paper_no_equity": "No equity data for the week",
    "paper_equity_week": "Paper: Equity {days}d. Start={start_eq}",
    "btc_forecast_caption": "BTC {slot}: 4h/12h forecast",
    "btc_risk_caption": "BTC {slot}: risk charts",
}


def get_message(key: str, **kwargs: object) -> str:
    """Return localized message by key with optional formatting."""
    template = TEMPLATES.get(key, key)
    return _(template).format(**kwargs)
