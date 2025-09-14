from __future__ import annotations

from typing import Any, Dict


def latest_supply_demand_metrics(meta: Dict[str, Any]) -> Dict[str, float]:
    """Compute supply/demand related metrics from provided market data.

    Parameters
    ----------
    meta : dict
        Expected structure:
        {
            "orderbook": {"bid_vol": float, "ask_vol": float},
            "derivatives": {"open_interest": float, "funding_rate": float},
            "onchain": {"long_term_holders": float, "short_term_holders": float},
        }

    Returns
    -------
    dict
        {
            "orderbook_imbalance": float,
            "open_interest": float,
            "funding_rate": float,
            "long_term_holder_ratio": float,
            "short_term_holder_ratio": float,
        }
    """
    orderbook = meta.get("orderbook", {}) or {}
    derivatives = meta.get("derivatives", {}) or {}
    onchain = meta.get("onchain", {}) or {}

    bid = float(orderbook.get("bid_vol", 0.0) or 0.0)
    ask = float(orderbook.get("ask_vol", 0.0) or 0.0)
    denom = bid + ask
    ob_imbalance = (bid - ask) / denom if denom else 0.0

    open_interest = float(derivatives.get("open_interest", 0.0) or 0.0)
    funding_rate = float(derivatives.get("funding_rate", 0.0) or 0.0)

    lt = float(onchain.get("long_term_holders", 0.0) or 0.0)
    st = float(onchain.get("short_term_holders", 0.0) or 0.0)
    total = lt + st
    lt_ratio = lt / total if total else 0.0
    st_ratio = st / total if total else 0.0

    return {
        "orderbook_imbalance": ob_imbalance,
        "open_interest": open_interest,
        "funding_rate": funding_rate,
        "long_term_holder_ratio": lt_ratio,
        "short_term_holder_ratio": st_ratio,
    }
