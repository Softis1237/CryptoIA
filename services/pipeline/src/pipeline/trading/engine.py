from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from loguru import logger

from .exchange import ExchangeClient


@dataclass
class BracketOrder:
    entry_id: Optional[str]
    sl_id: Optional[str]
    tp_id: Optional[str]


class TradingEngine:
    def __init__(self) -> None:
        self.client = ExchangeClient()
        self.dry = os.getenv("DRY_RUN", "1") in {"1", "true", "True"}

    def open_bracket(self, symbol: str, side: str, amount: float, entry_type: str = "market", price: Optional[float] = None, sl_price: Optional[float] = None, tp_price: Optional[float] = None, leverage: Optional[int] = None) -> BracketOrder:
        if leverage and leverage > 1:
            try:
                self.client.set_leverage(symbol, int(leverage))
            except Exception:
                pass
        if self.dry:
            logger.info(f"DRY open {side} {amount} {symbol} at {price} with SL={sl_price} TP={tp_price}")
            return BracketOrder(entry_id=None, sl_id=None, tp_id=None)
        # Entry
        if entry_type == "limit" and price is not None:
            res = self.client.create_limit(symbol, side, amount, price)
        else:
            res = self.client.create_market(symbol, side, amount)
        sl_id = None
        tp_id = None
        opp = "sell" if side.lower() == "buy" or side.upper() == "LONG" else "buy"
        # Protective SL (reduce only)
        if sl_price is not None:
            try:
                sl = self.client.create_stop(symbol, opp, amount, sl_price, reduce_only=True)
                sl_id = sl.id
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to place SL: {e}")
        # Take profit as limit reduce-only
        if tp_price is not None:
            try:
                tp = self.client.create_limit(symbol, opp, amount, tp_price, params={"reduceOnly": True})
                tp_id = tp.id
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to place TP: {e}")
        return BracketOrder(entry_id=res.id, sl_id=sl_id, tp_id=tp_id)

    def close_position(self, symbol: str, amount: float, side: str) -> None:
        if self.dry:
            logger.info(f"DRY close {symbol} {amount} by {side}")
            return
        self.client.create_market(symbol, side, amount, params={"reduceOnly": True})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", default=os.getenv("EXEC_SYMBOL", "BTC/USDT"))
    ap.add_argument("side", choices=["buy", "sell", "LONG", "SHORT"], help="Entry side (buy/sell or LONG/SHORT)")
    ap.add_argument("amount", type=float)
    ap.add_argument("--entry", choices=["market", "limit"], default="market")
    ap.add_argument("--price", type=float)
    ap.add_argument("--sl", type=float)
    ap.add_argument("--tp", type=float)
    ap.add_argument("--leverage", type=int)
    args = ap.parse_args()

    eng = TradingEngine()
    bo = eng.open_bracket(args.symbol, args.side, args.amount, entry_type=args.entry, price=args.price, sl_price=args.sl, tp_price=args.tp, leverage=args.leverage)
    print({"entry_id": bo.entry_id, "sl_id": bo.sl_id, "tp_id": bo.tp_id})


if __name__ == "__main__":
    main()

