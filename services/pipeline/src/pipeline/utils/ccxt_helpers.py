from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Sequence

import ccxt  # type: ignore
from loguru import logger


def _normalize_providers(providers: Sequence[str] | str | None) -> list[str]:
    if providers is None:
        return []
    if isinstance(providers, str):
        return [providers]
    return [str(p) for p in providers if str(p)]


def _create_exchange(provider: str, timeout_ms: int | None = None):
    params: dict[str, Any] = {"enableRateLimit": True}
    if timeout_ms is not None:
        params["timeout"] = int(timeout_ms)
    return getattr(ccxt, provider)(params)


def _call_with_retries(
    providers: Sequence[str] | str,
    func: Callable[[Any], Any],
    *,
    timeout_ms: int | None = None,
    retries: int = 3,
    backoff: float = 1.5,
    context: str = "ccxt",
) -> Any:
    prov_list = _normalize_providers(providers) or []
    if not prov_list:
        raise ValueError(f"{context}: no providers configured")
    last_exc: Exception | None = None
    for provider in prov_list:
        try:
            exchange = _create_exchange(provider, timeout_ms=timeout_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{context}: failed to init exchange '{provider}': {exc}")
            last_exc = exc
            continue
        try:
            attempt = 0
            while attempt < max(1, retries):
                try:
                    return func(exchange)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    attempt += 1
                    logger.debug(
                        f"{context}: provider={provider} attempt={attempt} failed: {exc}"
                    )
                    if attempt < retries:
                        sleep_for = backoff ** (attempt - 1) if backoff > 0 else 0.0
                        if sleep_for:
                            time.sleep(min(5.0, sleep_for))
            logger.warning(f"{context}: provider {provider} exhausted retries")
        finally:
            try:
                exchange.close()
            except Exception:  # noqa: BLE001
                pass
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{context}: providers {prov_list} unavailable")


def fetch_ohlcv(
    providers: Sequence[str] | str,
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    since_ms: int | None = None,
    timeout_ms: int | None = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> list[list[float]]:
    def _call(exchange):
        return exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=max(10, min(1000, int(limit))),
        )

    data = _call_with_retries(
        providers,
        _call,
        timeout_ms=timeout_ms,
        retries=retries,
        backoff=backoff,
        context=f"fetch_ohlcv:{symbol}:{timeframe}",
    )
    return data or []


def fetch_order_book(
    providers: Sequence[str] | str,
    symbol: str,
    *,
    limit: int | None = None,
    timeout_ms: int | None = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> dict[str, Any]:
    def _call(exchange):
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        return exchange.fetch_order_book(symbol, **kwargs)

    data = _call_with_retries(
        providers,
        _call,
        timeout_ms=timeout_ms,
        retries=retries,
        backoff=backoff,
        context=f"fetch_order_book:{symbol}",
    )
    return data or {}


def fetch_trades(
    providers: Sequence[str] | str,
    symbol: str,
    *,
    since_ms: int | None = None,
    limit: int = 100,
    timeout_ms: int | None = None,
    retries: int = 3,
    backoff: float = 1.5,
    params: dict[str, Any] | None = None,
) -> list[dict]:
    def _call(exchange):
        return exchange.fetch_trades(symbol, since=since_ms, limit=limit, params=params or {})

    data = _call_with_retries(
        providers,
        _call,
        timeout_ms=timeout_ms,
        retries=retries,
        backoff=backoff,
        context=f"fetch_trades:{symbol}",
    )
    return data or []


__all__ = [
    "fetch_ohlcv",
    "fetch_order_book",
    "fetch_trades",
]
