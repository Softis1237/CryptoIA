from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..trading.backtest import BacktestConfig, run_from_dataframe


def _parse_strategy_params(values: list[str] | None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not values:
        return params
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError("Параметры стратегии должны быть в формате key=value")
        key, value = item.split("=", 1)
        params[key] = _coerce_value(value)
    return params


def _coerce_value(raw: str) -> Any:
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:  # noqa: PERF203 - полезное сообщение
            raise argparse.ArgumentTypeError(f"Не удалось распарсить JSON: {exc}") from exc
    return raw


def _load_prices(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Векторный бэктест торгового конвейера CryptoIA")
    parser.add_argument("price_path", type=Path, help="Путь к CSV или Parquet файлу с OHLCV")
    parser.add_argument("--symbol", default="BTC/USDT", help="Торговый символ")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--taker-fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--slippage-jitter-bps", type=float, default=0.0, help="Случайное проскальзывание (bps, ±) для элементов рынка")
    parser.add_argument("--volume-limit", type=float, default=0.25)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        default="pipeline.trading.backtest.strategies:MovingAverageCrossStrategy",
        help="Путь к стратегии вида module:Class",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        help="Дополнительные параметры стратегии в формате key=value",
    )
    parser.add_argument("--report", type=Path, help="Путь для сохранения JSON отчёта", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = _load_prices(args.price_path)
    if args.start:
        df = df[pd.to_datetime(df["timestamp"]) >= pd.to_datetime(args.start)]
    if args.end:
        df = df[pd.to_datetime(df["timestamp"]) <= pd.to_datetime(args.end)]
    if df.empty:
        raise SystemExit("Датасет пустой после фильтрации по датам")

    config = BacktestConfig(
        symbol=args.symbol,
        starting_cash=args.starting_cash,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        slippage=args.slippage,
        volume_limit=args.volume_limit,
        risk_free_rate=args.risk_free_rate,
        slippage_jitter_bps=args.slippage_jitter_bps,
    )

    params = _parse_strategy_params(args.strategy_param)
    report = run_from_dataframe(df, config=config, strategy_path=args.strategy, strategy_params=params)

    print("\n== Метрики ==")
    for key, value in report.metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Сделок: {len(report.trades)}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        print(f"Отчёт сохранён: {args.report}")


if __name__ == "__main__":  # pragma: no cover
    main()
