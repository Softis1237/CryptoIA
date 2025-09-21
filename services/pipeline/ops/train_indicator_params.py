from __future__ import annotations

"""Скрипт обучения моделей подбора параметров индикаторов.

Ожидается CSV/Parquet с колонками:
- market_regime — строка (например, trend_up, trend_down, range, choppy_range)
- feature_* — числовые признаки (например, тренды/волатильность)
- window_rsi, window_atr, bollinger_window, bollinger_std — целевые значения

Пример запуска:
  python -m pipeline.ops.train_indicator_params \
      --input data/indicator_training.csv \
      --output artifacts/indicator_params

Для конвертации в ONNX используются scikit-learn и skl2onnx.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def _lazy_import():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Не удалось импортировать зависимости (scikit-learn, numpy, pandas, skl2onnx). "
            "Установите их в окружение прежде чем запускать скрипт."
        ) from exc
    return np, pd, RandomForestRegressor, MultiOutputRegressor, convert_sklearn, FloatTensorType


def _load_frame(path: Path, selected_cols: Iterable[str]):
    _, pd, *_ = _lazy_import()
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {', '.join(missing)}")
    return df


def _train_for_regime(df, regime: str, feature_cols: List[str], out_dir: Path) -> None:
    (
        _,
        pd,
        RandomForestRegressor,
        MultiOutputRegressor,
        convert_sklearn,
        FloatTensorType,
    ) = _lazy_import()
    subset = df[df["market_regime"] == regime]
    if subset.empty:
        return
    if len(subset) < 50:
        print(f"[train_indicator_params] regime={regime}: недостаточно примеров ({len(subset)}) — пропуск")
        return
    X = subset[feature_cols].astype(float).values
    y = subset[["window_rsi", "window_atr", "bollinger_window", "bollinger_std"]].astype(float).values
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(X, y)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, len(feature_cols)]))],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{regime}.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[train_indicator_params] regime={regime} → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение моделей динамических индикаторов")
    parser.add_argument("--input", required=True, help="Путь к CSV/Parquet с обучающей выборкой")
    parser.add_argument("--output", required=True, help="Каталог для сохранения ONNX моделей")
    parser.add_argument(
        "--features",
        nargs="*",
        default=["trend_feature", "vol_feature", "news_factor"],
        help="Список колонок признаков, используемых в моделях",
    )
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=["trend_up", "trend_down", "range", "choppy_range"],
        help="Список режимов рынка",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл {input_path}")

    target_cols = [
        "market_regime",
        "window_rsi",
        "window_atr",
        "bollinger_window",
        "bollinger_std",
    ] + list(args.features)
    df = _load_frame(input_path, target_cols)

    out_dir = Path(args.output)
    for regime in args.regimes:
        _train_for_regime(df, regime, list(args.features), out_dir)

    print("Готово. Модели сохранены в", out_dir)


if __name__ == "__main__":
    main()
