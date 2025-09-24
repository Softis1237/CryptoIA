from __future__ import annotations

"""Подготовка обучающего датасета для моделей индикаторов."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_dataset(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, parse_dates=['snapped_at']).sort_values('snapped_at')
    price = df['price'].astype(float)
    returns = price.pct_change().fillna(0.0)
    trend_feature = price.ewm(span=20, adjust=False).mean() - price.ewm(span=50, adjust=False).mean()
    trend_feature = trend_feature / price.replace(0, np.nan)
    vol_feature = returns.rolling(24).std().bfill()
    news_factor = df['total_volume'].astype(float).fillna(0.0)
    news_factor = (news_factor - news_factor.rolling(30).mean()) / (news_factor.rolling(30).std() + 1e-9)
    news_factor = news_factor.fillna(0.0).clip(-3, 3)

    regimes = []
    for t, v in zip(trend_feature.fillna(0.0), vol_feature.fillna(0.0)):
        if t > 0.002 and v < 0.04:
            regimes.append('trend_up')
        elif t < -0.002 and v < 0.04:
            regimes.append('trend_down')
        elif v >= 0.04:
            regimes.append('choppy_range')
        else:
            regimes.append('range')

    config_map = {
        'trend_up': (10, 12, 18, 2.0),
        'trend_down': (10, 12, 18, 2.0),
        'choppy_range': (16, 20, 24, 2.5),
        'range': (14, 14, 20, 2.2),
    }
    windows = np.array([config_map.get(r, (14, 14, 20, 2.0)) for r in regimes])

    dataset = pd.DataFrame(
        {
            'market_regime': regimes,
            'trend_feature': trend_feature,
            'vol_feature': vol_feature,
            'news_factor': news_factor,
            'window_rsi': windows[:, 0],
            'window_atr': windows[:, 1],
            'bollinger_window': windows[:, 2],
            'bollinger_std': windows[:, 3],
        }
    ).dropna()

    if len(dataset) > 5000:
        dataset = dataset.iloc[::3].reset_index(drop=True)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Подготовка датасета индикаторов из CSV CoinGecko')
    parser.add_argument('--source', default='data/btc-usd-max.csv')
    parser.add_argument('--output', default='data/indicator_training.csv')
    args = parser.parse_args()

    dataset = build_dataset(Path(args.source))
    dataset.to_csv(args.output, index=False)
    print(f'Saved {len(dataset)} rows to {args.output}')


if __name__ == '__main__':
    main()
