from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import List, Optional

import matplotlib

# Headless backend for servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ..infra.s3 import download_bytes, upload_bytes
ёimport os
import numpy as np


def _overlay_brand(fig, ax, title: str | None = None) -> None:
    """Overlay simple watermark and optional logo from S3.

    Env:
      - LOGO_S3: S3 URI to logo (PNG with transparent bg recommended)
      - BRAND_WATERMARK: text watermark (defaults to BRAND_NAME or 'CryptoIA')
    """
    try:
        wm = os.getenv("BRAND_WATERMARK") or os.getenv("BRAND_NAME") or "CryptoIA"
        ax.text(
            0.99,
            0.02,
            wm,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#888888",
            alpha=0.8,
        )
        logo_s3 = os.getenv("LOGO_S3")
        if logo_s3:
            try:
                raw = download_bytes(logo_s3)
                import PIL.Image as Image  # lazy

                img = Image.open(io.BytesIO(raw))
                # Add small axes for the logo at top-left
                ax_logo = fig.add_axes([0.01, 0.82, 0.12, 0.12], anchor="NW")
                ax_logo.imshow(img)
                ax_logo.axis("off")
            except Exception:
                pass
        if title:
            ax.set_title(title)
    except Exception:
        pass


def plot_price_with_levels(
    features_path_s3: str,
    title: str,
    y_hat_4h: Optional[float] = None,
    y_hat_12h: Optional[float] = None,
    levels: Optional[List[float]] = None,
    slot: str = "manual",
) -> str:
    raw = download_bytes(features_path_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas()
    df = df.sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("UTC")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["dt"], df["close"], label="Close", color="#1f77b4", linewidth=1.2)

    if levels:
        for lv in levels:
            ax.axhline(lv, color="#cccccc", linestyle="--", linewidth=0.8)

    if y_hat_4h is not None:
        ax.axhline(y_hat_4h, color="#2ca02c", linestyle=":", linewidth=1.2, label="4h forecast")
    if y_hat_12h is not None:
        ax.axhline(y_hat_12h, color="#ff7f0e", linestyle=":", linewidth=1.2, label="12h forecast")

    # Title & watermark/branding
    _overlay_brand(fig, ax, title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = f"runs/{date_key}/{slot}/chart.png"
    s3_uri = upload_bytes(path, buf.getvalue(), content_type="image/png")
    return s3_uri


def plot_risk_breakdown(
    features_path_s3: str,
    slot: str = "manual",
    title: str = "Risk breakdown",
    var95: float | None = None,
    es95: float | None = None,
) -> str:
    """Render a two-panel chart: returns histogram with VaR/ES and drawdown curve.

    Returns S3 URI of the saved image.
    """
    raw = download_bytes(features_path_s3)
    table = pq.read_table(pa.BufferReader(raw))
    df = table.to_pandas().sort_values("ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    close = df["close"].astype(float)
    ret = close.pct_change().dropna()
    if ret.empty:
        # fallback to price chart
        return plot_price_with_levels(features_path_s3, title=title, slot=slot)

    # Drawdown
    wealth = (1 + ret).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes
    # Histogram
    ax1.hist(ret.values, bins=50, color="#1f77b4", alpha=0.7)
    ax1.set_title("Returns distribution")
    if var95 is not None:
        ax1.axvline(-float(var95), color="#d62728", linestyle="--", label=f"VaR95={var95:.3f}")
    if es95 is not None:
        ax1.axvline(-float(es95), color="#ff7f0e", linestyle=":", label=f"ES95={es95:.3f}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    # Drawdown curve
    ax2.plot(df["dt"].iloc[-len(dd):], dd.values, color="#2ca02c", linewidth=1.2)
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.2)
    # branding on first axes
    _overlay_brand(fig, ax1, title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = f"runs/{date_key}/{slot}/risk.png"
    s3_uri = upload_bytes(path, buf.getvalue(), content_type="image/png")
    return s3_uri
