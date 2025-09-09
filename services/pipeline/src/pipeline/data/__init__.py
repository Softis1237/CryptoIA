"""Data ingestion modules."""

from .ingest_prices import IngestPricesInput, IngestPricesOutput, run as ingest_prices
from .ingest_news import IngestNewsInput, IngestNewsOutput, run as ingest_news
from .ingest_orderbook import (
    IngestOrderbookInput,
    IngestOrderbookOutput,
    run as ingest_orderbook,
)
from .ingest_onchain import IngestOnchainInput, IngestOnchainOutput, run as ingest_onchain
from .ingest_prices_lowtf import (
    IngestPricesLowTFInput,
    IngestPricesLowTFOutput,
    run as ingest_prices_lowtf,
)
from .ingest_futures import IngestFuturesInput, IngestFuturesOutput, run as ingest_futures
from .ingest_social import IngestSocialInput, IngestSocialOutput, run as ingest_social

__all__ = [
    "IngestPricesInput",
    "IngestPricesOutput",
    "ingest_prices",
    "IngestNewsInput",
    "IngestNewsOutput",
    "ingest_news",
    "IngestOrderbookInput",
    "IngestOrderbookOutput",
    "ingest_orderbook",
    "IngestOnchainInput",
    "IngestOnchainOutput",
    "ingest_onchain",
    "IngestPricesLowTFInput",
    "IngestPricesLowTFOutput",
    "ingest_prices_lowtf",
    "IngestFuturesInput",
    "IngestFuturesOutput",
    "ingest_futures",
    "IngestSocialInput",
    "IngestSocialOutput",
    "ingest_social",
]
