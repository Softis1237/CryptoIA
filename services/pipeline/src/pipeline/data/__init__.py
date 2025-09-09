"""Data ingestion modules."""

 codex/create-ingest_prices.py-and-models
from .ingest_news import IngestNewsInput, IngestNewsOutput, run

__all__ = ["IngestNewsInput", "IngestNewsOutput", "run"]

codex/create-ingest_news.py-and-data-models
from .ingest_news import IngestNewsInput, IngestNewsOutput, run

from .ingest_news import IngestNewsInput, IngestNewsOutput
 codex/create-ingest_onchain.py-for-metrics
from .ingest_news import run as run_news
from .ingest_onchain import IngestOnchainInput, IngestOnchainOutput
from .ingest_onchain import run as run_onchain

from .ingest_news import run as ingest_news
from .ingest_onchain import IngestOnchainInput, IngestOnchainOutput
from .ingest_onchain import run as ingest_onchain
from .ingest_orderbook import IngestOrderbookInput, IngestOrderbookOutput
from .ingest_orderbook import run as ingest_orderbook
from .ingest_prices import IngestPricesInput, IngestPricesOutput
from .ingest_prices import run as ingest_prices
from .ingest_prices_lowtf import (IngestPricesLowTFInput,
                                  IngestPricesLowTFOutput)
from .ingest_prices_lowtf import run as ingest_prices_lowtf
main
main

__all__ = [
    "IngestNewsInput",
    "IngestNewsOutput",
 codex/create-ingest_onchain.py-for-metrics
    "run_news",
    "IngestOnchainInput",
    "IngestOnchainOutput",
    "run_onchain",

    "ingest_news",
    "IngestOnchainInput",
    "IngestOnchainOutput",
    "ingest_onchain",
    "IngestOrderbookInput",
    "IngestOrderbookOutput",
    "ingest_orderbook",
    "IngestPricesInput",
    "IngestPricesOutput",
    "ingest_prices",
    "IngestPricesLowTFInput",
    "IngestPricesLowTFOutput",
    "ingest_prices_lowtf",
 main
]
 main
