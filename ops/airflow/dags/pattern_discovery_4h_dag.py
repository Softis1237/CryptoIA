from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_discovery_4h(**context):
    import sys
    if '/opt/airflow/repo/services/pipeline/src' not in sys.path:
        sys.path.append('/opt/airflow/repo/services/pipeline/src')
    from pipeline.agents.pattern_discovery import DiscoveryInput, run as run_discovery
    payload = DiscoveryInput(
        symbol=os.environ.get('DISC_SYMBOL', 'BTC/USDT'),
        provider=os.environ.get('DISC_PROVIDER', 'binance'),
        timeframe='4h',
        days=int(os.environ.get('DISC_DAYS_4H', '240')),
        move_threshold=float(os.environ.get('DISC_MOVE_THRESHOLD_4H', '0.08')),
        window_hours=int(os.environ.get('DISC_WINDOW_HOURS_4H', '48')),
        lookback_hours_pre=int(os.environ.get('DISC_LOOKBACK_PRE_HOURS_4H', '96')),
        sample_limit=int(os.environ.get('DISC_SAMPLE_LIMIT_4H', '40')),
        dry_run=os.environ.get('DISC_DRY_RUN', '1') in {'1','true','True'},
    )
    res = run_discovery(payload)
    print(res)


default_args = {
    'owner': 'crypto-forecast',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    dag_id='pattern_discovery_weekly_4h',
    default_args=default_args,
    description='Discover new patterns on 4h timeframe',
    schedule_interval='0 4 * * 1',  # weekly, Mon 04:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    run = PythonOperator(
        task_id='run_discovery_4h',
        python_callable=_run_discovery_4h,
        provide_context=True,
    )

