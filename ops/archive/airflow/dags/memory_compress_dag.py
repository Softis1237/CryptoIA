from __future__ import annotations

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_memory_compress(**context):
    if '/opt/airflow/repo/services/pipeline/src' not in sys.path:
        sys.path.append('/opt/airflow/repo/services/pipeline/src')
    from pipeline.agents.memory_compressor import MemoryCompressInput, run as run_mc
    params = context.get('params', {}) or {}
    payload = MemoryCompressInput(n=int(params.get('n', 20)), scope=str(params.get('scope', 'global')))
    res = run_mc(payload)
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
    dag_id='memory_compress_monthly',
    default_args=default_args,
    description='Compress recent run summaries into actionable lessons',
    schedule_interval='0 2 1 * *',  # monthly, day 1 02:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={'n': 50, 'scope': 'global'},
) as dag:
    run = PythonOperator(
        task_id='compress_memory',
        python_callable=_run_memory_compress,
        provide_context=True,
    )

