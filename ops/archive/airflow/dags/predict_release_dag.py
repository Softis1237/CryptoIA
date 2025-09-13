from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_release_flow(**context):
    # Ensure PYTHONPATH points to mounted repo in Airflow image
    import sys
    if '/opt/airflow/repo/services/pipeline/src' not in sys.path:
        sys.path.append('/opt/airflow/repo/services/pipeline/src')
    from pipeline.orchestration.agent_flow import run_release_flow
    slot = os.environ.get('SLOT', 'scheduled')
    run_release_flow(slot=slot)


default_args = {
    'owner': 'crypto-forecast',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='predict_release_v2',
    default_args=default_args,
    description='Crypto Forecast release flow orchestrated by Airflow -> AgentCoordinator',
    schedule_interval='@hourly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    run = PythonOperator(
        task_id='run_release_flow',
        python_callable=_run_release_flow,
        provide_context=True,
    )

