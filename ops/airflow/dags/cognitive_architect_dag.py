from __future__ import annotations

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_cognitive(**context):
    if '/opt/airflow/repo/services/pipeline/src' not in sys.path:
        sys.path.append('/opt/airflow/repo/services/pipeline/src')
    from pipeline.agents.cognitive_architect import CognitiveArchitectInput, run as run_cog
    params = context.get('params', {}) or {}
    payload = CognitiveArchitectInput(
        analyze_n=int(params.get('analyze_n', 50)),
        target_agents=params.get('target_agents') or ["ChartReasoningAgent","DebateArbiter"],
    )
    res = run_cog(payload)
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
    dag_id='cognitive_architect_monthly',
    default_args=default_args,
    description='Meta-agent to improve prompts/configs based on outcomes and summaries',
    schedule_interval='0 4 1 * *',  # monthly, day 1 04:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    params={'analyze_n': 50},
) as dag:
    run = PythonOperator(
        task_id='run_cognitive',
        python_callable=_run_cognitive,
        provide_context=True,
    )

