"""DAG que prepara o ambiente e gera o modelo baseline."""

from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="generate_baseline_model",
    description="Executa o script de preparo e o generate_baseline_model.py",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["credit-monitoring", "baseline"],
) as dag:
    # O espaço final evita que o Airflow trate o .sh como template Jinja.
    prepare = BashOperator(
        task_id="prepare_environment",
        bash_command="bash /opt/airflow/dags/scripts/prepare_baseline.sh ",
    )

    generate = BashOperator(
        task_id="generate_baseline_model",
        bash_command=(
            "cd /app && "
            "${CREDIT_PYTHON:-/opt/venv/bin/python} "
            "-m src.credit_monitoring.generate_baseline_model"
        ),
    )

    prepare >> generate
