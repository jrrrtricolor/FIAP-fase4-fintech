#!/usr/bin/env bash
# Valida o ambiente antes de gerar o modelo baseline.
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
RAW_DIR="${APP_DIR}/data/raw/home-credit-default-risk"

if [ ! -d "${RAW_DIR}" ]; then
  echo "Dados brutos não encontrados em ${RAW_DIR}" >&2
  exit 1
fi

mkdir -p "${APP_DIR}/data/processed/monitoring"
echo "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-}"
echo "MLFLOW_ARTIFACT_LOCATION=${MLFLOW_ARTIFACT_LOCATION:-<padrão local>}"
echo "Ambiente pronto para gerar o baseline."
