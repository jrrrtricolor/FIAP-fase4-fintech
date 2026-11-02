#!/usr/bin/env bash
# Carrega as credenciais AWS em variáveis de ambiente e executa o comando.
# Cada valor vem de AWS_*_FILE, de /run/secrets/<nome> ou da própria variável.
set -euo pipefail

SECRETS_DIR="${SECRETS_DIR:-/run/secrets}"

load_secret() {
  local var="$1" file_var="${1}_FILE" secret_name="$2" path=""
  if [ -n "${!file_var:-}" ]; then
    path="${!file_var}"
  elif [ -z "${!var:-}" ] && [ -f "${SECRETS_DIR}/${secret_name}" ]; then
    path="${SECRETS_DIR}/${secret_name}"
  fi
  if [ -n "${path}" ]; then
    [ -r "${path}" ] || { echo "Segredo ilegível: ${path}" >&2; exit 1; }
    export "${var}=$(<"${path}")"
  fi
}

load_secret AWS_ACCESS_KEY_ID aws_access_key_id
load_secret AWS_SECRET_ACCESS_KEY aws_secret_access_key
load_secret AWS_SESSION_TOKEN aws_session_token

exec "$@"
