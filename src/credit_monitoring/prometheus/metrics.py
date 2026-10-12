from prometheus_client import Counter, Gauge, Histogram, Summary

PREDICTIONS_TOTAL = Counter(
    "total_de_predicoes",
    "Total de predições realizadas",
)
ERRORS_TOTAL = Counter(
    "total_de_erros_na_api",
    "Total de respostas HTTP com erro e exceções não tratadas",
)
PREDICTION_DURATION = Histogram(
    "duracao_da_predicao_segundos",
    "Duração do processamento da predição em segundos",
)
PREDICTION_CONFIDENCE = Histogram(
    "confianca_da_predicao",
    "Maior probabilidade retornada em cada predição",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge(
    "modelo_info",
    "Identifica a versão do modelo carregado pela API",
    labelnames=("version",),
)

REFERENCE_MODEL_AMT_INCOME_TOTAL = Summary(
    "referencia_de_renda",
    "Summary de valores de renda do modelo em produção",
    labelnames=['ref_amt_income']

)
REFERENCE_MODEL_AMT_CREDIT = Summary(
    "referencia_de_credito",
    "Summary de valores de credito do modelo em produção",
    labelnames=['ref_amt_credit']

)
REFERENCE_MODEL_AGE_YEARS = Summary(
    "referencia_de_idades_em_anos",
    "Summary de valores de idade em anos do modelo em produção",
    labelnames=['ref_age_years']
)
