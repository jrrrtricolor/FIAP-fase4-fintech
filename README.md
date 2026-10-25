# Credit Monitoring — FIAP Tech Challenge Fase 4

[![CI — Qualidade, testes e Docker](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml)

Sistema acadêmico de monitoramento de um modelo de risco de crédito. O projeto
valida os dados, simula mudanças de comportamento, identifica drift e treina um
candidato quando há alertas. O foco é a sustentação e a confiabilidade do modelo.

> O retreino não garante melhora. A referência é preservada e a promoção de um
> candidato exige revisão explícita. Os cenários são sintéticos e não autorizam
> decisões reais de crédito.

## Entrega

| Item | Link / situação |
|---|---|
| Repositório | [FIAP-fase4-fintech](https://github.com/jrrrtricolor/FIAP-fase4-fintech) |
| Branch de entrega | [`main`](https://github.com/jrrrtricolor/FIAP-fase4-fintech/tree/main) |
| CI | [GitHub Actions](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml) |
| Enunciado oficial | [Tech Challenge — Fase 4](docs/MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf) |
| Governança e causalidade | [Plano neste README](#governança-e-causalidade) |
| Vídeo STAR | Pendente — demonstração de até cinco minutos |

## Roteiro do avaliador

### 1. Obter o projeto e os dados

Pré-requisitos: Git, [Git LFS](https://git-lfs.com/), Docker com Compose e Make.
No macOS com Homebrew, instale o Git LFS com `brew install git-lfs`.
Execute os comandos a partir da raiz do projeto:

```bash
git lfs install
git clone --branch main https://github.com/jrrrtricolor/FIAP-fase4-fintech.git
cd FIAP-fase4-fintech
git lfs pull
git lfs fsck
```

Em um clone existente, execute `git lfs install`, `git pull` e `git lfs pull`.
O CSV completo deve estar disponível antes de executar os notebooks. Prefira o
clone com Git LFS: o ZIP do GitHub pode conter somente o ponteiro para o CSV.

### 2. Executar os notebooks

```bash
make docker-build
make docker-run
```

Abra [localhost:8888](http://localhost:8888) com o token exibido no terminal.
`make docker-run` permanece ativo enquanto o Jupyter estiver em execução.
Execute todas as células de cada notebook, nesta ordem:

| Notebook | O que demonstra |
|---|---|
| [01 — Análise exploratória](notebooks/01_eda_home_credit.ipynb) | Conhecimento da base, tipos, nulos e validação dos dados |
| [02 — Preparação e treinamento](notebooks/02_preparacao_treinamento.ipynb) | Limpeza, separação, normalização, baseline e registro no MLflow |
| [03 — Estratégia de monitoramento](notebooks/03_estrategia_monitoramento.ipynb) | Contratos, lote inválido, base v2, data/concept drift e histogramas sobrepostos |

### 3. Executar o monitoramento e o retreino

Após os notebooks, abra outro terminal na raiz do projeto:

```bash
docker compose --file docker/docker-compose.yml run --rm notebook \
  python -m src.credit_monitoring
```

O workflow reutiliza a referência e a base v2. Primeiro valida o lote e detecta
mudanças; depois, se houver alerta de drift ou desempenho, treina e avalia um
candidato. Cada execução registra métricas, logs e artefatos no MLflow.
A execução é iniciada por comando; ainda não existe agendamento por DAG.

### 4. Subir e testar a API

O Compose da raiz disponibiliza a API, o Prometheus e o Grafana. Inicialize o
modelo de produção uma única vez:

```bash
export AIRFLOW_UID=$(id -u)
docker compose build credit_monitoring
docker compose run --rm --no-deps \
  --volume ./data/raw:/app/data/raw:ro credit_monitoring \
  python -m src.credit_monitoring.generate_baseline_model
docker compose up -d credit_monitoring prometheus grafana
curl --fail http://localhost:8000/health
curl --fail --retry 10 --retry-all-errors --retry-delay 2 http://localhost:8000/ready
```

A inicialização reutiliza `experiment.joblib` produzido pelo notebook 02. Se não
existir, executa preparação, treino e avaliação pelo mesmo framework. O comando
recusa uma segunda inicialização para preservar o alias `production`. Se ele já
estiver publicado, inicie os serviços sem repetir a inicialização.

Após publicar o baseline, não reexecute o notebook 02 sobre o mesmo armazenamento
de uma API em uso. O UID alinha as permissões dos arquivos entre host e containers.

Exemplo de inferência com as dez features do contrato:

```bash
curl --fail http://localhost:8000/predict \
  --header 'Content-Type: application/json' \
  --data '{"prediction_data":[{
    "AGE_YEARS":35.5,"AMT_INCOME_TOTAL":120000.0,
    "AMT_CREDIT":250000.0,"AMT_ANNUITY":12000.0,"CNT_CHILDREN":0,
    "NAME_INCOME_TYPE":"Working",
    "NAME_EDUCATION_TYPE":"Secondary / secondary special",
    "NAME_HOUSING_TYPE":"House / apartment",
    "FLAG_OWN_CAR":1,"FLAG_OWN_REALTY":0
  }]}'
```

A resposta contém os scores de dificuldade de pagamento, a versão carregada e
o tempo de inferência. O Pipeline calcula os scores uma única vez por chamada.

| Método | Endpoint | Finalidade |
|---|---|---|
| `GET` | `/health` | Processo da API ativo |
| `GET` | `/ready` | Modelo disponível e versão; retorna 503 se a carga falhar |
| `POST` | `/predict` | Validar perfis e calcular scores |
| `GET` | `/metrics` | Métricas para o Prometheus |
| `GET` | `/docs` | Contrato OpenAPI |
| `GET` | `/income-types`, `/education-types`, `/housing-types` | Categorias conhecidas pelo modelo |

### 5. Verificar o monitoramento

| Serviço | Endereço | Verificação |
|---|---|---|
| API | [localhost:8000/docs](http://localhost:8000/docs) | Contrato e inferência |
| Prometheus | [localhost:9090/targets](http://localhost:9090/targets) | Target da API ativo |
| Grafana | [localhost:3000](http://localhost:3000) | Login local `admin` / `admin` |
| MLflow | [localhost:5000](http://localhost:5000) | Iniciar a interface conforme a seção de MLflow abaixo |

No Grafana, abra **Monitor de score de crédito — API e drift**. O dashboard
apresenta métricas da API, drift, contratos, disponibilidade e última avaliação.
As credenciais locais são configuráveis por `GRAFANA_ADMIN_USER` e
`GRAFANA_ADMIN_PASSWORD`.

Para gerar novamente a v2 e atualizar a avaliação após a preparação do baseline:

```bash
docker compose run --rm --no-deps credit_monitoring \
  python -m src.credit_monitoring.generate_drift_data
docker compose run --rm --no-deps credit_monitoring python -m src.credit_monitoring
```

Para encerrar os ambientes:

```bash
docker compose down
docker compose --file docker/docker-compose.yml down
```

## Requisitos oficiais e evidências

O estado abaixo distingue a implementação das pendências de entrega, conforme
os critérios do [enunciado](docs/MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf).

| Critério | Peso | Estado | Evidência principal |
|---|---:|---|---|
| Validação de dados | 25% | Implementada | [Contrato Pandera](ml_prep_kit/src/ml_prep_kit/home_credit_data.py): idade, valores monetários, flags e IDs; [notebook 03](notebooks/03_estrategia_monitoramento.ipynb) demonstra o bloqueio de um lote inválido |
| Detecção de drift | 25% | Implementada | [DriftMonitor](ml_prep_kit/src/ml_prep_kit/drift_monitor.py): Evidently, KS/PSI e relatórios; notebook 03 compara referência e produção |
| Observabilidade e logs | 20% | Parcial | [MLflow e logs](ml_prep_kit/src/ml_prep_kit/experiment_tracker.py), [alertas](src/credit_monitoring/prometheus/alerts.yml) e [dashboard](src/credit_monitoring/grafana/dashboards/credit-monitoring.json); métricas de recursos da infraestrutura pendentes |
| Governança e LGPD | 15% | Documentada, com confirmações pendentes | Plano de privacidade, retenção, vieses e causalidade neste README; licença e enquadramento institucional a confirmar |
| Vídeo STAR | 15% | Pendente | Gravar e inserir o link da demonstração de até cinco minutos |

CI, API e retreino complementam a entrega. Airflow e AWS são metas de evolução
do grupo; não substituem as evidências exigidas de contratos, drift e governança.

## Arquitetura

Os notebooks apresentam o experimento; `src/credit_monitoring` coordena a
execução; o `ml_prep_kit` concentra validação, preparação, modelagem, avaliação,
rastreamento e drift. O modelo usa o `sklearn.pipeline.Pipeline` nativo.

| Etapa | Responsabilidade |
|---|---|
| Referência | Carregar → limpar → validar → separar → normalizar e treinar → avaliar → registrar no MLflow |
| Monitoramento | Carregar lote → validar contrato → comparar com a referência → detectar drift antes de reajustar transformações |
| Candidato | Retreinar quando houver alerta → comparar no teste separado → registrar para revisão |
| Inferência | API carrega a versão do alias `production` na inicialização |
| Observabilidade | `DriftMonitor → summary.json → /metrics → Prometheus → Grafana` |

Metadados do MLflow ficam em SQLite; modelos e relatórios ficam em arquivos.
`data/processed/` é compartilhado entre containers e permanece fora do Git.
Os dados brutos entram por volume; as imagens usam estágios e cache de dependências.

### Airflow e nuvem

O [Compose](docker-compose.yml) e o [Dockerfile do Airflow](Dockerfile.airflow)
preparam a infraestrutura, mas ainda não há DAG de agendamento. As futuras tarefas
devem usar `/opt/venv/bin/python` (`CREDIT_PYTHON`), com diretório de execução `/app`.
O ambiente Python próprio do Airflow permanece separado. O deploy na AWS está pendente.

## Dados e modelo

A fonte é o [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).
São utilizados somente estes arquivos em `data/raw/home-credit-default-risk/`:

| Arquivo | Uso |
|---|---|
| `application_train.csv` | Dados com alvo; 166.133.370 bytes (aproximadamente 158 MiB), versionados pelo Git LFS |
| `HomeCredit_columns_description.csv` | Dicionário de colunas, versionado diretamente no Git |

O baseline é uma regressão logística balanceada, comparada a um modelo Dummy.

A amostra estratificada tem 20.000 registros, semente 42 e dez features. As partições têm IDs
disjuntos: treino (12.000), validação (2.000), teste (2.000) e produção simulada (4.000). Imputação
e normalização são ajustadas no treino; o limiar é escolhido na validação. Teste e produção não
participam do ajuste. `TARGET=1` significa dificuldade de pagamento conforme o dicionário da base.

### Resultados do experimento

| Cenário | Resultado | Interpretação |
|---|---|---|
| Controle | 0/10 features com drift; AUC 0,602 | Referência de desempenho do lote |
| Lote inválido | 20 entradas; nenhuma predição | Contrato bloqueou quatro violações |
| Renda ×0,70 e crédito ×1,40 | 2/10 features com drift; AUC 0,602 | Mudança detectada sem degradação relevante |
| Concept drift | 0/10 features com drift; AUC 0,456 | Queda de 0,145; alertou pela mudança dos desfechos |
| Retreino da v2 | AUC 0,62815 → 0,62795 | Sem ganho suficiente para substituir a referência |
| Recuperação sintética | AUC 0,49460 → 0,78393 | Ganho no teste separado; recall 0,42326 → 0,69535; Brier 0,25154 → 0,19202 |

Resultados de 14/10/2026. Na v2, o novo Min-Max pode compensar as multiplicações; no cenário
sintético, o candidato aprende novos rótulos. Compare modelos dentro do mesmo cenário.

O [resumo local](data/processed/review_evidence/results.json) reúne métricas, partições, hashes e
run IDs. Os [histogramas sobrepostos](data/processed/review_evidence/histogramas_v1_v2.png) mostram
renda e crédito deslocados e idade estável. Essas evidências ficam fora do Git; em um clone novo,
execute os notebooks e consulte os artefatos no MLflow.

### Demonstração de recuperação

```bash
docker compose --file docker/docker-compose.yml run --rm notebook \
  python -m src.credit_monitoring --recovery-demo
```

A demonstração usa o percentil 75 da renda do treino original e semente 42. A partir desse corte,
altera de 0 para 1 parte dos rótulos, com probabilidade 0,50. Features e IDs permanecem iguais. A
mudança é aplicada às quatro partições; a degradação na produção simulada aciona o retreino. O
candidato usa somente treino e validação; os dois modelos são comparados no mesmo teste separado. O
Pipeline completo é salvo no MLflow e verificado após recarga.

`--recovery-demo` usa `credit_scoring.csv` do notebook 02, sem depender da v2, e não pode ser
combinado com `--data`. Após o treino, salva a versão sintética na pasta da execução. Corte e
quantidade de rótulos alterados ficam em `simulation` no resultado e no MLflow. O hash de entrada
identifica o CSV original. A referência permanece preservada; o teste é separado do ajuste,
mas pertence à mesma base. Esta simulação não é validação externa ou temporal.

## Critérios de aceitação e rollback

A aceitação é uma revisão explícita das métricas registradas, sem motor de aprovação ou promoção
automática. Antes de substituir a referência, exigir:

- Contratos válidos, partições disjuntas e Pipeline equivalente após recarga.
- Pelo menos 500 registros de teste e 30 exemplos de cada classe.
- AUC ≥0,60 e ganho de AUC ≥0,02 no mesmo teste da referência congelada.
- Average Precision sem queda, recall com queda máxima de 0,05 e Brier com aumento máximo de 0,01.
- Revisão dos erros por segmento, calibração e custos de falsos positivos/negativos.

Se algum critério falhar, manter a referência e registrar o motivo na revisão. A v2 não atende ao
ganho mínimo; a recuperação atende às métricas globais, mas os limites acadêmicos não autorizam
uso real. Novos ajustes após consultar o teste exigem outro conjunto reservado. Cada candidato
tem sua própria run; a referência permanece disponível. `status=treinado` não indica aprovação.

### Rollback de uma promoção

O alias `production_rollback` é preservado pelo framework quando ocorre uma
promoção explícita para outra versão. Para reverter essa promoção:

```bash
docker compose run --rm --no-deps credit_monitoring \
  python -m src.credit_monitoring.rollback_model
docker compose restart credit_monitoring
curl --fail --retry 10 --retry-all-errors --retry-delay 2 http://localhost:8000/ready
```

Reinicie a API também após uma promoção: ela resolve o alias somente na
inicialização. Sem uma versão anterior no alias de rollback, não há reversão disponível.

## Observabilidade e resposta aos alertas

O MLflow centraliza parâmetros, métricas, status e logs como artefatos. Registra
`run_duration_seconds` e `fit_duration_seconds`; falhas registram a etapa em `failed_stage` e o tipo
em `error_type`, além do status `FAILED` da run.

| Sinal | Resposta do fluxo / ação do responsável |
|---|---|
| `quality_passed=0` | Bloqueia predições; executor corrige a origem e reexecuta |
| `drift_alert=1` | ≥20% das features alteradas; revisar histogramas e testes KS/PSI; workflow aciona candidato |
| `performance_alert=1` | Queda de AUC ≥0,05; investigar desfechos; workflow aciona candidato |
| Run `FAILED` | Consultar etapa e logs, corrigir a causa e iniciar nova execução |
| Run `RUNNING` sem progresso | Conferir processo/container e último log; interrupções abruptas podem impedir o registro da falha |

O collector lê o resumo mais recente do workflow ou do notebook 03, sem
recalcular drift na coleta. As métricas usam o prefixo `credit_monitoring_` e
o label `scenario`; `drift_share` é uma proporção entre 0 e 1.
`summary_available=0` indica ausência de evidência utilizável e
`summary_read_error=1` identifica erro de leitura. O alerta de desatualização
usa 24 horas, ajustáveis em [alerts.yml](src/credit_monitoring/prometheus/alerts.yml).

Sem rótulos recentes, não há avaliação de degradação supervisionada. O Prometheus avalia
os alertas de drift, qualidade e disponibilidade; notificações externas ainda não estão
configuradas. Para verificar infraestrutura:

```bash
docker compose ps
docker compose logs --tail=100 credit_monitoring
docker compose --file docker/docker-compose.yml ps
docker compose --file docker/docker-compose.yml logs --tail=100 notebook
docker stats --no-stream
```

O executor investiga falhas; o mantenedor verifica recursos e faz backup conjunto
de SQLite e artefatos; outro integrante revisa o candidato.

## Desenvolvimento local e MLflow

Pré-requisitos locais: Python 3.12 e Make. As dependências diretas estão fixadas
em `requirements.txt`; PyTorch e ONNX são opcionais em `requirements-ml-prep.txt`.

```bash
make install PYTHON=python3.12
make check
```

Para usar os componentes opcionais do framework, execute `make install-ml-prep`.
O roteiro do avaliador executa os experimentos no Docker para manter os caminhos
do banco e dos artefatos consistentes.

### Consultar o MLflow

```bash
docker compose --file docker/docker-compose.yml run --rm \
  -p 127.0.0.1:5000:5000 notebook \
  mlflow ui --backend-store-uri sqlite:////app/data/processed/monitoring/mlflow.db \
  --host 0.0.0.0 --port 5000
```

Abra [MLflow local](http://localhost:5000). Os experimentos são `tech-challenge-4-fintech-training`
(baseline), `tech-challenge-4-fintech` (cenários) e `tech-challenge-4-credit-monitoring` (retreino).
Na run principal, consulte **Artifacts → result.json / workflow.jsonl**. No candidato, compare
`teste_roc_auc`, `teste_average_precision`, `teste_recall` e `teste_brier_score`; abra `evaluation/`
e os relatórios HTML do monitoramento. O banco usa URIs em `/app`: mantenha SQLite e artefatos
juntos e use o Docker.

### Integração contínua

O workflow [CI](.github/workflows/ci.yml) executa em pushes e pull requests para
`develop` e `main`, com execução manual também configurada. Usa Python 3.12 e
`make check`: Ruff, Flake8, formatação, testes e `pip check`. Os testes usam dados
sintéticos, sem baixar o CSV pelo Git LFS; os módulos opcionais de PyTorch são
ignorados quando essa dependência não está instalada.

Após as verificações, constrói as imagens da API, notebooks e Airflow com cache
separado e valida dependências e imports. Não publica imagens nem executa deploy.
Os resultados ficam na aba **Actions**.

### Publicar alterações nos dados

Mantenha o Git LFS instalado ao fazer commits ou enviar alterações. O hook
configurado por `git lfs install` envia os objetos LFS durante o `git push`.
O outro membro só poderá baixá-los depois que o commit e o objeto LFS tiverem
sido enviados ao GitHub.

Para incluir os objetos LFS no ZIP do GitHub, habilite **Settings → Archives →
Include Git LFS objects in archives**.
[Referência do GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-git-lfs-objects-in-archives-of-your-repository).

## Governança e causalidade

A finalidade é acadêmica, com dados do [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk).
`SK_ID_CURR` serve à rastreabilidade e não entra no modelo. Idade e dados socioeconômicos podem
identificar pessoas em combinação; não se certifica anonimização. Restringir acesso à equipe e
não reidentificar indivíduos. O CSV bruto completo está no Git LFS; filtrar colunas não o minimiza.

Na [LGPD](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm), o art. 7º, X pode
fundamentar tratamentos necessários à proteção do crédito; o art. 20 prevê revisão de decisões
exclusivamente automatizadas. Isso não autoriza qualquer uso acadêmico ou redistribuição.
O enquadramento institucional e a [licença da competição](https://www.kaggle.com/competitions/home-credit-default-risk/rules)
continuam pendentes de confirmação. O grupo não presume ser órgão de pesquisa:
[orientação da ANPD](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/web-guia-anpd-tratamento-de-dados-para-fins-academicos.pdf/@@display-file/file).

Retenção proposta: dados/modelos/banco até 30 dias após avaliação e revisão; logs até 90 dias;
relatórios agregados até 12 meses. O mantenedor revisa acesso, descarte e backups, sem expiração
automática. Apagar um arquivo não remove histórico Git/LFS ou clones. Em exposição suspeita,
restringir acesso e acionar o grupo e a instituição.

No teste original, o recall foi 0,757 abaixo de 35 anos, 0,667 entre 35 e 54 e **0 para os 21
positivos de 55 anos ou mais**. São grupos pequenos, sem intervalos por segmento; a média global
não basta para avaliar equidade. Rever variáveis e erros por grupo antes de qualquer uso real.

A simulação testa intervenções de código em features ou rótulos; não identifica causas econômicas
de inadimplência. Histogramas, KS/PSI e coeficientes não bastam para isso. Um estudo causal exige
temporalidade, controle de confundidores e desenho experimental ou quase experimental adequado.

## Limitações e pendências

- Os cenários simulam mudanças; não representam uma validação temporal ou externa.
- Drift nas entradas não implica queda de desempenho; sem rótulos recentes, não é possível medir essa queda.
- Retreino e revisão de candidatos não garantem recuperação nem autorizam decisões reais de crédito.
- Faltam DAG de agendamento, notificações externas, métricas de recursos, backup automático e deploy na AWS.
- Faltam confirmar licença e enquadramento institucional e publicar o vídeo STAR.

## Equipe

- Cassio Junior
- Júlia Otranto Aulicino
