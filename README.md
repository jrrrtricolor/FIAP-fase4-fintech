# Credit Monitoring — FIAP Tech Challenge Fase 4

[![CI — Qualidade, testes e Docker](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml)

Sistema acadêmico de monitoramento de um modelo de risco de crédito. O projeto
valida os dados, simula mudanças de comportamento, identifica drift e treina um
candidato quando há alertas. O foco é a sustentação e a confiabilidade do modelo.

> O retreino não garante melhora. O candidato só é promovido se passar pelos
> critérios automáticos de aceitação; a versão anterior permite rollback.
> Os cenários são sintéticos e não autorizam decisões reais de crédito.

## Entrega

| Item | Link / situação |
|---|---|
| Repositório | [FIAP-fase4-fintech](https://github.com/jrrrtricolor/FIAP-fase4-fintech) |
| Branch de entrega | [`main`](https://github.com/jrrrtricolor/FIAP-fase4-fintech/tree/main) — publicação da versão validada pendente |
| CI | [GitHub Actions](https://github.com/jrrrtricolor/FIAP-fase4-fintech/actions/workflows/ci.yml) |
| Enunciado oficial | [Tech Challenge — Fase 4](docs/MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf) |
| Governança e causalidade | [Plano neste README](#governança-e-causalidade) |
| Vídeo STAR | Pendente — demonstração de até cinco minutos |

## Roteiro do avaliador

### 1. Obter o projeto e os dados

A versão validada está no ambiente local da branch `develop`. O roteiro de clonagem
abaixo se aplica após a publicação na `main`; essa publicação e o vídeo continuam pendentes.

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
mudanças. O retreino é automático dentro dessa execução quando há alerta de drift
ou desempenho e `TARGET` completo, com as duas classes, em cada partição de
treino, validação e teste. Produção pode permanecer sem rótulos.

| Condição | Comportamento |
|---|---|
| Sem alerta | Encerra com `sem_drift`, sem treinar |
| Alerta e rótulos suficientes | Treina, avalia e registra a aprovação ou rejeição; promove somente se aprovado |
| Alerta e rótulos ausentes ou incompletos | Mantém o monitoramento e registra `aguardando_rotulos`, com motivo `rotulos_indisponiveis` |
| Alerta e apenas uma classe em treino, validação ou teste | Adia o treino com motivo `classes_insuficientes` |
| Violação do contrato | Interrompe o fluxo e registra a falha no MLflow |

Rótulos parciais não são preenchidos: a avaliação supervisionada é omitida na
partição incompleta. Quando os rótulos estiverem disponíveis, atualize o CSV
preservando os IDs e execute novamente o mesmo comando. O workflow reavalia as
condições e inicia o retreino se elas forem atendidas. A comparação usa o modelo
em produção e o candidato no mesmo teste separado do treinamento.

Cada execução registra decisão, métricas, logs, modelos e falhas no MLflow.
O acionamento permanece por comando; ainda não existe agendamento por DAG.

### 4. Subir e testar a API

O Compose da raiz disponibiliza a API, o Prometheus e o Grafana. Aprove e publique
o baseline uma única vez para habilitar a API e as promoções automáticas:

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
o tempo de inferência. A API verifica o alias a cada 30 segundos, quando recebe uma
requisição. Carrega somente versões aprovadas; se a nova carga falhar, preserva a
anterior válida. Uma aprovação revogada bloqueia o uso daquela versão.

| Método | Endpoint | Finalidade |
|---|---|---|
| `GET` | `/health` | Processo da API ativo |
| `GET` | `/ready` | Modelo aprovado disponível e versão; retorna 503 quando indisponível |
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
`GRAFANA_ADMIN_PASSWORD`. Para CPU, memória e disco, inicie
`docker compose --profile infra up -d cadvisor`. No Docker Desktop, os recursos
representam a VM Linux; sem esse perfil, essas métricas não ficam disponíveis.

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

Situação dos critérios do [enunciado](docs/MLET%20-%20Tech%20Challenge%20-%20Fase%204.pdf):

| Critério | Peso | Estado | Evidência principal |
|---|---:|---|---|
| Validação de dados | 25% | Implementada | [Contrato Pandera](ml_prep_kit/src/ml_prep_kit/home_credit_data.py): idade, valores monetários, flags e IDs; [notebook 03](notebooks/03_estrategia_monitoramento.ipynb) demonstra o bloqueio de um lote inválido |
| Detecção de drift | 25% | Implementada | [DriftMonitor](ml_prep_kit/src/ml_prep_kit/drift_monitor.py): Evidently, KS/PSI e relatórios; notebook 03 compara referência e produção |
| Observabilidade e logs | 20% | Implementada localmente | [MLflow e logs](ml_prep_kit/src/ml_prep_kit/experiment_tracker.py), [alertas](src/credit_monitoring/prometheus/alerts.yml), [dashboard](src/credit_monitoring/grafana/dashboards/credit-monitoring.json) e cAdvisor; notificações externas pendentes |
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
| Candidato | Retreinar quando houver alerta → comparar no teste separado → aprovar ou rejeitar → publicar |
| Inferência | API acompanha o alias `production`; mantém a versão anterior válida se a recarga falhar |
| Observabilidade | `DriftMonitor / workflow → summary.json / result.json → /metrics → Prometheus → Grafana`; recursos via cAdvisor |

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

Demonstração de **27/10/2026**, com 20.000 registros e mudanças sintéticas: 30 verificações
aprovadas, com os arquivos críticos conferidos contra o código atual. Em **29/10/2026**,
os 245 testes passaram (quatro módulos opcionais ignorados) e os três notebooks executaram
sem erros em ambientes temporários. Ruff, Flake8, formatação e regras Prometheus passaram.
O [resumo](docs/evidencias_workflow.json) contém decisões e run IDs; os
[histogramas sobrepostos](docs/evidencias_drift.png) mostram renda e crédito deslocados e idade estável.

| Cenário | Resultado | Interpretação |
|---|---|---|
| Controle | 0/10 features com drift; AUC no lote 0,60153 | Sem retreino |
| Lote inválido | Renda zero inserida em uma linha | Contrato bloqueou antes do drift/treino; run `FAILED` |
| Drift sem rótulos | Renda ×0,70 e crédito ×1,40; 2/10 features com drift | Aguarda rótulos, sem inventar desempenho |
| Drift com rótulos | AUC no lote 0,60218; no teste 0,62815 → 0,62795 | Retreinou e rejeitou o candidato por ganho insuficiente |
| Recuperação sintética | AUC no lote 0,46756; no teste 0,49460 → 0,78393 | Alerta de desempenho; candidato aprovado e publicado |

Na recuperação, recall no teste passou de 0,42326 para 0,69535 e Brier de 0,25154 para
0,19202. Na v2, o novo Min-Max pode compensar as multiplicações; na recuperação, o
modelo aprende novos rótulos. Compare os modelos dentro do mesmo cenário.
A API preservou v1 com o artefato novo indisponível, carregou v2 após restauração e
retornou à v1 com o mesmo score. Os detalhes dessa validação estão em
`data/processed/validation_20261027/`, fora do Git; um clone novo gera suas próprias runs.

### Demonstração de recuperação

```bash
docker compose --file docker/docker-compose.yml run --rm notebook \
  python -m src.credit_monitoring --recovery-demo
```

Usa o percentil 75 da renda do treino original (202.500), semente 42 e probabilidade
0,50 para mudar parte dos rótulos de 0 para 1: 2.657 na demonstração registrada. Preserva
features e IDs; altera as quatro partições, treina somente no treino e escolhe o
limiar na validação. A comparação usa o mesmo teste reservado para ambos os modelos.
`--recovery-demo` usa `credit_scoring.csv`, não pode ser combinado com `--data` e
registra corte e rótulos alterados em `simulation`. O teste é reutilizado entre
cenários: a demonstração não constitui validação temporal ou externa.

## Critérios de aceitação e rollback

O [ModelEvaluator](ml_prep_kit/src/ml_prep_kit/model_evaluator.py) aplica os critérios abaixo.
Ambos os modelos são avaliados nas mesmas linhas de teste, fora do ajuste e da escolha do limiar:

- Contratos válidos, partições disjuntas e Pipeline equivalente após recarga.
- Pelo menos 500 registros de teste e 30 exemplos de cada classe.
- AUC ≥0,60 e ganho de AUC ≥0,02 no mesmo teste da referência congelada.
- Average Precision sem queda, recall com queda máxima de 0,05 e Brier com aumento máximo de 0,01.

A rejeição registra os motivos em `result.json → approval` e preserva a produção. A publicação exige
artefato aprovado, run `FINISHED` e referência inalterada durante o treino. `status=treinado`
não significa aprovação: consulte `approval` e `publication` no resultado. O backend SQLite
usa bloqueio local; não oferece coordenação distribuída. Revisão por segmento, calibração e
custos dos erros continuam necessárias antes de uso real. Novos ajustes após consultar o teste
exigem outro conjunto reservado.

### Rollback de uma promoção

O alias `production_rollback` preserva a versão anterior. Para restaurá-la,
o comando verifica sua aprovação e disponibilidade antes da troca:

```bash
docker compose run --rm --no-deps credit_monitoring \
  python -m src.credit_monitoring.rollback_model
curl --fail --retry 10 --retry-all-errors --retry-delay 2 http://localhost:8000/ready
```

A API adota a troca na próxima verificação; aguarde até 30 segundos e confirme a
versão em `/ready`. Sem uma versão anterior aprovada, o rollback é recusado.

## Observabilidade e resposta aos alertas

O MLflow registra parâmetros, decisões, métricas e logs, incluindo duração total,
de treino e de cada etapa (`step_<etapa>_duration_seconds`). O workflow publica
`result.json` atomicamente; falhas preservam `failed_stage` e `error_type`, com run `FAILED`.

| Sinal | Resposta do fluxo / ação do responsável |
|---|---|
| `quality_passed=0` | Bloqueia o lote; corrigir a origem antes de repetir |
| `drift_alert=1` | ≥20% das features alteradas; revisar histogramas e KS/PSI; aciona candidato com rótulos suficientes |
| `performance_alert=1` | Queda de AUC ≥0,05; investigar desfechos; aciona candidato com rótulos suficientes |
| `up=0` / `api_ready=0` / `model_reload_error=1` | Verificar container, Registry e artefatos; consultar `/ready` e recuperar uma versão aprovada |
| `workflow_failed=1` / erro de leitura | Consultar etapa, logs, formato e permissões do volume; corrigir e reexecutar |
| Etapa >600 s / workflow sem atualização por 1.800 s | Conferir processo, último log e recursos antes de reiniciar |
| Avaliação com mais de 24 horas | Executar monitoramento; ajustar a janela à frequência dos lotes |
| CPU >90%, memória >90% ou disco >85% | Com perfil `infra`, investigar consumo e volumes; preservar banco e artefatos |

O collector lê `summary.json` e `result.json` sem recalcular drift. As métricas usam
`credit_monitoring_`, com `scenario`, `feature` ou `stage` conforme o caso.
`drift_share` varia de 0 a 1; `feature_drift` identifica as variáveis afetadas.
`summary_available=0` e erros de leitura indicam ausência de evidência utilizável.
Os horários de avaliação e progresso são persistidos; uma falha recente não exibe
um resumo antigo como atual. Sem rótulos, não há avaliação supervisionada.

As [regras](src/credit_monitoring/prometheus/alerts.yml) também cobrem erros HTTP,
p95 de inferência e confiança média; usam confirmação de um ou cinco minutos.
Notificações externas ainda não têm canal configurado. Para investigar:

```bash
docker compose ps
docker compose logs --tail=100 credit_monitoring
docker compose --profile infra logs --tail=100 cadvisor
docker stats --no-stream
```

O executor investiga falhas; o mantenedor verifica recursos e faz backup conjunto
de SQLite e artefatos; o grupo revisa limites e resultados por segmento.

A validação de 27/10 provisionou os 14 painéis e confirmou consultas HTTP 200 e os targets
API, Prometheus e cAdvisor ativos. Os painéis de p95 e confiança ficaram sem valores por
tráfego insuficiente; gerá-lo e conferir esses painéis faz parte da preparação do vídeo.

## Desenvolvimento local e MLflow

Pré-requisitos locais: Python 3.12 e Make. As dependências diretas estão fixadas
em `requirements.txt`; PyTorch e ONNX são opcionais em `requirements-ml-prep.txt`.

```bash
make install PYTHON=python3.12
make check
```

Para os componentes opcionais, execute `make install-ml-prep`. Use o Docker no
roteiro do avaliador para manter consistentes os caminhos do banco e dos artefatos.
Para definir a localização dos artefatos em novos experimentos, configure
`MLFLOW_ARTIFACT_LOCATION` no ambiente de execução. Os comandos de treinamento,
monitoramento e rollback encaminham esse valor para `ml_prep_kit`.

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

Consulte a decisão em `result.json → retraining` e na tag `retraining_reason`.
As runs de monitoramento e candidato estão ligadas à execução principal; as métricas
`retraining_triggered` e `labels_available` registram acionamento e disponibilidade de rótulos.
Na run principal, `result.json` reúne `approval` e `comparison`; no candidato,
consulte `approval_status` e `candidate_approved`. Confira os aliases no Registry.

Para consultar as runs históricas do resumo, com o arquivo local disponível, acrescente
`-v ./data/processed/validation_20261027:/validation` antes de `notebook` no comando acima
e use `sqlite:////validation/demo-20261027T190753124789Z/data/processed/monitoring/mlflow.db`
como backend. Um clone novo deve executar o roteiro para gerar suas próprias runs.

### Integração contínua

O workflow [CI](.github/workflows/ci.yml) executa em pushes e pull requests para
`develop` e `main`, com execução manual também configurada. Usa Python 3.12 e
`make check`: Ruff, Flake8, formatação, testes e `pip check`. Os testes usam dados
sintéticos, sem baixar o CSV pelo Git LFS; os módulos opcionais de PyTorch são
ignorados quando essa dependência não está instalada.

A CI também valida as regras com `promtool`. Após as verificações, constrói as imagens da API, notebooks e Airflow com cache
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
Antes de ampliar a distribuição da base, o mantenedor deve registrar essa confirmação.
As evidências da entrega apresentam resultados agregados, sem incluir registros individuais.

Retenção proposta: dados/modelos/banco até 30 dias após avaliação e revisão; logs até 90 dias;
relatórios agregados até 12 meses. O mantenedor revisa acesso, descarte e backups, sem expiração
automática. Apagar um arquivo não remove histórico Git/LFS ou clones. Em exposição suspeita,
restringir acesso e acionar o grupo e a instituição.

No teste original, o recall foi 0,757 abaixo de 35 anos, 0,667 entre 35 e 54 e **0 para os 21
positivos de 55 anos ou mais**. São grupos pequenos, sem intervalos por segmento; a média global
não basta para avaliar equidade. A aprovação automática usa métricas globais. O grupo deve
investigar os falsos negativos por faixa, avaliar representação e qualidade dos dados e
comparar recall e taxa de falsos positivos com intervalos de confiança. Uma eventual
mitigação deve ser avaliada em novo conjunto reservado; esse risco ainda não foi resolvido.

A simulação testa intervenções de código em features ou rótulos; não identifica causas econômicas
de inadimplência. Histogramas, KS/PSI e coeficientes não bastam para isso. Um estudo causal exige
temporalidade, controle de confundidores e desenho experimental ou quase experimental adequado.

A hipótese do cenário é que mudanças econômicas afetem renda, crédito solicitado e capacidade
de pagamento. Na simulação, renda ×0,70 e crédito ×1,40 geraram drift em duas features,
com AUC do lote praticamente estável (0,60153 → 0,60218). Esse resultado sustenta revisar a
distribuição das entradas; não sustenta atribuir maior inadimplência à queda de renda.
Na recuperação sintética, a regra que altera rótulos é conhecida e o retreino aprende essa
nova relação. A melhora observada comprova a resposta do workflow ao experimento.

Na operação real, emprego, composição etária e política de aprovação seriam explicações
alternativas a investigar. Seriam necessários dados por período e desfechos posteriores
à concessão para distinguir mudança econômica de seleção dos clientes ou erro na origem.
A ação proposta é validar a origem do alerta, acompanhar desempenho quando os rótulos
chegarem e comparar o candidato no teste reservado antes de qualquer promoção.

## Limitações e pendências

- Os cenários simulam mudanças; não representam validação temporal ou externa.
- Drift não implica perda de desempenho; sem rótulos recentes, essa perda não pode ser medida.
- Aprovação automática não garante recuperação nem autoriza decisões reais de crédito.
- SQLite e volume compartilhado atendem à demonstração local, sem publicação distribuída.
- Evoluções: DAG de agendamento, controle de lotes já processados, notificações externas,
  backup automático e deploy na AWS. O fluxo atual compara versões com os mesmos IDs da referência.
- Confirmar licença e enquadramento institucional antes de ampliar a distribuição dos dados.
- Entrega pendente: publicar a versão validada em `main`, confirmar sua CI e gravar/linkar o vídeo STAR.

## Equipe

- Cassio Junior
- Júlia Otranto Aulicino
