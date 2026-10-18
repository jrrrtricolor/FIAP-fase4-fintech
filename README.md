# Tech Challenge 4 — Fintech

O projeto explora dados de crédito, treina um modelo de referência e identifica
mudanças nos dados. Os notebooks e o workflow reutilizam o `ml_prep_kit` e o
Pipeline do scikit-learn. O MLflow armazena os metadados em SQLite e os modelos
e relatórios em arquivos locais.

## Obter o projeto e os dados

Instale Git, [Git LFS](https://git-lfs.com/), Docker com Compose e Make.
No macOS com Homebrew, instale o Git LFS com `brew install git-lfs`.

Depois de instalar o Git LFS, execute:

```bash
git lfs install
git clone --branch develop https://github.com/jrrrtricolor/FIAP-fase4-fintech.git
cd FIAP-fase4-fintech
git lfs pull
git lfs fsck
```

Se já tiver o projeto clonado, entre na pasta e execute `git lfs install`,
`git pull` e `git lfs pull`.

Os notebooks usam estes dois arquivos do Home Credit:

```text
data/raw/home-credit-default-risk/
├── application_train.csv
└── HomeCredit_columns_description.csv
```

O primeiro tem 166.133.370 bytes, aproximadamente 158 MiB, e é versionado pelo
Git LFS. O dicionário de colunas é versionado diretamente no Git. O arquivo
de dados completo precisa estar disponível antes de executar os notebooks.

Prefira clonar com Git LFS. O download ZIP do GitHub pode conter apenas um
ponteiro no lugar do CSV: por padrão, os objetos LFS não entram nos arquivos ZIP.
Para distribuí-los dessa forma, habilite **Settings → Archives → Include Git LFS
objects in archives** no repositório.
[Referência do GitHub](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-git-lfs-objects-in-archives-of-your-repository).

## Executar com Docker

Com o Docker iniciado, execute na raiz do projeto:

```bash
make docker-build
make docker-run
```

Abra o endereço `http://localhost:8888` com o token exibido no terminal. O comando
`docker-run` permanece em execução enquanto o Jupyter estiver ativo.

Execute as células de cada notebook na ordem:

1. `notebooks/01_eda_home_credit.ipynb`: exploração e validação dos dados.
2. `notebooks/02_preparacao_treinamento.ipynb`: preparação, treinamento e MLflow.
3. `notebooks/03_estrategia_monitoramento.ipynb`: geração da v2 e detecção de drift.

Depois dos notebooks, execute o workflow de monitoramento e retreinamento:

```bash
docker compose --file docker/docker-compose.yml run --rm notebook \
  python -m src.credit_monitoring
```

O workflow treina um candidato quando há alertas e o compara ao modelo original.
A promoção para produção não é automática. Os cenários são sintéticos; retreinar
não garante melhora de desempenho.

Dados processados, modelos e o banco do MLflow são gerados em `data/processed/`,
fora do Git. Os dados brutos entram no container pelo volume do Compose e não
são copiados para as camadas da imagem Docker.

## Publicar alterações nos dados

Mantenha o Git LFS instalado ao fazer commits ou enviar alterações. O hook
configurado por `git lfs install` envia os objetos LFS durante o `git push`.
O outro membro só poderá baixá-los depois que o commit e o objeto LFS tiverem
sido enviados ao GitHub.

## Dados e resultados

A amostra estratificada tem 20.000 registros, semente 42 e dez features. As partições têm IDs
disjuntos: treino (12.000), validação (2.000), teste (2.000) e produção simulada (4.000). Imputação
e normalização são ajustadas no treino; o limiar é escolhido na validação. Teste e produção não
participam do ajuste. `TARGET=1` significa dificuldade de pagamento conforme o dicionário da base.

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

## Demonstrar recuperação

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

## Critérios para revisar o candidato

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

## Consultar o MLflow

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

Sem rótulos recentes, não há avaliação de degradação supervisionada. Os alertas são flags
consultáveis, sem notificações externas. Para verificar infraestrutura:

```bash
docker compose --file docker/docker-compose.yml ps
docker compose --file docker/docker-compose.yml logs --tail=100 notebook
docker stats --no-stream
```

O executor investiga falhas; o mantenedor verifica recursos e faz backup conjunto de SQLite e
artefatos; outro integrante revisa o candidato. Docker usa estágios e cache de dependências.
Não há monitor externo, backup automático, Grafana, Airflow ou deploy na AWS.

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
A entrega ainda precisa do vídeo STAR de até cinco minutos e das confirmações de governança.
