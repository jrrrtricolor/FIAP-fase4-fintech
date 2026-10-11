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
