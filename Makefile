SHELL := /bin/sh
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
IMAGE ?= fiap-fase4-fintech
TAG ?= dev
export IMAGE TAG

VENV_PYTHON := $(VENV)/bin/python
VENV_RUFF := $(VENV)/bin/ruff
VENV_PYTEST := $(VENV)/bin/pytest
DEPENDENCIES_STAMP := $(VENV)/.dependencies-installed

.PHONY: help install lint format format-check test check notebook docker-build docker-run

help: ## Exibe os comandos disponíveis
	@awk 'BEGIN {FS = ":.*## "; printf "Uso: make <comando>\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(DEPENDENCIES_STAMP): requirements.txt requirements-ml-prep.txt $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install --requirement requirements.txt
	$(VENV_PYTHON) -m pip install --requirement requirements-ml-prep.txt
	@touch $(DEPENDENCIES_STAMP)

install: $(DEPENDENCIES_STAMP) ## Cria o ambiente virtual e instala as dependências

lint: install ## Verifica qualidade e conformidade PEP 8
	$(VENV_RUFF) check .

format: install ## Formata o código Python e os notebooks
	$(VENV_RUFF) format .

format-check: install ## Confere a formatação sem alterar arquivos
	$(VENV_RUFF) format --check .

test: install ## Executa os testes do projeto
	PYTHONPATH=src:ml_prep_kit/src $(VENV_PYTEST) tests ml_prep_kit/tests

check: lint format-check test ## Executa todas as verificações de qualidade

notebook: install ## Inicia o JupyterLab localmente
	$(VENV)/bin/jupyter lab --notebook-dir=.

docker-build: ## Constrói a imagem Docker
	docker compose --file docker/docker-compose.yml build

docker-run: ## Inicia o JupyterLab no container
	docker compose --file docker/docker-compose.yml up
