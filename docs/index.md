# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de
Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso em Visão Computacional.
Autor: Arthur Ghizi · Orientador: Rodrigo Ramos Silva.

## Resumo

Detectores de queda baseados exclusivamente em pose degradam justamente onde
mais importam: oclusão parcial, truncamento pela borda do quadro e as poses
atípicas do corpo já no chão são os regimes em que o estimador de pose perde
confiança. O GateFall investiga se a informação visual densa de modelos de
fundação congelados (DINOv3, SAM 3) recupera esses casos, e como a fusão deve
ponderar pose e informação visual em função da confiança de cada fonte.

O pipeline é RGB monocular. Os backbones são congelados e as features são
pré-computadas offline; apenas a cabeça de fusão e o codificador temporal (TCN)
são treinados.

## Configurações experimentais

| Config | Conteúdo de cada linha da janela |
| --- | --- |
| A | Pose (YOLO-Pose) |
| B | Pose + embedding visual (DINOv3) |
| C | Pose + descritor de máscara (SAM 3) |

As três configurações são mantidas rigorosamente idênticas em tamanho de janela,
split treino/teste, seed, codificador temporal e número de épocas. A única
variável é o conteúdo de cada linha da janela.

## Ambiente de desenvolvimento

Instalação via `uv sync`. Ver a seção
[Instalação](https://github.com/Arthu085/gate-fall-engine/blob/main/README.md#instalação)
do README para os comandos, incluindo como servir esta documentação
localmente (`uv run mkdocs serve`). Para checagem de tipos (mesmos
diagnósticos do Pylance no VS Code), rode `uv run pyright`.

### Instruções para agentes de código

As regras específicas do projeto são mantidas em `CLAUDE.md`, para Claude Code,
e em `AGENTS.md`, para Codex. Os dois arquivos preservam os mesmos invariantes
experimentais e gates do repositório, adaptados ao formato nativo de cada
agente. Configurações, skills e agentes personalizados globais permanecem fora
do repositório, na configuração pessoal do desenvolvedor.

## Dados

O diretório `data/` separa vídeos brutos (`data/raw/`, não versionado),
anotações (`data/labels/`, não versionado) e features pré-computadas
(`data/features/`, não versionado). As anotações do Le2i são obtidas do
dataset [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall)
via `uv run python scripts/fetch_labels.py`; os vídeos em si continuam sendo
obtidos manualmente na fonte original. Depois de colocados em
`data/raw/le2i/`, `uv run python -m gatefall.data.ingest ingest` casa cada
vídeo com sua anotação e grava `data/manifest.parquet`. Ver a seção
[Dados](https://github.com/Arthu085/gate-fall-engine/blob/main/README.md#dados)
do README para o detalhamento completo e [Proveniência dos
dados](data-provenance.md) para a revisão fixada, checksums, a regra de
casamento vídeo-label e citações.

## Estado do projeto

Repositório em fase de estruturação inicial. Já existem scripts de bootstrap
das anotações (`scripts/fetch_labels.py`) e de ingestão dos vídeos
(`gatefall.data.ingest`), mas ainda não há código de extração de features,
treino ou avaliação.
