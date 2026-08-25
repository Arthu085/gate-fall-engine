# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de
Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso (TCC) em Visão Computacional.
**Autor:** Arthur Ghizi · **Orientador:** Rodrigo Ramos Silva.

O GateFall investiga se informação visual de modelos de fundação congelados
melhora a robustez de um detector de quedas baseado em pose nos casos em que a
estimativa de pose se degrada, como oclusões, truncamentos e poses atípicas do
corpo caído. O pipeline usa vídeo RGB monocular; os backbones são executados
offline e apenas a cabeça de fusão e o codificador temporal são treinados.

## Configurações experimentais

| Config | Conteúdo de cada linha da janela    | Codificador temporal |
| ------ | ----------------------------------- | -------------------- |
| **A**  | Pose (YOLO-Pose)                    | TCN                  |
| **B**  | Pose + embedding visual (DINOv3)    | TCN                  |
| **C**  | Pose + descritor de máscara (SAM 3) | TCN                  |

As três configurações são idênticas em tamanho de janela, split treino/teste,
seed, codificador temporal e número de épocas. A única variável é o conteúdo do
vetor de features por timestep.

## Instalação

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. O `uv` usa a versão
definida em `.python-version`.

```bash
git clone https://github.com/Arthu085/gate-fall-engine.git
cd gate-fall-engine
uv sync
```

Para servir a documentação localmente:

```bash
uv run mkdocs serve
```

Para verificar os tipos do código Python:

```bash
uv run pyright
```

## Dados

Os artefatos locais ficam em `data/raw/`, `data/labels/`, `data/features/` e
`data/manifest.parquet`. Vídeos, anotações baixadas, features e o manifesto não
são versionados.

A documentação detalhada está dividida por responsabilidade:

- [Organização geral e fluxo dos dados](docs/data/organization.md);
- [Anotações e proveniência do OmniFall](docs/data/omnifall.md);
- [Obtenção e preparação do Le2i](docs/data/le2i.md);
- [Manifesto e verificações](docs/data/manifest-verification.md).

## Licença

O **código** deste repositório é distribuído sob a licença [MIT](LICENSE). Os
datasets e as anotações possuem termos próprios, indicados junto às respectivas
fontes na documentação de dados.
