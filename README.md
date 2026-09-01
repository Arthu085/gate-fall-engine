# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de
Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso em Visão Computacional de Arthur Ghizi,
orientado por Rodrigo Ramos Silva. O GateFall investiga detecção de quedas em
vídeo RGB monocular com backbones congelados e features pré-computadas.

O braço A, implementado, usa YOLO-Pose e uma TCN. Os braços B (DINOv3) e C
(SAM 3) estão planejados; entre os três, somente o vetor de features por
timestep deve mudar.

## Instalação e reprodução

Requer Python 3.12, [uv](https://docs.astral.sh/uv/) e FFmpeg/ffprobe.

```bash
git clone https://github.com/Arthu085/gate-fall-engine.git
cd gate-fall-engine
uv sync
uv run python -m gatefall.pipeline run --dataset le2i --arm A
```

Os dados externos precisam ser preparados conforme as licenças de suas
fontes. A reprodução grava somente em `runs/local/`; resultados históricos
versionados ficam em `runs/reference/`.

## Documentação

- [Arquitetura](docs/architecture/overview.md)
- [Tecnologias](docs/architecture/technology-stack.md)
- [Organização e preparação dos dados](docs/data/organization.md)
- [Referência de comandos](docs/reference/commands.md)
- [Runbook completo do pipeline A](docs/runbooks/pipeline-a.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos](docs/eval/baseline-a-events.md)

Sirva o site com `uv run mkdocs serve`. O gate de desenvolvimento é
`uv run pyright` e `uv run mkdocs build --strict`; a CI também executa todos
os selftests sintéticos.

## Licença

O **código do GateFall** é distribuído sob a licença [MIT](LICENSE). Datasets,
dependências de modelos e pesos pré-treinados têm termos próprios; a MIT do
repositório não se estende automaticamente a esses materiais externos.
