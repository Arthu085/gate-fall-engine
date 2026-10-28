# Início rápido

O roteiro completo, com pré-requisitos, artefatos, recursos e reuso por estágio, está em
[`docs/experiments/comparative-study-runbook.md`](../../docs/experiments/comparative-study-runbook.md).
Os comandos exatos saem do próprio plano, sempre na raiz do checkout que contém `data/` e `runs/`:

```bash
uv run python -m gatefall.study selftest                          # contratos sintéticos (CPU)
uv run python -m gatefall.study protocol                          # protocolo e SHA-256
uv run python -m gatefall.study plan                              # contagens e estado
uv run python -m gatefall.study commands --matrix upfall          # comandos exatos por estágio
uv run python -m gatefall.study run --matrix upfall               # dry-run: só imprime
uv run python -m gatefall.study run --matrix upfall --execute     # execução real (pendente)
uv run python -m gatefall.study report --matrix upfall            # depois de todas as avaliações
```
