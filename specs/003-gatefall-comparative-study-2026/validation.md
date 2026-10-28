# Validação

Data: 2026-10-28 · Branch: `feat/gatefall-comparative-study-2026`

## Executado (CPU, dados sintéticos)

| Checagem | Resultado |
| --- | --- |
| `uv run pyright` | 0 erros, 0 avisos |
| `uv run mkdocs build --strict` | build sem avisos |
| `git diff --check` | sem problemas |
| `uv run python -m gatefall.data.staged selftest` | PASS: alinhamento causal sintético, preparação parcial retomável, artefatos e anotações alterados recusados |
| `uv run python -m gatefall.experiments.physics selftest` | PASS (inclui o treino original idêntico) |
| `uv run python -m gatefall.experiments.motion selftest` | PASS |
| `uv run python -m gatefall.experiments.tcnte selftest` | PASS |
| `uv run python -m gatefall.study selftest` | PASS: plano 80/40/40/150/50 com 300 treinos únicos, causalidade de 9 classificadores, normalização só do treino, métricas, bootstrap por sujeito e gravação, figuras, reuso, integridade antes do reuso, retomada, executor e relatório |
| Selftests sintéticos da CI (45 comandos) | PASS |
| `--help` de todos os subcomandos de `gatefall.study` | OK |
| `gatefall.study plan` e `run` (dry-run) no checkout principal | só leitura e verificação de metadados; nada executado |

## Não executado (pendente)

Os itens abaixo dependem de dados, pesos ou GPU e **não foram executados**, por restrição do
pedido:

- download do UP-Fall e preparação real (`stage-archives`, `fetch-labels`, `manifest`,
  `timegrid`, `audit`) do UP-Fall e do GMDCSA-24;
- extração de pose, qualidade, DINOv3 e SAM 3 dos datasets novos, e `features-report` real;
- montagem dos pacotes reais e dos alvos de movimento;
- `reuse-audit` sobre os 150 runs externos, cujos artefatos não estão disponíveis;
- todas as matrizes (`le2i`, `upfall`, `transfer`, `motion`, `ablations`), o perfil de extração e
  os relatórios reais.

Nenhum resultado científico foi produzido ou é afirmado.
