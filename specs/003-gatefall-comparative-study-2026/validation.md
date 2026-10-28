# Validação

Data: 2026-10-28 · Branch: `feat/gatefall-comparative-study-2026`

## Validação real executada pelo usuário: matriz `motion`

A matriz `motion` foi executada pelo usuário com os dados reais do Le2i, no worktree da branch
(revisão `b4463d9`, árvore Git limpa). As evidências ficam em `runs/local/` (fora do Git) e foram
reconferidas com os verificadores do estudo, sem reexecutar treino nem avaliação.

| Evidência | Observado |
| --- | --- |
| Atestados de equivalência (`runs/local/{le2i,le2i_cv}/study2026/equivalence/motion-{dinov3,sam3}.json`) | 4/4 gravados às 16:28Z e reconferidos contra os pacotes e alvos atuais. Cada pacote tem 190 HDF5 relidos e os alvos têm 133 (CS) ou 97 (CV) NPZ, nos lados original e atual. Só há diferenças da allowlist. |
| `runs/local/study2026/reuse-audit.json` (sem atestado) | 0/150 compatíveis, recusados por `manifest_sha256` e `targets_sha256`. |
| `runs/local/study2026/reuse-audit-attested.json` | 150/150 compatíveis, todos via atestado (`compatible_attested = 150`). |
| `execution/execution-2026-10-28T163251Z-a3b8fe46.jsonl` (`--stage train`) | 150 passos `reused`, nenhum `finished`: os treinos não foram refeitos. |
| `execution/execution-2026-10-28T163450Z-f9f71901.jsonl` (`--stage evaluate`) | 150 avaliações `finished`, todas com código 0, em cerca de 495 s de CPU. |
| `study-provenance.json` dos 150 runs | 150/150 com `origin = reused-verified` e o resumo do atestado. 150/150 treinos e 150/150 avaliações conferem nos verificadores atuais. As 150 avaliações são novas: têm `equivalence` e revisão `b4463d9`. |
| `reports/motion-2026-10-28T164408Z/` | 150 avaliações agregadas, todas `reused-verified`: 30 combinações (5 braços × 3 receitas × 2 protocolos), com 5 sementes (42–46) cada. |
| Protocolo congelado | `d61f7769cb5c8deba16d9d935e7dcdad3c01ac820aa9bb40e97241917744c531`, igual em ledgers, proveniências e manifesto do relatório. |
| Estado do Git | revisão `b4463d9`, `git_dirty = false`, em atestados, ledgers, proveniências, avaliações e relatório. |

Conjuntos de teste: o Le2i-CS tem 2 sujeitos e 38 gravações, e o Le2i-CV tem 5 sujeitos e 60
gravações. A interpretação dos resultados está em
`docs/experiments/comparative-study.md#resultados-matriz-motion`.

## Executado na CI e localmente (CPU, dados sintéticos)

Os selftests abaixo validam só contratos sintéticos, não desempenho.


| Checagem | Resultado |
| --- | --- |
| `uv run pyright` | 0 erros, 0 avisos |
| `uv run mkdocs build --strict` | build sem avisos |
| `git diff --check` | sem problemas |
| `uv run python -m gatefall.data.staged selftest` | PASS: alinhamento causal sintético, preparação parcial retomável, artefatos e anotações alterados recusados |
| `uv run python -m gatefall.experiments.physics selftest` | PASS (inclui o treino original idêntico e o pacote `dinov3` com proveniência HDF5 NumPy) |
| `uv run python -m gatefall.experiments.motion selftest` | PASS (inclui o atestado de equivalência: só metadados da allowlist, bytes relidos, 16 recusas, atestados adulterados e avaliação de run histórico sobre o pacote atual) |
| `uv run python -m gatefall.experiments.tcnte selftest` | PASS |
| `uv run python -m gatefall.study selftest` | PASS: plano 80/40/40/150/50 com 300 treinos únicos, causalidade de 9 classificadores, normalização só do treino, métricas, bootstrap por sujeito e gravação, figuras, reuso, integridade antes do reuso, retomada, estágios filtrados com pré-requisitos ausentes, desatualizados ou inválidos, relatório com alvo externo desatualizado, reuso atestado de run histórico (auditoria, `reused-verified`, avaliação refeita, relatório e atestado ou pacote alterados), executor e relatório |
| Selftests sintéticos da CI (45 comandos) | PASS |
| `--help` de todos os subcomandos de `gatefall.study` | OK |
| `gatefall.study plan` e `run` (dry-run) no checkout principal | só leitura e verificação de metadados; nada executado |

## Não executado (pendente)

Os itens abaixo **não foram executados**:

- download do UP-Fall e preparação real (`stage-archives`, `fetch-labels`, `manifest`,
  `timegrid`, `audit`) do UP-Fall e do GMDCSA-24;
- extração de pose, qualidade, DINOv3 e SAM 3 dos datasets novos, e `features-report` real;
- montagem dos pacotes do UP-Fall e do GMDCSA-24;
- as matrizes `le2i`, `upfall`, `transfer` e `ablations`;
- o perfil de extração e os relatórios dessas matrizes.

Os resultados da matriz `motion` são descritivos e pareados por semente. Nenhuma receita foi
selecionada pelo teste, e nenhum desses runs substitui as referências canônicas A/B/C.
