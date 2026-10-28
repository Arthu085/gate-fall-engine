# Validação da implementação

Data: 2026-10-09. Branch: `codex/physics-informed-gate`.
Base: `ef6b3c19d3484e2d33ccbb8e3bcd486e14040f69`.

## Spec Kit

Utilizado o Spec Kit oficial, revisão
`7666b7463ffc8b3dce49f75be017f163a46792fa`, integração Codex com skills.
Constituição, especificação, plano, pesquisa, modelo de dados, contrato CLI,
quickstart, checklist e tarefas registrados nesta entrega. Os scripts oficiais
`setup-plan`, `setup-tasks` e `check-prerequisites -RequireTasks -IncludeTasks`
completaram. Nenhum hook de extensão estava configurado.

## Evidências executadas

| Verificação | Resultado |
| --- | --- |
| `uv sync --locked --dev` | Ambiente instalado pelo lock existente; lock e dependências não alterados. |
| `uv run python -m gatefall.experiments.physics selftest` | Passou: cinemática, causalidade, consistência, sinais, saturação float32, corrupção, gradientes, margem, latência e preparação pelo adapter. |
| Integração no mesmo selftest | Dez treinos sintéticos completos: sete modos com SAM, pose sem visual, B0 e B1 com DINO. Os cinco braços completaram recarga e avaliação. |
| Isolamento e integridade | Modificar somente features/rótulos de teste não modifica stats; variantes só no treino; alteração de HDF5 é recusada pelo hash; identidade fonte divergente é recusada. |
| `run_events_selftest()` em `gatefall.eval.shared.selftests.events` | 18 testes legados passaram com os defaults preservados. |
| `lake build --wfail`, em `formal/` | Passou: 19 teoremas, somente axiomas padrão, sem lacunas. Detalhes em `formal/validation.md`. |
| `uv run mkdocs build --strict` | Passou. |
| `git diff --check` | Passou. |
| Referências, estatísticas legadas, `uv.lock` e `pyproject.toml` | Sem alterações. |

Os testes são contratos analíticos e de integração, não estimativas de eficácia
Le2i. Os escores sintéticos impressos pelo treino não são resultados científicos.

## Plataforma e limites

`uv run pyright` foi executado. No Windows ele reporta cinco erros preexistentes
em `src/gatefall/eval/shared/event_artifacts.py`, envolvendo `fcntl.flock` e
suas constantes. Esse arquivo não foi modificado. A checagem com
`uv run pyright --pythonplatform Linux`, correspondente à plataforma da CI,
passou com zero erros. O comando agregado `gatefall.eval.baseline_a selftest`
também falha na importação desse módulo Linux no Windows; a suíte de eventos
afetada pela mudança foi executada diretamente e passou integralmente.

A CI foi atualizada para executar o selftest novo e um job Lean separado.
O workflow remoto ainda não foi executado nesta entrega local.

Uma tentativa de preparação real confirmou a ausência de
`data/processed/le2i/frames.parquet`. Não foram executados retreinos ou avaliações
reais. O runner foi validado com fontes sintéticas; a validação dos arquivos
reais dependerá de restaurar/gerar as fontes offline descritas nos runbooks.

## Revisão de desenho

A revisão não encontrou vazamento entre treino e teste, corrupção cruzada de
modalidades ou consulta do teste para seleção. A qualidade de pose legada ainda
penaliza algumas rotações/perspectivas: a invariância se aplica ao residual novo,
não ao gate inteiro. Essa hipótese foi explicitada na spec e no guia.

A avaliação entregue é limpa. Avaliação de robustez em falhas reservadas,
blur por reinferência, calibração dos alarmes e correção de proporção DINO
são ablações posteriores. Ganhos reais não são garantidos pelas provas Lean.
