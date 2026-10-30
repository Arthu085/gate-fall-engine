# Tasks: Fusão orientada por consistência

**Input**: spec.md, plan.md, research.md, data-model.md e contracts/cli.md.
**Tests**: exigidos por FR-011 e pelos contratos numéricos/causais.

## Phase 1: Setup
- [x] T001 Instalar Spec Kit oficial e consolidar princípios em .specify/memory/constitution.md.
- [x] T002 Escrever spec/plan/research/data-model/contracts/quickstart em specs/001-physics-informed-gate/.
- [x] T003 Instalar ambiente fixado por uv.lock e verificar .gitignore.

## Phase 2: Foundational
- [x] T004 Definir pacotes experimentais e guardas de schema em src/gatefall/experiments/physics/__init__.py.

## Phase 3: US1 - Cinemática válida
- [x] T005 [US1] Escrever e executar casos analíticos antes da correção em src/gatefall/experiments/physics/selftest.py.
- [x] T006 [US1] Implementar derivadas válidas, gap e pose138 em src/gatefall/experiments/physics/kinematics.py.
- [x] T007 [US1] Preparar HDF5 por vídeo, manifest e validação de partições em src/gatefall/experiments/physics/data.py.

Teste independente: movimento uniforme/acelerado, aquisição, ausência e prefixo causal.

## Phase 4: US2 - Consistência e controles
- [x] T008 [US2] Adicionar casos de similaridade, articulação espúria e monotonicidade em src/gatefall/experiments/physics/selftest.py.
- [x] T009 [US2] Implementar resíduos causais e disponibilidade em src/gatefall/experiments/physics/consistency.py.
- [x] T010 [US2] Implementar controles e gate com sinais restritos em src/gatefall/experiments/physics/model.py.

Teste independente: movimentos globais coerentes, perturbação isolada, quatro direções do gate.

## Phase 5: US3 - Corrupção e aprendizado
- [x] T011 [US3] Testar determinismo e gradiente de margem em src/gatefall/experiments/physics/selftest.py.
- [x] T012 [US3] Preparar blocos de ausência apenas no treino em src/gatefall/experiments/physics/data.py.
- [x] T013 [US3] Implementar hinge com margem e CE pareada em src/gatefall/experiments/physics/training.py.

Teste independente: outra modalidade intacta, campos coerentes e perda com gradiente não nulo.

## Phase 6: US4 - Execução, avaliação e provas
- [x] T014 [US4] Testar seleção de checkpoint e integração sintética em src/gatefall/experiments/physics/selftest.py.
- [x] T015 [US4] Implementar stats de treino, checkpoints e métricas em src/gatefall/experiments/physics/training.py.
- [x] T016 [US4] Acrescentar precisão integral opt-in em src/gatefall/eval/shared/events.py.
- [x] T017 [US4] Implementar CLI prepare/train/evaluate/selftest em src/gatefall/experiments/physics/__main__.py.
- [x] T018 [P] [US4] Implementar e compilar teoremas sem sorry/admit em formal/GateFall.lean.

Teste independente: treino CPU, recarga, avaliação e lake build.

## Phase 7: Polish
- [x] T019 Documentar operação e limites em docs/train/physics-gate.md e formal/README.md.
- [x] T020 Registrar novos checks em .github/workflows/ci.yml e navegação em mkdocs.yml.
- [x] T021 Executar selftests pertinentes, uv run pyright, MkDocs e Lean; registrar resultados em specs/001-physics-informed-gate/validation.md.
- [x] T022 Auditar diff e referências, e atualizar conclusão em specs/001-physics-informed-gate/tasks.md.

## Dependencies & Execution Order
Setup -> foundational -> US1 -> US2 -> US3 -> integração US4 -> polish.
T018 é independente do código Python após fixar contratos do plano.

## Parallel Examples
US1: conferir manualmente trajetórias enquanto revisa contratos (sem edições concorrentes).
US2: revisão matemática pode ocorrer durante leitura dos dados, antes de alterar modelo.
US3: testes de gradiente e causalidade rodam separadamente após implementar os respectivos módulos.
US4: provas Lean e validação Python são independentes; integrar seus resultados em T021.

## Implementation Strategy
Incrementos com testes de comportamento antes da implementação. MVP US1 validado
antes dos gates. Concluir as quatro histórias autorizadas, sem parada para nova aprovação.
Não executar retreino real sem fontes; registrar essa limitação e os comandos.
