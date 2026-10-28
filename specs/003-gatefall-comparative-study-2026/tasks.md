# Tarefas

## Preparação

- [x] T001 Importar os módulos physics (`301b32e`) e motion (`5f6e2ae`), MIT, com as specs 001/002
  intactas.
- [x] T002 Escrever esta especificação e os contratos.

## US1 — datasets

- [x] T003 [US1] Implementar `SubjectSplit`/`StagedDatasetSpec` e congelar `upfall-subject-v1` em
  `src/gatefall/data/staged/specs.py`.
- [x] T004 [US1] Implementar rótulos OmniFall pinados e o mapeamento 10–15 → `IGNORE` em `labels.py`.
- [x] T005 [US1] Implementar sequências de imagens, timestamps e staging seguro de ZIP em `sequences.py`.
- [x] T006 [US1] Implementar a grade causal com `source_gap` em `timeline.py`.
- [x] T007 [US1] Implementar manifesto, identidades e proveniência em `build.py`.
- [x] T008 [US1] Implementar vazamento, sincronia, auditoria do split OmniFall e cobertura em `audit.py`.
- [x] T009 [US1] Implementar adaptadores `upfall`/`gmdcsa24` em `src/gatefall/datasets/staged.py`.

## US2 — extração

- [x] T010 [US2] Implementar o despacho vídeo × imagem em `src/gatefall/data/frame_source.py`.
- [x] T011 [US2] Abrir pose, qualidade, DINOv3 e SAM 3 para `upfall`/`gmdcsa24`.
- [x] T012 [US2] Implementar o relatório de cobertura em `src/gatefall/study/coverage.py`.

## US3 — métodos

- [x] T013 [US3] Implementar o TCNTE adaptado em `src/gatefall/experiments/tcnte/`.
- [x] T014 [US3] Integrar TCNTE, `timing` e alarmes em `experiments/physics/training.py`.
- [x] T015 [US3] Implementar o teste sintético de causalidade de todos os métodos e do classificador
  de movimento.

## US4 — transferência

- [x] T016 [US4] Implementar o pacote só de teste e `evaluate-external` sem adaptação.

## US5 — movimento

- [x] T017 [US5] Implementar a matriz de 150 runs no layout importado e `reuse-audit`.

## US6 — registro e relatórios

- [x] T018 Implementar o protocolo congelado e o hash em `src/gatefall/study/protocol.py`.
- [x] T019 Implementar métodos, plano, identidade e ordenação em `methods.py` e `plan.py`.
- [x] T020 Implementar o executor, a proveniência escrita uma vez e o ledger em `registry.py`.
- [x] T021 Implementar as métricas em `metrics.py` e a estatística em `statistics.py`.
- [x] T022 Implementar tabelas, figuras SVG, linhas do tempo e manifesto em `report.py`/`figures.py`.
- [x] T023 Implementar o perfil de extração isolado em `profiling.py`.
- [x] T024 Documentar o estudo no MkDocs (pt-BR).

## Execução real (pendente, depende de dados e GPU)

- [ ] T025 Baixar o UP-Fall (câmeras 1 e 2) e preparar os rótulos e a grade dos dois datasets.
- [ ] T026 Extrair as features do UP-Fall e do GMDCSA-24 e rodar `features-report`.
- [ ] T027 Montar os pacotes e rodar `reuse-audit` para os runs externos de movimento. Pacotes Le2i, atestados e auditoria concluídos (`validation.md`); faltam UP-Fall e GMDCSA-24.
- [ ] T028 Rodar as matrizes `le2i`, `upfall`, `ablations` e `motion` (treino e depois teste). `motion` concluída por reuso atestado (`validation.md`); as demais seguem pendentes.
- [ ] T029 Rodar a matriz `transfer`.
- [ ] T030 Rodar `profile-extraction` e `report` por matriz e revisar os resultados antes do artigo.

## Dependências

- T003–T009 vêm antes de T010–T012.
- T013–T016 dependem dos pacotes.
- T018–T023 dependem de T019.
- T025 → T026 → T027 → T028 → T029 → T030.
