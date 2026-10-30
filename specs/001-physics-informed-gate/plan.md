# Implementation Plan: Fusão orientada por consistência

**Branch**: `codex/physics-informed-gate` | **Date**: 2026-10-09 | **Spec**: [spec.md](spec.md)

**Input**: especificação aprovada pelo pedido de implementação nesta conversa.

## Summary
Criar uma CLI experimental versionada, com preparação offline por vídeo,
cinemática corrigida, controles de fusão e evidência causal de consistência.
Preservar integralmente entradas e caminhos históricos. Provas Lean verificam
as equações ideais; selftests verificam a implementação numérica e a integração.

## Technical Context
**Language/Version**: Python 3.12; Lean 4.19.0 e Mathlib fixados.
**Primary Dependencies**: NumPy, PyTorch, pandas, h5py, PyYAML já no projeto.
**Storage**: HDF5 por vídeo, manifest/config/stats/metrics JSON e checkpoints PyTorch.
**Testing**: selftests existentes e novos, uv run pyright, MkDocs estrito, lake build.
**Target Platform**: Linux/WSL e CPU Windows para validação sintética.
**Project Type**: biblioteca e CLI de pesquisa.
**Performance Goals**: sem reinferência de backbone; preparação linear nos quadros,
com pequenas operações de similaridade 2D por quadro; sem materializar janelas duplicadas.
**Constraints**: causal, RGB monocular, backbones congelados, estatísticas de treino,
TCN [32,32,32], dilatações [1,2,4], janela 24, 30 épocas por padrão em todos os braços.
**Scale/Scope**: A/B0/B1/C0/C1 e ablações internas, sem novo dataset/backbone.

## Constitution Check
PASS antes e depois do desenho: partições originais preservadas; preparo comum;
qualquer mudança de arquitetura é registrada como ablação, sem promoção; artefatos
v2 separados; pesquisa substitui novos backbones; dados reais ausentes explicitados.
A extração não altera os dados brutos, estatísticas legadas ou referências.

## Project Structure
### Documentation (this feature)
`specs/001-physics-informed-gate/`: spec, plan, research, data-model,
contracts/cli.md, quickstart, tasks e checklist de requisitos.
### Source Code (repository root)
- `src/gatefall/experiments/physics/kinematics.py`: derivadas válidas e features v2.
- `src/gatefall/experiments/physics/consistency.py`: evidência geométrica causal.
- `src/gatefall/experiments/physics/data.py`: preparação, variantes e proveniência.
- `src/gatefall/experiments/physics/model.py`: controles e gate com restrição de sinais.
- `src/gatefall/experiments/physics/training.py`: estatísticas, janelas, CE pareada, seleção e avaliação.
- `src/gatefall/experiments/physics/__main__.py`: CLI prepare/train/evaluate/selftest.
- `src/gatefall/experiments/physics/selftest.py`: casos analíticos e integração.
- `src/gatefall/eval/shared/events.py`: opção explícita de precisão integral, default legado.
- `formal/`: projeto Lean com versões fixadas e teoremas.
- `docs/train/physics-gate.md`: operação e interpretação do experimento.

**Structure Decision**: módulo experimental isolado evita incompatibilidade silenciosa
com as 134 features, estatísticas e checkpoints já publicados. Reutilizar TCN,
métricas, janelamento, formatos brutos e verificadores de proveniência existentes.

## Design
- Correção usa velocidades entre observações reais e aceleração entre centros
  dos intervalos. Gap máximo padrão 0,5 s, alterável antes do experimento.
- Pose v2 mantém 134 canais e acrescenta person_found, velocity_valid,
  acceleration_valid e idade normalizada da última observação (138 canais).
- Evidência: q_pose, q_visual, r_pose, r_visual, r_pose_valid, r_visual_valid,
  pose_present, visual_present. Somente os quatro primeiros são scores.
- r_pose deriva de ajuste robusto de similaridade entre keypoints confiáveis,
  removendo movimento global 2D; não penaliza aceleração global.
- r_visual SAM usa variação relativa de área/preenchimento entre observações
  próximas. É proxy suave, não dinâmica física. DINO: r_visual/valid iguais a zero.
- gate physics: sigmoid(b+softplus(a)qp-softplus(c)qv-softplus(d)rp*vp
  +softplus(e)rv*vv). Peso inicial dos resíduos pequeno. Disponibilidade é
  informada também ao classificador; não forçar um vencedor quando ambas faltam.
- Controles: pose (A), concat (B0/C0), constant, quality_concat, learned,
  monotonic, physics (B1/C1). Entradas auxiliares explicitamente registradas.
- Corrupção inicial: apagamento causal em blocos, offline, uma modalidade por vez;
  pose é reconstruída do dado bruto com person_found falso; visual indisponível
  é zero com q/presença atualizados e residual recalculado para SAM.
- CE limpa e corrompida; hinge de margem positiva sobre gates no grafo,
  somente nos quadros alterados e originalmente presentes. Não usa rótulos futuros.
- Validação escolhe melhor macro-F1 (primeiro empate), sem early stopping;
  último e escolhido salvos. Estatísticas e corrupção não usam teste.
- Eventos usam política legada de associação e refratário, com precisão integral
  explicitamente solicitada no novo runner. Não mudar simultaneamente alarmes.
- Artefatos novos recusam overwrite; relatórios de teste são explícitos após treino.

## Complexity Tracking
Sem violação das invariantes. A CLI separada evita modificar todos os leitores
legados e torna o novo contrato explícito; não duplica TCN/backbones/métricas.
