# Treino — Arma C0 (fusão pose + SAM 3 por concatenação)

`src/gatefall/train/c0_*.py` implementa a arma C0 do braço C: pose (134-d) e o
descritor de máscara do SAM 3 `V_t` (10-d, ver [Fundação
SAM 3](../data/sam3-foundation.md#descritor-v_t)) projetados separadamente e
fundidos por concatenação simples antes da mesma TCN causal dilatada do
[Braço A](baseline-a.md). É a contraparte exata da [arma B0](b0-fusion.md):
só a fonte visual por timestep muda.

## Arquitetura

`C0FusionClassifier` (`src/gatefall/train/c0_model.py`):

- `E_P`: `Linear(134, 128) -> LayerNorm(128) -> ReLU`, projeta a pose.
- `E_V`: `Linear(10, 128) -> LayerNorm(128) -> ReLU`, projeta `V_t`.
- Concatenação de `E_P(pose)` e `E_V(V_t)` no eixo de feature, produzindo um
  vetor fundido de 256 dimensões por timestep (`FUSED_DIM = 256`).
- O vetor fundido alimenta a mesma `TCNEncoder` do braço A, seguida de uma
  cabeça linear many-to-one sobre o último timestep.

Não há gate, atenção cruzada nem peso de confiança entre as fontes: cada
canal projetado entra com peso fixo 1 e é a TCN que aprende a combiná-los. O
`sam_score` do SAM, gravado fora de `v_t`, não entra no modelo.

## Receita idêntica ao braço A e ao B0

`C0_FUSION_CONFIG` (`src/gatefall/train/c0_config.py`) copia campo a campo a
receita compartilhada de `BASELINE_A_CONFIG` (seed, janela, strides, número
de classes, TCN, otimizador, agenda, épocas, perda). `c0_config_selftest.py`
verifica essa igualdade e verifica também que, entre os campos que C0 e B0
têm em comum, só `run_name`, `arm` e `visual_dim` (10 vs. 1536) divergem —
invariante experimental 1 do `CLAUDE.md`. O loop de treino
(`c0_engine.py`) reutiliza `configure_determinism` do braço A.

Campos de auditoria exclusivos do C0: `pose_dim`, `visual_dim`,
`projection_dim`, `fused_dim`, caminho/sha256 das estatísticas de pose e das
[estatísticas SAM 3](../data/sam3-standardization.md)
(`visual_standardization_stats_*`), `sam3_features_path`,
`sam3_features_sha256` (digest do conjunto de `.h5`) e `sam3_provenance`
(proveniência comum dos `.h5`: modelo, prompt, hashes do checkpoint e do
lock do runtime, revisão upstream, dtype de autocast e `target_fps`).
`report` exige que o run persistido coincida com esses valores recalculados,
aceitando divergência só em `seed` e `trainable_param_count`.

## Alinhamento temporal e proveniência antes do treino

`FusionWindowDataset` (`src/gatefall/data/fusion_dataset.py`) é o mesmo
dataset do B0, com `visual_dim = 10`: pose e `V_t` são recortados com os
mesmos índices causais (`window_frame_indices`) para cada `(video_id, k_end)`,
e o dataset recusa qualquer vídeo cujo `K` de pose ou de `V_t` divirja do
`n_frames` de `frames.parquet` ou cuja largura não seja 134/10.

Antes de montar qualquer janela, `train` e `report` validam todos os `.h5` do
SAM 3 (`gatefall.sam3.features`): identidade (`video_id`, `K` e `split`),
estrutura, proveniência obrigatória bem formada, homogeneidade entre vídeos e
o contrato congelado da extração (prompt `"person"`, `facebook/sam3`,
10 fps). Em seguida conferem o layout e o frescor das estatísticas de pose e
SAM 3 — este último contra `frames.parquet` e contra o digest atual dos
`.h5`. Detalhes em [Padronização do descritor
SAM 3](../data/sam3-standardization.md).

## Backbone SAM 3 congelado e offline

C0 nunca importa `gatefall.sam3.runtime` para inferência nem sobe o
sub-projeto `sam3_runtime/`: treino e report leem apenas os `V_t` já
persistidos. Os únicos parâmetros treináveis são `E_P`, `E_V`, a TCN e a
cabeça (`CLAUDE.md`, invariante 4). O schema persistido do SAM 3 e o
contrato quadro a quadro de extração/seleção de `"person"` não mudam.

## Somente Le2i CS

`gatefall.train.c0_fusion` aceita apenas `--dataset le2i`
(`SAM3_SUPPORTED_DATASET_IDENTIFIERS`), pelo mesmo motivo das demais
operações do SAM 3.

## `run_dir` isolado

O destino padrão é `runs/local/le2i/c0_fusion/`. Antes de tocar no
`run_dir`, `train` e `report` recusam qualquer `--run-dir` igual, ancestral
ou descendente dos runs canônicos de A (`baseline_a`), B0 (`b0_fusion`) e B1
(`b1_adaptive_gate`), ancorados na raiz do repositório — um `--force` do C0
nunca pode substituir um run de comparação. `validate_local_run_dir` recusa
ainda `runs/reference/` e a árvore local do `le2i-cv`. O `--output` do
`report` não pode apontar para artefatos protegidos (`config.yaml`,
`metrics.json`, `checkpoint.pt`, `alarm_protocol.yaml`,
`event_metrics.json`) do run pedido, do run canônico do C0 nem dos runs de A,
B0 e B1, nem para dentro de `runs/reference/`.

## Como executar

```bash
uv run python -m gatefall.train.c0_fusion selftest
```

Roda checagens sintéticas da leitura validada de `V_t`, da padronização SAM
3, do dataset fundido, do modelo, da configuração compartilhada, do loop de
treino, dos validadores de artefato e das guardas de CLI, sem dataset real.

```bash
uv run python -m gatefall.features.standardize_sam3 build --dataset le2i
uv run python -m gatefall.train.c0_fusion train --dataset le2i \
  --run-dir runs/local/le2i/c0_fusion
```

Treina C0 sobre o Le2i real, com o mesmo lifecycle de staging/promoção do
braço A e do B0 (diretório temporário irmão, validação antes de publicar,
`--force` com backup e rollback).

```bash
uv run python -m gatefall.train.c0_fusion report --dataset le2i \
  --run-dir runs/local/le2i/c0_fusion \
  --output runs/local/le2i/c0_fusion/classification_report.json --force
```

Gera o mesmo diagnóstico de classificação do B0 a partir de um run C0 já
treinado, sem alterar nenhum artefato existente.

## Limitação conhecida: sem identidade de quadro por linha

Como no B0, nenhum dos HDF5 persiste um identificador de quadro por linha. O
alinhamento entre pose e `V_t` repousa em ambos os extratores preservarem a
ordem contígua de `frame_index` de `frames.parquet`; as checagens acima
cobrem `video_id`, `split` e `K`, não uma reordenação interna de linhas.

## Fora do escopo desta entrega

C1 e qualquer gate adaptativo, um `q_visual` específico do SAM 3 (a fórmula
do DINOv3 não é reutilizada), atenção cruzada, tracking de vídeo do SAM 3,
SAM 3.1, prompts relacionados a queda, bboxes de pose como prompt, avaliação
por eventos do C0 e orquestração em `gatefall.pipeline`.
