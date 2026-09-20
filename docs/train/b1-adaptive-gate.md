# Treino — Arma B1 (fusão adaptativa por gate)

`src/gatefall/train/b1_*.py` implementa a arma B1 do braço B: as mesmas
projeções de pose (134-d) e DINOv3 (1536-d) da [arma B0](b0-fusion.md), mas
ponderadas por um **gate escalar aprendido por timestep** antes da
concatenação, seguidas da mesma TCN causal dilatada do [Braço A](baseline-a.md).
B0 permanece intocada: B1 é uma arma irmã, com dataset, modelo, config,
engine, validadores e `run_dir` próprios.

## Gate adaptativo

`B1AdaptiveGateClassifier` (`src/gatefall/train/b1_model.py`):

- `E_P`: `Linear(134, 128) -> LayerNorm(128) -> ReLU`, projeta a pose.
- `E_V`: `Linear(1536, 128) -> LayerNorm(128) -> ReLU`, projeta o DINOv3.
- `AdaptiveGate`: a parametrização treinável mínima
  `g_t = sigmoid(Linear([q_pose_t, q_visual_t]))`, isto é, uma única
  `nn.Linear(2, 1)` — dois pesos e um bias — aplicada ao par de qualidade do
  quadro `t`.
- Fusão: `concat(g_t · E_P(pose_t), (1 - g_t) · E_V(visual_t))`. `g_t` é o peso
  da **pose** e `1 - g_t` o peso do **visual**; os dois somam exatamente `1.0`
  em `float32`.
- O vetor fundido continua com **256 dimensões** por timestep: o gate pondera,
  não redimensiona. A TCN e a cabeça linear many-to-one sobre o último timestep
  são idênticas às do braço A.

O gate é escalar (um valor por timestep), não um vetor por canal: não há
ablação de gate vetorial nesta entrega.

### Qualidade como proxy operacional, não confiança

`q_pose` e `q_visual` são proxies operacionais de qualidade em `[0, 1]` — ver
[Índice de qualidade de pose](../data/pose-quality.md) e [Índice de qualidade
visual DINOv3](../data/dinov3-quality.md). **Não** são probabilidades
calibradas nem escores de confiança, e o mecanismo é descrito como fusão
adaptativa, não como gating por confiança. Coerentemente, a entrada do gate
**não é padronizada**: `q_pose` e `q_visual` entram crus, na escala em que
foram definidos.

### Alinhamento temporal e causalidade

Os dois sinais são lidos dos sidecars `[K, 2]` descritos em [Features de
qualidade](../data/quality-features.md), na mesma grade temporal de
`frames.parquet` usada por pose e DINOv3.
`GatedFusionWindowDataset` (`src/gatefall/data/gated_fusion_dataset.py`) calcula
`window_frame_indices(k_end, n_frames)` **uma vez** e corta as três fontes com
esse mesmo vetor de índices: o alinhamento quadro a quadro e a replicação de
borda no início do vídeo são estruturais, não verificados a posteriori. Nenhum
quadro posterior a `k_end` entra na janela, e nenhum dos dois sinais olha para
o futuro.

## Receita de treino idêntica a A e B0

`src/gatefall/train/b1_config.py` monta `B1_ADAPTIVE_GATE_CONFIG` copiando,
campo a campo, todos os campos da receita compartilhada de `BASELINE_A_CONFIG`
(seed, `window_frames`, `train_stride`, `eval_stride`, `num_classes`,
`kernel_size`, `dilations`, `channels`, `dropout`, `receptive_field`,
`optimizer_name`, `lr`, `weight_decay`, `grad_clip_norm`, `lr_schedule_name`,
`batch_size`, `epochs`, `loss_name`, `class_weighted`). `b1_config_selftest.py`
verifica mecanicamente essa igualdade e, além dela, a paridade campo a campo
com `B0_FUSION_CONFIG`: fora de `run_name` e `arm`, a única diferença admitida
são os campos exclusivos do gate (`gate_input_dim=2`, `gate_output_dim=1`,
`gate_activation="sigmoid"`, `gate_weighted_branch="pose"`) e o par
`quality_features_path`/`quality_features_sha256`. `fused_dim` permanece 256.

Isso é a aplicação direta do invariante experimental 1 (`CLAUDE.md`): A, B0, B1
e C só podem diferir no vetor de feature por timestep. Ver ["Receita de treino
congelada"](baseline-a.md#receita-de-treino-congelada) no braço A para o
detalhamento de cada hiperparâmetro.

`src/gatefall/train/b1_engine.py` reutiliza `configure_determinism` do braço A
(`train/engine.py`) e o mesmo `torch.Generator` semeado do DataLoader de treino,
em vez de reimplementar determinismo. O selftest do engine trava que dois
treinos B1 com a mesma seed produzem checkpoints com o mesmo SHA-256.

O gate é construído em posição fixa no `__init__`, depois de `E_P`/`E_V` e antes
da TCN, para que o consumo de RNG na inicialização permaneça estável entre
execuções.

## Backbones congelados e features offline

B1 nunca importa `gatefall.dinov3.backbone` nem nenhum extrator: treino e report
leem apenas features de pose, features DINOv3 e sidecars de qualidade já
persistidos. Os únicos parâmetros treináveis são `E_P`, `E_V`, o gate, a TCN e a
cabeça de classificação (`CLAUDE.md`, invariante 4).

## Somente Le2i CS

Como B0, `gatefall.train.b1_gate` aceita apenas `--dataset le2i`
(`DINOV3_SUPPORTED_DATASET_IDENTIFIERS = ("le2i",)`). B1 não tem suporte a
`le2i-cv` nesta entrega.

## `run_dir` irmão de A e B0, nunca dentro deles

`default_run_dir_for_arm(dataset, "b1_adaptive_gate")` resolve o destino padrão
para `runs/local/le2i/b1_adaptive_gate/`, irmão de `baseline_a/` e de
`b0_fusion/`. `run_train` e `run_report` aplicam **três** guardas antes de tocar
no `run_dir`:

- `guard_not_arm_a_run_dir` (`b1_run.py`) rejeita `--run-dir` igual, ancestral
  ou descendente do run do braço A;
- `guard_not_arm_b0_run_dir` (`b1_run.py`) faz o mesmo em relação ao run da arma
  B0 — é essa guarda que impede operacionalmente que um `--force` do B1
  sobrescreva ou renomeie o run de comparação do B0;
- `validate_local_run_dir(run_dir, dataset_name)` rejeita qualquer `--run-dir`
  sob a árvore de runs local do `le2i-cv`.

Cada guarda cobre o que as outras não alcançam. Os run dirs padrão são
resolvidos contra `REPOSITORY_ROOT`, não contra o diretório corrente, de modo
que as guardas continuam valendo quando a CLI roda de outro `cwd`.

As três guardas só conhecem os run dirs **canônicos**. Um run não canônico de
outra arma (por exemplo `runs/local/le2i/b0_fusion_seed7`) passaria por elas, e
por isso `run_b1_training` ainda recusa qualquer `run_dir` cujo `config.yaml`
declare uma `arm` diferente de `B1` — nem `--force` sobrescreve o run de outra
arma. Um run B1 válido continua sendo preservado sem `--force` e reconstruído
com `--force`, como antes.

`report` protege os artefatos (`config.yaml`, `metrics.json`, `checkpoint.pt`,
`alarm_protocol.yaml`, `event_metrics.json`) de quatro run dirs: o run pedido
em `--run-dir`, o run **canônico** do B1 e os run dirs canônicos das armas A e
B0. `--output` apontando para qualquer um deles é recusado, assim como qualquer
caminho sob `runs/reference/`.

Os três run dirs canônicos entram ancorados em `REPOSITORY_ROOT`, não no
diretório corrente. Por isso o run canônico do B1 continua protegido mesmo
quando `--run-dir` aponta para outro run B1 (por exemplo
`runs/local/le2i/b1_adaptive_gate_seed7`) ou quando a CLI roda de outro `cwd` —
casos em que o `resolve()` do run pedido, sozinho, não o cobriria.

## Integridade dos artefatos

`b1_artifacts.py` espelha o validador do B0: exige `config.yaml`,
`metrics.json` e `checkpoint.pt`, confere `config_sha256`/`checkpoint_sha256`,
a consistência de `history`, `final` e das classes restritas, e carrega o
checkpoint com `strict=True`. Consequência direta: um checkpoint do B0 é
**recusado** em um run B1, porque lhe faltam os pesos do gate. `report` valida o
run com `fields_allowed_to_differ = {"seed", "trainable_param_count"}`.

## Como executar

```bash
uv run python -m gatefall.train.b1_gate selftest
```

Roda as checagens sintéticas da montagem da qualidade, do armazenamento dos
sidecars, do `GatedFusionWindowDataset` (alinhamento, causalidade, borda e ordem
de canal), do gate (shape, faixa, complementaridade e direção), da paridade de
configuração com B0, do loop de treino, dos validadores de artefato e das
guardas de CLI — sem treinar nem tocar no dataset real.

```bash
uv run python -m gatefall.features.quality_extract extract-all --dataset le2i
uv run python -m gatefall.train.b1_gate train --dataset le2i \
  --run-dir runs/local/le2i/b1_adaptive_gate
```

Os sidecars de qualidade são pré-requisito do treino. Antes de montar os
splits, `train` valida as duas estatísticas de padronização exatamente como B0 e
calcula o digest do conjunto de sidecars. O lifecycle de staging/promoção do run
(diretório temporário irmão, validação antes de publicar, `--force` com backup e
rollback) é o mesmo do braço A.

```bash
uv run python -m gatefall.train.b1_gate report --dataset le2i \
  --run-dir runs/local/le2i/b1_adaptive_gate \
  --output runs/local/le2i/b1_adaptive_gate/classification_report.json --force
```

Gera o mesmo diagnóstico de classificação do braço A e do B0 (matriz de
confusão, `per_class`, projeção binária `fall`/`fallen`, verificação contra
`metrics.json`, `class_support_table`, `macro_f1_policy`) a partir de um run B1
já treinado, sem alterar nenhum artefato protegido.

## Fora do escopo desta entrega

B1 implementa apenas o gate escalar mínimo sobre `q_pose` e `q_visual`. Não
fazem parte desta entrega: avaliação por protocolo de eventos do B1
(`b1_events`), orquestração de B1 em `gatefall.pipeline`, braço C (SAM 3),
configurações C0/C1, atenção cruzada entre as duas fontes, ablações de gate
vetorial, buscas de hiperparâmetro (inclusive inicialização deliberada do bias
do gate), retuning do protocolo de alarme e suporte a `le2i-cv`.
