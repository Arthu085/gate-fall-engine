# Treino — Arma B0 (fusão pose + DINOv3 por concatenação)

`src/gatefall/train/b0_*.py` implementa a arma B0 do braço B: pose (134-d) e
DINOv3 (1536-d) projetados separadamente e fundidos por concatenação
simples, antes da mesma TCN causal dilatada do [Braço A](baseline-a.md).
`src/gatefall/data/fusion_dataset.py` (`FusionWindowDataset`) devolve, por
janela, o par cru `(pose_window, visual_window, label, (video_id, k_end))` —
a padronização de cada fonte acontece fora do dataset, exatamente como no
braço A: pose via [`apply_standardization`
(pose)](../data/pose-standardization.md), visual via [`apply_standardization`
(DINOv3)](../data/dinov3-standardization.md).

## Arquitetura

`B0FusionClassifier` (`src/gatefall/train/b0_model.py`):

- `E_P`: `Linear(134, 128) -> LayerNorm(128) -> ReLU`, projeta a pose.
- `E_V`: `Linear(1536, 128) -> LayerNorm(128) -> ReLU`, projeta o DINOv3.
- Concatenação de `E_P(pose)` e `E_V(visual)` no eixo de feature, produzindo
  um vetor fundido de 256 dimensões por timestep (`FUSED_DIM = 256`).
- O vetor fundido alimenta a mesma `TCNEncoder` do braço A (não uma cópia
  reimplementada) — `kernel_size`, `dilations`, `channels` e `dropout`
  idênticos, ver "Receita de treino congelada" abaixo — seguida de uma
  cabeça linear many-to-one sobre o último timestep.

Fusão por concatenação simples significa que não há nenhum mecanismo de
atenção, gate ou peso aprendido de confiança entre as duas fontes nesta
entrega — cada canal projetado entra no vetor fundido com peso fixo 1
(implícito pela concatenação), e é a TCN subsequente que aprende a combiná-
los.

## Receita de treino idêntica ao braço A

`src/gatefall/train/b0_config.py` monta `B0_FUSION_CONFIG` copiando, campo a
campo, todos os campos da receita compartilhada de `BASELINE_A_CONFIG`
(seed, `window_frames`, `train_stride`, `eval_stride`, `num_classes`,
`kernel_size`, `dilations`, `channels`, `dropout`, `receptive_field`,
`optimizer_name`, `lr`, `weight_decay`, `grad_clip_norm`,
`lr_schedule_name`, `batch_size`, `epochs`, `loss_name`, `class_weighted`) —
`b0_config_selftest.py` verifica mecanicamente essa igualdade, campo a
campo, contra `BASELINE_A_CONFIG`. Só os campos exclusivos de auditoria da
fusão são específicos do B0: `pose_dim`, `visual_dim`, `projection_dim`,
`fused_dim` e os caminhos/hashes das duas fontes de estatística de
padronização (`pose_standardization_stats_path`/`_sha256`,
`visual_standardization_stats_path`/`_sha256`).

Isso é a aplicação direta do invariante experimental 1 (`CLAUDE.md`): A, B0
e C só podem diferir no vetor de feature por timestep — todo o resto da
receita (janela, split, seed, encoder temporal, épocas) permanece fixo. Ver
["Receita de treino
congelada"](baseline-a.md#receita-de-treino-congelada) no braço A para o
detalhamento de cada hiperparâmetro.

`src/gatefall/train/b0_engine.py` reutiliza `configure_determinism` do
braço A (`train/engine.py`) em vez de duplicá-lo, mantendo as mesmas guardas
de determinismo de GPU (`cudnn.deterministic`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`).

## Backbone DINOv3 congelado e offline

B0 nunca importa `gatefall.dinov3.backbone` nem `gatefall.dinov3.extract`:
treino e report leem apenas features já persistidas em HDF5 via
`gatefall.dinov3.storage.read_features`. Os únicos parâmetros treináveis são
`E_P`, `E_V`, a TCN e a cabeça de classificação — o backbone DINOv3
permanece congelado e não entra no grafo de treino (`CLAUDE.md`, invariante
4).

## Somente Le2i CS

Assim como `standardize_dinov3` (ver [Padronização de features
DINOv3](../data/dinov3-standardization.md#somente-le2i-cs)),
`gatefall.train.b0_fusion` aceita apenas `--dataset le2i`
(`DINOV3_SUPPORTED_DATASET_IDENTIFIERS = ("le2i",)`) — `ensure_dinov3_dataset_supported`
recusa `le2i-cv`, porque os dois adapters do Le2i apontam para o mesmo
`dinov3_root` físico. B0 não tem suporte a `le2i-cv` nesta entrega.

## `run_dir` irmão do braço A, nunca dentro dele

`default_run_dir_for_arm(dataset, "b0_fusion")` resolve o destino padrão
para `runs/local/le2i/b0_fusion/`, irmão de `runs/local/le2i/baseline_a/`
(que continua acessível via `default_run_dir_for_arm(dataset, "baseline_a")`,
reexportado como `default_run_dir` para compatibilidade). `run_train` e
`run_report` aplicam duas guardas complementares antes de tocar no
`run_dir`: `_guard_not_arm_a_run_dir` rejeita qualquer `--run-dir` que seja
igual, ancestral ou descendente do `run_dir` canônico do braço A — fecha o
caminho onde um `--force` de B0 poderia ter sobrescrito ou destruído o run
de referência do braço A — e `validate_local_run_dir(run_dir, dataset_name)`
(a mesma guarda de isolamento entre protocolos do braço A, ver
[Generalização (le2i-cv)](../eval/le2i-cv-generalization.md)) rejeita
qualquer `--run-dir` sob a árvore de runs local do `le2i-cv`. Cada guarda
cobre o que a outra não alcança: a guarda compartilhada não detecta o
próprio diretório do braço A (mesma raiz de protocolo), e a guarda do braço
não detecta a raiz do `le2i-cv`.

## Como executar

```bash
uv run python -m gatefall.train.b0_fusion selftest
```

Roda checagens sintéticas de `FusionWindowDataset`, da padronização DINOv3,
da arquitetura de `B0FusionClassifier`, da configuração compartilhada com o
braço A, do loop de treino/avaliação, dos validadores de artefato e da CLI,
sem treinar nem tocar no dataset real.

```bash
uv run python -m gatefall.train.b0_fusion train --dataset le2i \
  --run-dir runs/local/le2i/b0_fusion
```

Treina B0 sobre o Le2i real. Antes de montar os splits, valida que as
estatísticas de padronização de pose e de DINOv3 batem com o layout de
feature atual e que as estatísticas DINOv3 não estão obsoletas em relação a
`frames.parquet` (ver [Padronização de features
DINOv3](../data/dinov3-standardization.md#frescor-contra-framescsvframesparquet)).
O lifecycle de staging/promoção do run (diretório temporário irmão, validação
antes de publicar, `--force` com backup e rollback) é o mesmo do braço A —
ver ["Como executar" no braço A](baseline-a.md#como-executar).

```bash
uv run python -m gatefall.train.b0_fusion report --dataset le2i \
  --run-dir runs/local/le2i/b0_fusion \
  --output runs/local/le2i/b0_fusion/classification_report.json --force
```

Gera o mesmo diagnóstico de classificação do braço A (matriz de confusão,
`per_class`, projeção binária `fall`/`fallen`, verificação contra
`metrics.json`, `class_support_table`, `macro_f1_policy`) a partir de um run
B0 já treinado, sem alterar nenhum artefato existente — mesmas guardas de
`--output` protegido e somente leitura descritas em ["Diagnóstico de
classificação: `report`"](baseline-a.md#diagnostico-de-classificacao-report)
no braço A.

### Avaliação por eventos

```bash
uv run python -m gatefall.eval.b0_events selftest
```

Roda checagens sintéticas da inferência fundida, das guardas de protocolo e
do lifecycle dos artefatos, sem acessar o dataset real nem um checkpoint.

```bash
uv run python -m gatefall.eval.b0_events evaluate --dataset le2i \
  --run-dir runs/local/le2i/b0_fusion
```

Avalia `val` e `test` de um run B0 completo usando pose e DINOv3 já
extraídos, as duas estatísticas de padronização e o checkpoint treinado. Sem
`--run-dir`, usa `runs/local/le2i/b0_fusion/`. A CLI aceita somente Le2i CS,
rejeita o run do braço A e não oferece suporte a `le2i-cv`.

A avaliação reutiliza `BASELINE_A_ALARM_PROTOCOL`, as mesmas métricas por
evento e por janela e o mesmo lifecycle atômico com lock, journal, staging e
hashes descrito em [Avaliação — Braço A](../eval/baseline-a-events.md). Ela
publica `alarm_protocol.yaml` e `event_metrics.json` no próprio run B0. Sem
`--force`, preserva um par de saídas íntegro e falha diante de artefatos
parciais ou inconsistentes; com `--force`, reconstrói e substitui o par
somente após validar os novos arquivos, com rollback em caso de falha. A
operação não retreina nem retuna o modelo ou o protocolo.

## Limitação conhecida: sem verificação de identidade de quadro

`FusionWindowDataset` valida, por vídeo, que o array de pose e o array
visual têm cada um o número de linhas (`K`) esperado a partir da tabela de
frames, e também que os dois arrays têm o mesmo `K` entre si — qualquer
divergência levanta `ValueError` nomeando o vídeo, a fonte (pose ou visual)
e o `K` esperado vs. observado. Isso não valida identidade de quadro por
quadro: nem o HDF5 do DINOv3 (`dinov3/storage.py`,
[schema](../data/dinov3-features.md#schema-do-hdf5)) nem o HDF5 de pose
(`pose/loading.py`) persistem um identificador de quadro por linha. A
garantia de alinhamento entre as duas fontes repousa inteiramente em ambos
os extratores offline preservarem a mesma ordem contígua de `frame_index` de
`frames.parquet` — se um dos dois extratores algum dia reordenar ou pular
linhas sem preservar essa contiguidade, `FusionWindowDataset` não teria como
detectar o desalinhamento resultante.

## Fora do escopo desta entrega

B0 implementa apenas fusão por concatenação simples. Não fazem parte desta
entrega: gating por confiança entre pose e visual (`q_visual`), braço C
(SAM 3), atenção cruzada entre as duas fontes, orquestração de B0 em
`gatefall.pipeline` e suporte a `le2i-cv` para B0.
