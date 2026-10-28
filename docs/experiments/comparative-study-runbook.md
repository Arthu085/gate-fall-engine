# Roteiro de execução do estudo 2026

Todos os comandos rodam na raiz do checkout que contém `data/` e `runs/`, com caminhos relativos.
Nenhum passo desta página foi executado com dados reais nesta branch; todos estão **pendentes**.

O plano é a fonte dos comandos. Para listar a sequência exata de uma matriz:

```bash
uv run python -m gatefall.study commands --matrix upfall          # todos os estágios
uv run python -m gatefall.study commands --matrix upfall --stage train
uv run python -m gatefall.study plan                              # contagens e estado das saídas
```

`run` sem `--execute` só imprime. Com `--execute`, o executor:

- roda os passos pendentes em ordem e para na primeira falha;
- reusa saídas existentes depois de verificar seus hashes;
- grava um ledger em `runs/local/study2026/execution/`;
- recusa árvore Git com alterações, salvo com `--allow-dirty`.

## 0. Ambiente e pesos

| Pré-requisito | Verificação |
| --- | --- |
| Dependências | `uv sync --frozen --dev` |
| Pesos YOLO-Pose | `data/scratch/weights/yolo26n-pose.pt` |
| DINOv3 | `data/scratch/dinov3_repo/` e o arquivo de `GATEFALL_DINOV3_WEIGHTS_PATH` (padrão `data/scratch/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`) |
| SAM 3 | runtime em `sam3_runtime/` e o checkpoint de `GATEFALL_SAM3_CHECKPOINT_PATH` (padrão `data/scratch/weights/sam3/sam3_checkpoint.pt`) |
| Contratos | `uv run python -m gatefall.study selftest` |

O SHA-256 dos pesos é calculado na primeira execução (`runs/local/study2026/weights-sha256.json`).

## 1. Dados

Pré-requisito manual: baixar os ZIPs das câmeras 1 e 2 do UP-Fall para
`data/scratch/upfall_archives/` e o GMDCSA-24 para `data/raw/gmdcsa24/GMDCSA24/`. O
Le2i usa os artefatos já existentes (ver [Le2i](../data/le2i.md)).

```bash
uv run python -m gatefall.data.staged stage-archives --archives data/scratch/upfall_archives
uv run python -m gatefall.data.staged fetch-labels --dataset upfall
uv run python -m gatefall.data.staged verify-labels --dataset upfall
uv run python -m gatefall.data.staged manifest --dataset upfall
uv run python -m gatefall.data.staged timegrid --dataset upfall
uv run python -m gatefall.data.staged audit --dataset upfall
uv run python -m gatefall.data.staged fetch-labels --dataset gmdcsa24
uv run python -m gatefall.data.staged verify-labels --dataset gmdcsa24
uv run python -m gatefall.data.staged manifest --dataset gmdcsa24
uv run python -m gatefall.data.staged timegrid --dataset gmdcsa24
uv run python -m gatefall.data.staged audit --dataset gmdcsa24
```

Recursos: rede (Hugging Face) para os rótulos e leitura completa das fontes no `manifest`.

Depois de baixar mais câmeras, rode `stage-archives` de novo. Ele prepara só os ZIPs sem
destino e confere os já preparados. Um destino divergente, ou um `.partial` deixado por uma
execução interrompida, faz o comando recusar tudo sem alterar nada.

Antes de extrair, confira em `coverage.json`:

- ausência de vazamento;
- vistas sincronizadas;
- contagem de `IGNORE` e de `source_gap` por partição;
- número de eventos de queda por partição.

## 2. Extração (GPU)

```bash
for dataset in upfall gmdcsa24; do
  uv run python -m gatefall.pose.extract extract-all --dataset "$dataset"
  uv run python -m gatefall.features.quality_extract extract-all --dataset "$dataset"
  uv run python -m gatefall.dinov3.extract extract-all --dataset "$dataset"
  uv run python -m gatefall.sam3.extract extract-all --dataset "$dataset"
  uv run python -m gatefall.study features-report --dataset "$dataset"
done
```

Sem `--force`, os vídeos já extraídos são pulados. A ordem importa: a qualidade DINOv3 lê a pose.
Siga para os pacotes só quando `features-report` indicar `complete: true` nas quatro
modalidades. O Le2i já está extraído; rode `features-report --dataset le2i` para confirmar.

## 3. Pacotes de features (CPU)

```bash
for dataset in le2i le2i-cv upfall; do
  for visual in none dinov3 sam3; do
    uv run python -m gatefall.experiments.physics prepare --dataset "$dataset" --visual "$visual" \
      --output "runs/local/$(echo $dataset | tr - _)/study2026/bundles/$visual"
  done
done
for visual in none dinov3 sam3; do
  uv run python -m gatefall.experiments.physics prepare --dataset gmdcsa24 --visual "$visual" \
    --output "runs/local/gmdcsa24/study2026/bundles/$visual"
done
```

Cada pacote confere a identidade, o `K`, a partição e a proveniência de cada HDF5, e recusa um
destino existente. O pacote GMDCSA-24 contém só `test`.

## 4. Alvos de movimento e reuso dos runs externos

```bash
for protocol in le2i le2i_cv; do
  for visual in dinov3 sam3; do
    uv run python -m gatefall.experiments.motion prepare \
      --bundle "runs/local/$protocol/study2026/bundles/$visual" --pose-root data/features/le2i/pose \
      --output "runs/local/$protocol/study2026/motion-targets/$visual"
  done
done
uv run python -m gatefall.study reuse-audit --source /caminho/para/runs_local_externo \
  --output runs/local/study2026/reuse-audit.json
```

`--source` aponta para o equivalente a `runs/local` da máquina de origem, por exemplo
`le2i/motion-c1/seed-42/motion_physics/`.

- Copie para o destino planejado só os runs marcados `compatible`.
- Os incompatíveis ficam como evidência histórica fora do estudo.
- O executor revalida os hashes antes de reusar.

## 5. Treino e teste

```bash
uv run python -m gatefall.study run --matrix le2i upfall ablations motion --stage train
uv run python -m gatefall.study run --matrix le2i upfall ablations motion --stage train --execute
uv run python -m gatefall.study run --matrix le2i upfall ablations motion --stage evaluate --execute
```

Os comandos individuais seguem o formato abaixo, gerado pelo plano:

```bash
uv run python -m gatefall.experiments.physics train --bundle runs/local/upfall/study2026/bundles/dinov3 \
  --run-dir runs/local/upfall/study2026/b1-physics/seed-42 --arm B1 --fusion physics --seed 42 \
  --epochs 30 --batch-size 64 --checkpoint-policy best_val
uv run python -m gatefall.experiments.physics evaluate --bundle runs/local/upfall/study2026/bundles/dinov3 \
  --run-dir runs/local/upfall/study2026/b1-physics/seed-42 --split test
uv run python -m gatefall.experiments.motion train --bundle runs/local/le2i/study2026/bundles/sam3 \
  --run-dir runs/local/le2i/motion-c1/seed-42/motion_physics --arm C1 --fusion physics --seed 42 \
  --epochs 30 --batch-size 64 --checkpoint-policy best_val \
  --targets runs/local/le2i/study2026/motion-targets/sam3 --recipe motion_physics
```

Recursos: o treino usa GPU quando disponível e é viável em CPU; a avaliação roda em CPU. Sem
reuso, são 300 treinos únicos, 300 avaliações de teste e 40 avaliações externas.

## 6. Transferência congelada

```bash
uv run python -m gatefall.study run --matrix transfer --stage transfer --execute
```

Formato de cada passo:

```bash
uv run python -m gatefall.experiments.physics evaluate-external \
  --bundle runs/local/upfall/study2026/bundles/sam3 --run-dir runs/local/upfall/study2026/c1-physics/seed-42 \
  --target-bundle runs/local/gmdcsa24/study2026/bundles/sam3
```

## 7. Custo e relatórios

```bash
uv run python -m gatefall.study profile-extraction --dataset upfall --videos 5 --execute
uv run python -m gatefall.study profile-extraction --dataset gmdcsa24 --videos 5 --execute
for matrix in le2i upfall transfer motion ablations; do
  uv run python -m gatefall.study report --matrix "$matrix"
done
```

O perfil extrai N vídeos de teste num diretório temporário, sem tocar no cache de features.
`report` grava em `runs/local/study2026/reports/<matriz>-<utc>/` e recusa matriz incompleta ou
incompatível. Antes de escrever o artigo, revise `summary.md`, `contrasts.csv`, as figuras e as
linhas do tempo.
