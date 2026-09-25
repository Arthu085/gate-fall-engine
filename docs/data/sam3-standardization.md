# Padronização do descritor SAM 3 (`V_t`)

`src/gatefall/features/sam3_standardization.py` aplica um z-score por canal ao
descritor `V_t ∈ R^10` do SAM 3 (ver [Fundação SAM 3](sam3-foundation.md#descritor-v_t)),
com estatísticas ajustadas **apenas** no split de treino — o mesmo esquema da
[padronização DINOv3](dinov3-standardization.md). A estatística pertence à
fonte (`source = "sam3"`), não a uma arma: qualquer arma que consuma `V_t`
reusa o mesmo arquivo `src/gatefall/features/stats/sam3_le2i_cs.json`.

## Ajuste e aplicação

- Janelas de treino em `TRAIN_STRIDE = 4` (as mesmas 5219 janelas do Le2i
  CS usadas pela pose e pelo DINOv3), acumulando soma e soma dos quadrados
  por canal (`mean_std_from_accumulators`, variância populacional).
- Canais com `std < 1e-6` são guardados: `mean = 0`, `std = 1`, então passam
  inalterados em vez de dividir por ~0.
- `apply_standardization` calcula `(x - mean) / std` em `float64` e devolve
  `float32`.
- O descritor ausente continua sendo o vetor zero gravado pela extração (sem
  imputação); a padronização só o desloca junto com os demais quadros.

## Layout e frescor

`validate_stats_layout` recusa estatísticas cujo `feature_dim`, `source`,
`split`, `stride`, `dataset` ou `channel_names` divirjam do descritor atual —
`channel_names` precisa reproduzir exatamente a ordem congelada de
`CHANNEL_NAMES`. `validate_stats_freshness` recusa estatísticas obsoletas em
duas dimensões:

- `frames_hash`: sha256 de `frames.parquet` no momento do ajuste;
- `sam3_features_sha256`: digest do conjunto de `.h5` do SAM 3
  (`gatefall.sam3.features.sam3_set_sha256`, sha256 das linhas
  `<video_id> <sha256 do arquivo>` ordenadas por `video_id`). Uma
  reextração muda `V_t` sem mudar `frames.parquet`, por isso o digest das
  features também entra no frescor.

## Validação das features antes do ajuste

`build` e `report` só leem `V_t` via `gatefall.sam3.features`, que nunca sobe
o runtime do SAM 3 nem altera o schema persistido. Antes de montar qualquer
janela, todos os `.h5` do Le2i são conferidos:

- `load_v_t`: o atributo `video_id` do arquivo é o vídeo pedido, `v_t` tem
  shape `[K, 10]` `float32` com `K` igual ao atributo `K`, `sam_score` e
  `n_instances` existem com shape e dtype corretos, e os valores são finitos;
- `collect_sam3_provenance`: o atributo `split` bate com `frames.parquet`, os
  atributos obrigatórios de proveniência não estão ausentes nem malformados,
  a proveniência é idêntica em todos os `.h5` (inclusive o
  `sam3_inference_autocast_dtype`, que não pode misturar FP16 e BF16) e o
  conjunto respeita o contrato congelado da extração: `model_name =
  facebook/sam3`, `text_prompt = "person"` e `target_fps = 10.0`.

## Somente Le2i CS

Assim como as demais operações do SAM 3, `standardize_sam3` aceita apenas
`--dataset le2i` (`ensure_sam3_dataset_supported`).

## Como executar

```bash
uv run python -m gatefall.features.standardize_sam3 selftest
```

Testa layout, ajuste só no treino (sem vazamento), guarda de canal constante,
round-trip de persistência e frescor contra `frames.parquet` e o conjunto de
`.h5`, sem tocar no dataset real.

```bash
uv run python -m gatefall.features.standardize_sam3 build [--dataset le2i] [--force]
```

Valida os `.h5` do SAM 3, calcula as estatísticas do treino e grava
`src/gatefall/features/stats/sam3_le2i_cs.json`; preserva o arquivo existente
sem `--force`.

```bash
uv run python -m gatefall.features.standardize_sam3 report [--dataset le2i]
```

Revalida as estatísticas persistidas contra o layout, `frames.parquet`, o
digest atual dos `.h5`, a contagem de janelas de treino, `mean≈0`/`std≈1` no
treino padronizado e a ausência de valores não finitos em `val`/`test`.
