# Padronização de features DINOv3

`src/gatefall/features/dinov3_standardization.py` calcula estatísticas de
z-score por dimensão para o vetor de 1536 features DINOv3 descrito em
[Features DINOv3](dinov3-features.md#vetor-de-features-por-quadro), e
`src/gatefall/features/standardize_dinov3.py` é a CLI fina (`build`,
`selftest`, `report`) sobre essa lógica — o mesmo desenho de
[Padronização de pose](pose-standardization.md), reaproveitando
`GUARD_STD_THRESHOLD`, `TRAIN_SPLIT`, `WindowSource` e
`mean_std_from_accumulators` de `features/standardization.py` em vez de
duplicá-los.

Diferente da pose, não há bloco de confiança (`kp_conf`) a excluir: o DINOv3
não expõe um canal de confiança por dimensão, então todas as 1536 dimensões
são padronizadas e não existe máscara de exclusão análoga.

## Só no split de treino

Média e desvio-padrão são calculados exclusivamente sobre janelas do split
`train`, em `TRAIN_STRIDE`, do mesmo jeito que a padronização de pose — sem
vazamento de estatística de validação/teste para a normalização.

## Guarda de dimensão degenerada

Se o desvio-padrão bruto de uma dimensão for menor que `GUARD_STD_THRESHOLD`,
a dimensão é tratada como constante: `std` vira `1.0` e `mean` vira `0.0`,
gravados em `guarded_mask`/`guarded_count`, no mesmo esquema usado pela
padronização de pose.

## Protocolos Le2i CS e CV

`build` e `report` aceitam `--dataset le2i` ou `--dataset le2i-cv`.
Ambos leem os mesmos HDF5 por vídeo. Antes do uso em CV, a CLI confere que
manifesto e grade temporal coincidem com CS, exceto pelo split, e valida
identidade, estrutura e proveniência dos HDF5. A média e o desvio de CV usam
somente as 4168 janelas de treino de CV; nunca reutilizam a estatística CS.
A extração DINOv3 continua restrita a CS.

## Frescor contra `frames.csv`/`frames.parquet`

Cada arquivo de estatísticas persiste `frames_hash`, o hash SHA-256 de
`frames.parquet` no momento do `build`. `validate_stats_freshness` compara
esse hash com o do arquivo de frames atual e aborta com `ValueError` se
divergirem — evita treinar B0 com estatísticas calculadas sobre uma grade
temporal que já mudou. `gatefall.train.baseline_b0` chama essa validação antes
de montar `train`/`val`/`test`, tanto em `train` quanto em `report`.

## Estrutura do JSON

Cada arquivo persiste: fonte (`dinov3`), `dataset` (identificador do
adapter, ex. `le2i`), split (`train`), `TARGET_FPS`, `WINDOW_FRAMES`,
stride, número de janelas usadas, `feature_dim` (1536), `mean` e `std` por
dimensão, a contagem e a máscara de dimensões guardadas, e `frames_hash`. A
gravação é atômica (escreve em `.tmp` e usa `os.replace`) e relê o arquivo
do disco após o `replace`, comparando byte a byte com o conteúdo em memória.

`dinov3_le2i_cs.json` e `dinov3_le2i_cv.json` são versionados separadamente.
Cada arquivo registra o hash da grade do próprio protocolo e o ajuste apenas
no respectivo split de treino.

## Como executar

```bash
uv run python -m gatefall.features.standardize_dinov3 selftest [--dataset {le2i,le2i-cv}]
```

Roda checagens sintéticas, sem tocar no dataset real.

```bash
uv run python -m gatefall.features.standardize_dinov3 build [--dataset {le2i,le2i-cv}] [--force]
```

Calcula as estatísticas sobre o `train` do protocolo escolhido e grava em
`src/gatefall/features/stats/dinov3_le2i_{cs,cv}.json`. Idempotente: sem
`--force`, uma segunda execução não sobrescreve o arquivo existente.

```bash
uv run python -m gatefall.features.standardize_dinov3 report [--dataset {le2i,le2i-cv}]
```

Carrega o JSON persistido (falha se `build` nunca rodou), compara o layout
persistido com o layout DINOv3 atual, valida a contagem de janelas de
treino, checa que a média/desvio pós-padronização ficam a menos de `1e-3` de
0/1 nas dimensões padronizadas, confere ausência de valor não finito em
`val`/`test` após aplicar a padronização, e valida que o `frames_hash`
persistido bate com o arquivo atual. JSON inválido ou layout incompatível
encerra o comando com código 1 e indica o arquivo a reconstruir.
