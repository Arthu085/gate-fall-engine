# Contrato temporal

Esta etapa define a grade de reamostragem temporal do Le2i e o rótulo por
quadro dessa grade. É a base sobre a qual o janelamento (etapa futura, ainda
não implementada) vai operar.

A etapa é somente CPU, tabular, e não grava nada em disco — nenhum `.npy`,
`.h5`, `.parquet` ou `.csv` é produzido. O comando `report` apenas imprime no
stdout, para inspeção manual sobre o manifesto e as anotações já construídos.

## Grade de reamostragem

Para cada vídeo, com `n_frames` quadros contados e `fps_src` medido pelo
`ffprobe` (ver [Manifesto e verificação](manifest-verification.md)):

```
duration_s = n_frames / fps_src
K = floor(duration_s * TARGET_FPS)
times[k] = k / TARGET_FPS,           k = 0, ..., K-1
src_indices[k] = clip(round(times[k] * fps_src), 0, n_frames - 1)
```

`TARGET_FPS = 10.0`. O protocolo de segmentação temporal do OmniFall
(arXiv:2505.19889v1, §4.3) reamostra os datasets staged para 10 fps; usar o
mesmo valor preserva a comparabilidade com os números publicados.

O `floor` em `K` significa que a soma de `K` por vídeo fica sistematicamente
abaixo da projeção ingênua `sum(duration_s) * TARGET_FPS` — cada vídeo perde,
em média, menos de um quadro para o truncamento. Isso é esperado e não indica
perda de dados: os quadros perdidos ficariam além da última amostra completa
da grade daquele vídeo.

`build_time_grid` e `labels_for_grid`, em `src/gatefall/data/resampling.py`,
são agnósticas de dataset — não fazem I/O, não imprimem nada e não chamam
`sys.exit`. A parte específica do Le2i (carregar manifesto e anotações,
montar os DataFrames de relatório) vive em
`src/gatefall/data/le2i/timeline.py`.

## Rótulo por quadro

Os segmentos anotados (`train.csv`, `val.csv`, `test.csv`, unidos) definem o
rótulo de cada `times[k]` sob a convenção de intervalo semiaberto `[start,
end)`: um timestamp pertence ao segmento se `start <= t < end`.

Quando dois segmentos se sobrepõem em um mesmo timestamp da grade, vence o
segmento de menor `start`. `n_overlap_resolved` conta quantos timestamps
foram afetados por essa regra — no snapshot atual, um total pequeno em todo o
dataset (ver saída do `report`).

Nenhuma heurística de encaixe ou preenchimento de lacunas é aplicada nesta
etapa: um timestamp não coberto por nenhum segmento recebe `IGNORE_LABEL`, e
a lacuna não é suavizada nem estendida a partir dos segmentos vizinhos.

## `IGNORE_LABEL`

`IGNORE_LABEL = -1` marca timestamps da grade não cobertos por nenhum
segmento anotado. Isso é uma consequência direta da auditoria de cobertura já
documentada em [Manifesto e verificação](manifest-verification.md#auditoria-de-cobertura):
cerca de 6,72% da duração medida do Le2i não está coberta por nenhum
segmento, concentrada majoritariamente no `leading_gap_s` — o prefixo de sala
vazia antes da primeira anotação, mais presente em `Lecture_room` e `Office`.

O relatório desta etapa reproduz essa decomposição na granularidade de
quadros da grade (contagem de `IGNORE_LABEL` por `leading`/`trailing`/
`interior`, via `gap_position`) e reporta separadamente os casos em que a
lacuna não coberta é menor que a duração de um quadro-fonte do vídeo — sinal
de uma anotação que começa ou termina a poucos milissegundos da borda do
vídeo, e não de um erro do algoritmo de grade.

## Como executar

```bash
uv run python -m gatefall.data.timegrid report
```

Roda sobre o manifesto e as anotações já construídos (ver [Manifesto e
verificação](manifest-verification.md)) e imprime, entre outros: o total de
quadros da grade (`K`) e sua comparação com a projeção ingênua; `K` para os
vídeos mais curtos e mais longos; contagens de quadros por split e por
`(split, label)`; quadros e segundos por classe; o total de sobreposições
resolvidas; a fração de `IGNORE_LABEL` geral, por split e por ambiente, com a
decomposição por `gap_position`; a maior sequência de `IGNORE_LABEL`
consecutivos por ambiente; e a composição dos splits (ambientes e sujeitos).

```bash
uv run python -m gatefall.data.timegrid selftest
```

Verifica `build_time_grid` e `labels_for_grid` contra entradas sintéticas —
sem acessar o dataset real. Cobre a fronteira de arredondamento do `floor` em
`K`, a convenção semiaberta na borda entre dois segmentos, um gap sub-quadro
entre segmentos, a resolução de sobreposição, um gap inicial (`leading`), o
caso `K = 0` e o clamp de `src_indices`. Cada caso imprime uma linha
`PASS`/`FAIL`; o comando termina com código diferente de zero se algum caso
falhar.

## Janelamento

TBD — etapa futura, ainda não implementada nesta versão do repositório.
