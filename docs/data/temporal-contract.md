# Contrato temporal

Esta etapa define a grade de reamostragem temporal do Le2i e o rótulo por
quadro dessa grade. É a base sobre a qual o janelamento (etapa futura, ainda
não implementada) vai operar.

A etapa é somente CPU e tabular. Os comandos `report` e `selftest` não gravam
nada em disco — apenas imprimem no stdout, para inspeção manual sobre o
manifesto e as anotações já construídos. A única exceção é o comando `build`
(ver "Como executar" abaixo), que persiste a grade em
`data/labels/le2i/frames.parquet`.

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
resolvidas; o total de segmentos anotados que não contêm nenhum ponto de
grade (`n_segments_skipped` — falha crítica se maior que zero); a fração de
`IGNORE_LABEL` geral, por split e por ambiente, com a decomposição por
`gap_position`; uma reconciliação entre a contagem de quadros
`IGNORE_LABEL` (convertida em segundos via `TARGET_FPS`) e `gap_s` — o mesmo
fenômeno medido pela varredura de intervalos em segundos — cujo total precisa
bater dentro de uma tolerância relativa de 5% (as parcelas `leading`/
`trailing` podem divergir de propósito por causa da direção de arredondamento
em cada uma, mas só o total é verificado); os quadros `IGNORE_LABEL` cujo gap
é menor que um quadro-fonte, agora com `gap_position` e `gap_length_s` por
linha; a maior sequência de `IGNORE_LABEL` consecutivos por ambiente; a
fração de `IGNORE_LABEL` por `(split, env)`, com o valor observado por split
comparado ao previsto por uma mistura ponderada das taxas de cada ambiente; e
a composição dos splits (ambientes e sujeitos).

Assim como `coverage audit`, `report` agora termina com um código de saída
diferente de zero se alguma checagem crítica falhar (segmentos sem ponto de
grade, ou reconciliação de `IGNORE_LABEL`/`gap_s` acima do limite), imprimindo
`timegrid report OK: nenhuma falha crítica encontrada` no caso de sucesso —
deixou de ser puramente descritivo.

```bash
uv run python -m gatefall.data.timegrid build
```

Grava a grade em `data/labels/le2i/frames.parquet`, uma linha por ponto de
grade, nesta ordem de colunas: `video_id`, `split`, `env`, `subject`,
`frame_index`, `time_s`, `src_index`, `label`, `gap_position`. `subject` não
existe no DataFrame de grade por quadro produzido por `build_grid_frames` — é
unido a partir da tabela por vídeo (`per_video`) antes da gravação.

Três colunas do relatório não entram no artefato persistido: `is_ignore` é
redundante (derivável de `label`) e `gap_length_s`/`frame_duration_s` são
diagnósticos que só fazem sentido para o `report`, não para o consumo
posterior da grade. `gap_position` é mantida deliberadamente: mais adiante ela
permite cruzar, via join em vez de recomputação, os quadros de gap `leading`
com os quadros em que o YOLO-Pose não retorna nenhuma detecção.

O índice de janela não é persistido. Ele depende do stride escolhido, que
difere entre treino e avaliação — persistir um índice de janela criaria dois
artefatos (este parquet, mais o esquema de janelamento futuro) que
precisariam ser mantidos em concordância. A etapa de janelamento (ainda não
implementada) recomputa as fronteiras de janela sob demanda a partir de
`frame_index` e do stride escolhido.

A tabela é ordenada de forma determinística por `(video_id, frame_index)`
antes da gravação, e a gravação é atômica (escreve em um arquivo temporário e
faz `os.replace`), de modo que uma execução interrompida nunca deixa um
parquet pela metade. Após gravar, o comando relê o arquivo do disco e verifica
que o DataFrame relido é idêntico ao gravado (valores e dtypes), encerrando
com código de saída diferente de zero em caso de divergência. Por fim,
imprime um resumo: caminho, tamanho em disco, número de linhas, contagem por
split e um hash de conteúdo (`sha256` sobre `pandas.util.hash_pandas_object`,
não sobre os bytes do arquivo — parquet não garante bytes estáveis entre
gravações).

`data/labels/` já está no `.gitignore` (`data/labels/*`), então
`frames.parquet` é um artefato derivado e não versionado — nunca deve ser
commitado, já que é derivado das anotações do OmniFall, licenciadas sob CC
BY-NC-SA.

```bash
uv run python -m gatefall.data.timegrid selftest
```

Verifica `build_time_grid` e `labels_for_grid` contra entradas sintéticas —
sem acessar o dataset real. Cobre a fronteira de arredondamento do `floor` em
`K`, a convenção semiaberta na borda entre dois segmentos, um gap sub-quadro
entre segmentos, a resolução de sobreposição, um gap inicial (`leading`), um
segmento anotado sem nenhum ponto de grade (`n_segments_skipped`), o caso
`K = 0` e o clamp de `src_indices`. Cada caso imprime uma linha `PASS`/`FAIL`;
o comando termina com código diferente de zero se algum caso falhar.

## Janelamento

O janelamento transforma a grade de reamostragem (uma linha por quadro) em
janelas deslizantes: sequências de `WINDOW_FRAMES` quadros consecutivos,
rotuladas pelo rótulo do último quadro da janela. As constantes vivem em
`src/gatefall/config.py`:

- `WINDOW_FRAMES = 24` — 2,4 s em `TARGET_FPS`; cobre o p75 da duração dos
  segmentos `fall` (2,35 s) e dá timesteps suficientes para uma TCN dilatada
  de 3 níveis, kernel 3, campo receptivo 29.
- `TRAIN_STRIDE = 4` — em stride 1, janelas de treino consecutivas se
  sobrepõem em 96% e viram quase-duplicatas.
- `EVAL_STRIDE = 1` — uma predição por quadro da grade, o que dá avaliação
  quadro a quadro e resolução de latência de 0,1 s.

O contrato de janela: cada janela termina em `k_end` e cobre os quadros
`k_end - WINDOW_FRAMES + 1 .. k_end`; para janelas próximas do início do
vídeo, os índices abaixo de 0 são clipados para 0 — isto é replicação de
borda (edge padding), não um caminho de código separado. O rótulo da janela é
o rótulo do seu último quadro. Janelas cujo último quadro é `IGNORE_LABEL`
são descartadas da loss e das métricas, mas seus quadros continuam contando
como contexto de entrada para outras janelas que os incluam fora da posição
final.

`src/gatefall/data/windowing.py` é puro e agnóstico de dataset — não importa
nada de `gatefall.data.le2i` ou `gatefall.data.omnifall`, e nada nele faz
I/O. Três funções:

- `window_frame_indices(k_end, n_frames, window_frames=WINDOW_FRAMES)`
  retorna os `window_frames` índices de quadro da janela que termina em
  `k_end`, clipados a `[0, n_frames - 1]`.
- `window_end_indices(n_frames, stride)` retorna `0, stride, 2*stride, ...`
  abaixo de `n_frames` — `ceil(n_frames / stride)` valores. O último fim não
  é forçado para `n_frames - 1`: em `EVAL_STRIDE=1` todo quadro já é um fim
  de janela, e em `TRAIN_STRIDE=4` perder no máximo 3 quadros finais por
  vídeo é irrelevante.
- `build_window_index(frames, stride, drop_ignored=True)` consome a tabela
  de quadros genérica (`video_id, split, env, subject, frame_index, time_s,
  src_index, label, gap_position`), agrupa por `video_id` e devolve um
  DataFrame com uma linha por janela: `video_id, split, env, subject, k_end,
  label, n_frames`. `n_frames` é o `K` daquele vídeo, necessário para quem só
  tem o índice de janela reconstruir os `WINDOW_FRAMES` índices de quadro a
  partir de `k_end`. Janelas nunca cruzam fronteira de vídeo.

Nesta etapa não há persistência nem relatório sobre o dataset real — como já
observado acima, o índice de janela depende do stride escolhido (diferente
entre treino e avaliação) e é recomputado sob demanda a partir de
`frame_index`, nunca gravado em disco.

```bash
uv run python -m gatefall.data.windows selftest
```

Verifica `window_frame_indices`, `window_end_indices` e `build_window_index`
contra entradas sintéticas — sem acessar o dataset real. Cobre janelas
totalmente preenchidas por padding, parcialmente preenchidas e sem padding, o
caso de vídeo de um único quadro, o encadeamento de janelas com stride 1 em
um vídeo curto, as fronteiras de `window_end_indices` para diferentes
strides, a ausência de mistura de `video_id` entre janelas, a correspondência
entre o rótulo da janela e o rótulo do quadro final, o comportamento de
`drop_ignored` e a contagem de janelas por vídeo antes do descarte. Cada caso
imprime uma linha `PASS`/`FAIL`; o comando termina com código diferente de
zero se algum caso falhar.

```bash
uv run python -m gatefall.data.windows report
```

Diferente de `selftest`, lê `data/labels/le2i/frames.parquet` (não reconstrói
a grade a partir do manifesto e das anotações do OmniFall) e relata as
contagens reais de janela sobre o dataset inteiro — o tamanho real do
problema de aprendizado, não uma estimativa. Termina com erro claro e código
de saída diferente de zero se o parquet não existir, indicando para rodar
`uv run python -m gatefall.data.timegrid build` primeiro. A lógica específica
do Le2i vive em `src/gatefall/data/le2i/windows.py`; `windows.py` em
`gatefall/data/` continua sendo apenas a CLI fina, do mesmo jeito que
`timegrid.py`.

Para `TRAIN_STRIDE=4` e `EVAL_STRIDE=1`, imprime: o total de janelas por
split antes de descartar as de quadro final `IGNORE_LABEL`; as janelas úteis
por split depois do descarte, com o percentual descartado; as janelas úteis
por `(split, label)` — o suporte por classe que precisa acompanhar todo F1 na
tese; e as janelas úteis por `(split, env)`. Apenas em `TRAIN_STRIDE`, imprime
dois diagnósticos adicionais por split: a fração de janelas úteis cujo
contexto de 24 quadros (via `window_frame_indices`, sem reimplementar a
expansão) contém ao menos um quadro `IGNORE_LABEL` não anotado; e a fração de
janelas úteis com ao menos um quadro de edge padding (`k_end < WINDOW_FRAMES -
1`) — o custo aceito ao escolher replicação de borda em vez de descartar os
23 primeiros quadros de cada vídeo.

Por fim, roda quatro checagens críticas, cada uma impressa como PASS/FAIL,
com código de saída diferente de zero se qualquer uma falhar — comparações
exatas, não tolerâncias: a contagem de janelas por split em `stride=1` com
`drop_ignored=False` bate com a contagem de quadros da grade por split (train
22246, val 2080, test 6168); a mesma contagem com `drop_ignored=True` bate com
quadros da grade menos quadros `IGNORE_LABEL` (train 20740, val 2079, test
5616); para todo vídeo e ambos os strides, a contagem de janelas antes do
descarte é `ceil(K / stride)`; e nenhuma janela tem `k_end >= n_frames`. Os
seis valores esperados são propriedades do dataset congelado — se o parquet
mudar, queremos uma falha ruidosa, não um relatório silenciosamente
diferente.
