# Padronização de features de pose

`src/gatefall/features/standardization.py` calcula estatísticas de z-score
por dimensão para o vetor de 134 features de pose descrito em [Contrato
temporal](temporal-contract.md#dataset-de-janelas-de-pose), e
`src/gatefall/features/standardize.py` é a CLI fina (`build`, `selftest`,
`report`) sobre essa lógica. Nenhuma padronização acontece dentro de
`PoseWindowDataset` — a fatia de janela que ele devolve continua crua; quem
consome o dataset para treino aplica `apply_standardization` depois.

A CLI recebe o dataset pelo adapter, usa `pose_root` para carregar os HDF5 e
`pose_stats_path` para localizar o JSON, e injeta a grade e o loader de
features; o núcleo genérico não conhece caminhos do Le2i. A dimensão 134 vem
do schema de pose em `gatefall.pose.kinematics.POSE_FEATURE_DIM`, não do
adapter. A mesma instância da fonte de janelas de treino é reutilizada para
acumulação e diagnósticos, evitando carregar/construir o dataset duas vezes.

## Só no split de treino

Média e desvio-padrão são calculados exclusivamente sobre janelas do split
`train`, em `TRAIN_STRIDE`. As mesmas estatísticas são aplicadas sem
recálculo a `val` e `test` — vazar estatística de validação/teste para a
normalização inflaria a métrica de generalização, do mesmo jeito que o
invariante de split por vídeo (ver `CLAUDE.md`) proíbe vazar janela.

Contagem, soma e soma dos quadrados são acumuladas em `float64` iterando
`PoseWindowDataset` janela a janela — nunca materializando o array denso
`[N, 24, 134]` inteiro em memória. Desvio-padrão é populacional (`ddof=0`),
sem clipping por percentil: a cauda de blocos como velocidade e aceleração
(picos em torno de quedas) é sinal, não ruído a ser cortado.

## Estatística em `TRAIN_STRIDE`, não por quadro único

A acumulação percorre janelas em `TRAIN_STRIDE`, não quadros únicos da
grade. Como janelas consecutivas se sobrepõem (`TRAIN_STRIDE < WINDOW_FRAMES`),
cada quadro entra na estatística uma vez por janela da qual faz parte — logo
vídeos e trechos que geram mais janelas (vídeos mais longos, e o padding por
replicação de borda no início de cada vídeo) pesam proporcionalmente mais na
média e no desvio do que um quadro de um trecho coberto por menos janelas.
Isso é intencional: a estatística de padronização espelha a mesma
distribuição de entradas que o encoder temporal vai realmente consumir no
treino, não uma distribuição uniforme por quadro da grade.

## Congelada por fonte de feature, não por braço

O arquivo de estatísticas é identificado por fonte de feature (`pose`), não
por braço A/B/C. Os três braços do experimento usam janela,
split, seed, encoder temporal e número de épocas idênticos — só o vetor de
feature por passo muda. Enquanto o vetor de 134 dimensões da fonte `pose`
for o mesmo em todos os braços que o consomem, a padronização é recalculada
uma vez e reutilizada, nunca recomputada por braço.

## `kp_conf` fica de fora

As 17 colunas do bloco `kp_conf` (confiança do YOLO-Pose por keypoint, já em
`[0, 1]`) são excluídas da padronização e passam intactas, byte a byte —
inclusive o `0.0` exato que `impute_missing` grava em quadros sem detecção.
Confiança não é uma magnitude física com escala arbitrária que precise ser
recentralizada; é já uma probabilidade normalizada, e z-scorá-la destruiria
o significado direto de "0 = sem detecção, 1 = detecção certa" que o resto
do pipeline (e qualquer inspeção manual) depende.

O intervalo de colunas de `kp_conf` nunca é hardcoded: é derivado de
`gatefall.pose.kinematics.feature_blocks()`, a mesma fonte de verdade que
`_BLOCKS` usa para descrever o layout de 134 colunas. Se o layout de blocos
mudar, a exclusão acompanha automaticamente.

## Guarda de dimensão degenerada

Se o desvio-padrão bruto de uma dimensão (fora de `kp_conf`) for menor que
`1e-6`, a dimensão é tratada como constante: `std` vira `1.0` e `mean` vira
`0.0`, de forma que a padronização a deixe passar praticamente intacta em
vez de dividir por um número próximo de zero e explodir em valores enormes
ou não finitos. O relatório (`report`) lista quantas dimensões foram
guardadas dessa forma e seus nomes; no snapshot atual do Le2i, nenhuma
dimensão é guardada.

## Por que o JSON é commitado

`src/gatefall/features/stats/pose_le2i_cs.json` fica versionado no Git, e
não em `data/`, que é git-ignored. As estatísticas de padronização são parte
da receita congelada compartilhada pelos três braços do experimento (ver
"Congelada por fonte de feature" acima) — precisam ser reproduzíveis
byte a byte entre execuções e entre máquinas sem depender de recomputar o
dataset real, do mesmo jeito que o código de `kinematics.py` está
versionado. Tratar como artefato derivado descartável (como os PNGs de
`data/scratch/`) quebraria essa garantia: duas pessoas rodando o mesmo
commit sem o mesmo `data/` obteriam z-scores diferentes.

## Estrutura do JSON

Cada arquivo persiste: fonte (`pose`), split (`train`), `TARGET_FPS`,
`WINDOW_FRAMES`, stride, número de janelas usadas, dimensão do vetor de
feature, nomes de feature na ordem do vetor, a máscara booleana de exclusão
de `kp_conf` (comprimento 134), `mean` e `std` por dimensão, a contagem e a
máscara de dimensões guardadas, e o hash SHA-256 de
`data/processed/le2i/frames.parquet` no momento em que as estatísticas foram
calculadas — usado por `report` para detectar se o parquet mudou depois do
último `build`.

A gravação é atômica (escreve em `.tmp` e usa `os.replace`) e, depois do
`replace`, relê o arquivo do disco e compara com o conteúdo em memória antes
de considerar a gravação bem-sucedida.

## Como executar

```bash
uv run python -m gatefall.features.standardize selftest [--dataset le2i]
```

Roda checagens sintéticas, sem tocar no dataset real: estatísticas sintéticas
consistentes com `gatefall.pose.kinematics` são aceitas, enquanto uma cópia
com `feature_names` ou `stride` divergentes é rejeitada; entrada conhecida gera
média 0 / desvio 1 nas dimensões padronizadas; as 17 colunas de `kp_conf`
saem byte a byte idênticas à entrada; a máscara de exclusão tem comprimento
134 e cobre exatamente `kp_conf`; uma dimensão constante é guardada sem
gerar `inf`/`NaN`; salvar e recarregar o JSON dá round-trip bit-idêntico; e a
acumulação em streaming bate com o cálculo em lote sobre os mesmos dados
sintéticos.

```bash
uv run python -m gatefall.features.standardize build [--dataset le2i] [--force]
```

Calcula as estatísticas sobre o `train` real do Le2i e grava em
`src/gatefall/features/stats/pose_le2i_cs.json`. É idempotente: sem
`--force`, uma segunda execução não sobrescreve o arquivo existente.

```bash
uv run python -m gatefall.features.standardize report [--dataset le2i]
```

Carrega o JSON persistido (falha se `build` nunca rodou) e roda checagens
fatais sobre o dataset real: antes de qualquer outra checagem, as estatísticas
persistidas são comparadas com o layout de feature vivo em
`gatefall.pose.kinematics` (`feature_names`, `feature_dim`, comprimento da
máscara de exclusão, `stride`, `source`, `split`) — se `_BLOCKS` for
reordenado ou estendido sem recalcular as estatísticas, essa checagem falha
antes de construir ou indexar máscaras dependentes da dimensão, evitando que
um layout obsoleto esconda o erro original com `IndexError`; a
contagem de janelas de treino em
`TRAIN_STRIDE` bate com `EXPECTED_USABLE_WINDOWS_STRIDE4["train"]`; `mean` e
`std` têm shape `[134]` e são finitos; depois de aplicar a padronização, a
média e o desvio por dimensão nas dimensões padronizadas (fora de `kp_conf`
e fora das guardadas) ficam a menos de `1e-3` de 0 e 1, respectivamente;
`val` e `test` não produzem valor não finito após aplicar; `kp_conf` no
treino fica em `[0, 1]` com mínimo exatamente `0.0`; e o hash persistido de
`frames.parquet` bate com o arquivo atual. Também imprime, informativamente:
p50/p99/p99.9/máximo de `|valor padronizado|` por bloco de feature, os
nomes das dimensões guardadas, e quantos vídeos distintos foram carregados
por split (esperado 133/19/38, total 190).
