# Análise — Consolidação multi-seed dos erros estratificados

`src/gatefall/eval/analysis/multiseed_error_consolidation.py` é a terceira
etapa do diagnóstico dos erros de classificação. Ela não refaz inferência:
consome relatórios `stratified_error_diagnosis.json` já gravados pelo
[diagnóstico estratificado](stratified-error-diagnosis.md) e resume, entre as
seeds de treino 42–46, como os erros de A, B0, B1, C0 e C1 variam por
**ambiente** e por **(ambiente, sujeito)** em `val` e `test`, nos protocolos
Le2i-CS (`le2i`) e Le2i-CV (`le2i-cv`).

A análise é **exploratória, não canônica e somente leitura**. Ela não
retreina, não reextrai features, não lê checkpoints, não ajusta
hiperparâmetros e não seleciona nem promove modelos. O split `test` é usado
apenas de forma descritiva; nada aqui escolhe modificação, seed, checkpoint,
limiar ou hiperparâmetro.

## Unidade de agregação

A unidade agregada é a **seed de treino independente**. Para cada protocolo,
arma, split e grupo, as estatísticas `n`/`mean`/`std`/`min`/`max` (de
`_aggregate_stats` do [sumário multi-seed](multiseed-summary.md), com desvio
amostral) são calculadas sobre o valor de cada seed. Contagens de janelas
**nunca são somadas entre seeds**: o suporte é idêntico em todas as seeds de
um protocolo e aparece uma vez. Le2i-CS e Le2i-CV nunca são agregados entre
si, e seeds de armas diferentes não formam pares.

Cada seed traz as predições da reexecução validada da Etapa 1B. A
reconciliação dessa reexecução com `metrics.json` não prova que ela coincide
janela a janela com as predições históricas, e matrizes globais iguais entre
relatórios também não provam identidade por janela.

## Entradas e validação

Cada relatório entra com `--report PATH` (repetível). Um relatório pode
conter várias seeds, como na execução do diagnóstico estratificado com um
`--run` por seed, ou uma só; a partição não altera o resultado. As seeds
esperadas são `EXPECTED_SEEDS` (42–46), ajustáveis com `--seed` repetível.

A consolidação falha, sem gravar nada, se:

- o relatório não for `stratified_error_diagnosis` não canônico e somente
  leitura, ou divergir dos demais em rótulos, `eval_stride`,
  `ignore_label`, splits, níveis ou definição da Macro-F1 restrita;
- uma seed estiver fora do conjunto esperado, a mesma identidade
  `(protocolo, arma, seed)` aparecer duas vezes, ou o mesmo
  `checkpoint_sha256` aparecer em duas identidades;
- a procedência for inválida: protocolo, arma ou `source` divergentes do
  bloco do relatório, `input_sha256` sem o `checkpoint.pt` do `source` com o
  mesmo SHA-256 ou sem exatamente um `frames.parquet`;
- algum split não tiver `reconciliation.ok: true`,
  `prediction_source: validated_rerun` e
  `proves_historical_window_predictions: false`;
- algum grupo for inconsistente com a própria matriz de confusão (Macro-F1
  restrita, suporte, precisão, revocação, F1, `tp`/`fp`/`fn`, janelas,
  erros e classes ausentes são recalculados) ou a soma das matrizes de um
  nível divergir da matriz do split;
- dentro de um protocolo, seeds ou armas usarem `frames.parquet`, grupos ou
  suporte por grupo diferentes.

Seeds ausentes não interrompem a execução: cada arma registra `coverage`
(`seeds`, `missing_seeds`, `complete`), `missing_inputs` lista as armas com
cobertura incompleta ou sem nenhuma seed e as estatísticas usam só as seeds
disponíveis. Entradas ausentes já registradas pela Etapa 1B, exceto
"nenhum run fornecido", também são repassadas.

## Métricas

Para o split inteiro, cada ambiente e cada grupo (ambiente, sujeito):

- **Macro-F1 restrita** e número de erros: média e desvio entre seeds, com o
  valor de cada seed. A definição é a do treino, com as oito classes de
  `RESTRICTED_CLASSES`; classes restritas ausentes no grupo entram com F1 0 e
  geram aviso.
- **Padrões por classe**: suporte, e média e desvio de revocação, precisão e
  F1 entre seeds; o principal destino de erro de cada classe por seed e o
  destino modal. Para classes ausentes no grupo, os falsos positivos por
  seed e as seeds que mesmo assim as predizem.
- **Consistência das confusões dominantes**: a maior confusão de cada seed,
  a confusão modal e, para a confusão dominante da seed 42, em quantas seeds
  ela é a maior, em quantas está entre as cinco maiores e a fração do
  suporte da classe verdadeira em cada seed. O status é `stable` (dominante
  em todas as seeds esperadas), `stable_in_available_seeds` (em todas as
  disponíveis, com cobertura incompleta), `majority`, `unstable` ou
  `inconclusive` (sem confusão na seed 42 ou com menos de duas seeds).
- **Previsão da classe residual**: fração das janelas fora de `other`
  preditas como `other` e fração dos erros que vão para `other`, por seed.

A seed 42 é a referência só porque o smoke da Etapa 1B foi feito nela; ela
não foi escolhida por desempenho.

## Observações e hipóteses sobre B0/B1

`focus_stability`, `observations` e `hypotheses` cobrem B0 e B1 no split
inteiro e por ambiente. As **observações confirmadas** são valores
calculados: Macro-F1 entre seeds ao lado da de A, fração de janelas
preditas como `other` e consistência da confusão dominante da seed 42. Uma
observação só é marcada como sustentada (`supported: true`) quando essa
confusão é a dominante em todas as seeds disponíveis;
`residual_above_comparison_in_all_seeds` indica se a menor fração predita
como `other` entre as seeds da arma supera a maior entre as seeds de A no
mesmo grupo.

As **hipóteses** vêm de regras fixas e não foram testadas: confusão
dominante estável (com leitura própria quando o destino é `other`),
confusão da seed 42 que não se repete em todas as seeds, desvio-padrão da
Macro-F1 acima de `UNSTABLE_SEED_STD` (0,05) e fração de janelas preditas
como `other` maior, em todas as seeds, que em qualquer seed de A. Essas
regras só disparam texto e não filtram resultados.

## Saídas

`--output-dir` recebe três arquivos:

- `multiseed_error_consolidation.json`: `canonical: false`, definição da
  Macro-F1, seeds esperadas, seed de referência, relatórios lidos e, por
  protocolo e arma, `coverage`, `runs` (seed, `run_name`, `source`,
  relatório e `checkpoint_sha256`) e `splits` com os níveis `split`,
  `environment` e `environment_subject`; além de `missing_inputs`,
  `focus_stability`, `observations`, `hypotheses`, `limitations` e o
  SHA-256 de cada relatório.
- `multiseed_error_consolidation.csv`: formato longo com `dataset`,
  `protocol`, `arm`, `split`, `level`, `group`, `env`, `subject`, `entity`,
  `metric`, `n`, `mean`, `std`, `min`, `max` e `value`. Estatísticas entre
  seeds preenchem `n`–`max`; valores fixos, como suporte, contagem de seeds
  e status, usam `value`. `level=coverage` registra seeds presentes e
  ausentes.
- `multiseed_error_consolidation.md`: cobertura de seeds e, por protocolo e
  split, tabelas de Macro-F1 por ambiente e por ambiente × sujeito,
  consistência da confusão dominante, fração predita como `other` e revocação
  por classe, seguidas das observações, hipóteses, entradas ausentes e
  limitações.

## Garantias de somente leitura

- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, no diretório de qualquer `--report` ou
  em qualquer `source` registrado nos relatórios. O mesmo relatório não pode
  ser fornecido duas vezes.
- Os três arquivos são gravados atomicamente. Sem `--force`, se algum deles
  já existir, nada é gravado.
- O SHA-256 de cada relatório é calculado antes da leitura e conferido depois
  da gravação. Checkpoints, métricas e features dos runs não são relidos; a
  procedência de cada seed é a registrada pela Etapa 1B.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.multiseed_error_consolidation selftest
uv run python -m gatefall.eval.analysis.multiseed_error_consolidation consolidate \
  --report PATH [--report PATH ...] [--seed N ...] --output-dir PATH [--force]
```

`selftest` usa somente relatórios sintéticos gerados pelo caminho da Etapa
1B e cobre agregação contra cálculo independente, padrões por classe,
consistência das confusões, separação entre observações sustentadas e
hipóteses, seeds e armas ausentes, suporte, grupos e frames incompatíveis,
identidades de seed e checkpoint duplicadas, procedência e reconciliação
inválidas, relatório adulterado, reprodutibilidade byte a byte e proteção da
saída.

Para os runs locais desta máquina, primeiro gere um relatório estratificado
com as 50 seeds (A, B0, B1, C0 e C1 × seeds 42–46 × dois protocolos; em
Le2i-CS, A usa `runs/local/le2i/baseline_a_multiseed/seed_<seed>` e as
demais armas usam `runs/local/le2i/<run>` para a seed 42 e
`runs/local/le2i/<run>_seed<seed>` para 43–46; em Le2i-CV, a seed 42 fica em
`runs/local/le2i_cv/baseline_<arma>` e as demais em
`runs/local/le2i_cv/multiseed/baseline_<arma>/seed_<seed>`), por exemplo:

```bash
uv run python -m gatefall.eval.analysis.stratified_error_diagnosis diagnose \
  --output-dir runs/local/multiseed_error_consolidation/stratified_seeds_42_46 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_42 \
  --run le2i B0 runs/local/le2i/b0_fusion \
  --run le2i B0 runs/local/le2i/b0_fusion_seed43 \
  ...
uv run python -m gatefall.eval.analysis.multiseed_error_consolidation consolidate \
  --report runs/local/multiseed_error_consolidation/stratified_seeds_42_46/stratified_error_diagnosis.json \
  --output-dir runs/local/multiseed_error_consolidation/consolidated
```

Na GTX 1650 desta máquina, o diagnóstico estratificado dos 50 runs levou
cerca de 4 min, e a consolidação, cerca de 6 s.

## Histórico de execuções (não canônico)

Os resultados abaixo são exploratórios, não foram revisados e não
substituem nenhuma referência. Os relatórios não são versionados; os
comandos acima os reproduzem.

### 2026-10-14 — seeds 42–46, cinco armas, dois protocolos

Executado no branch `feat/multiseed-error-consolidation` (base
`analysis/gatefall-improvements`) sobre os 50 runs locais. Todos os splits
de todos os runs reconciliaram com `metrics.json`, a cobertura ficou
completa (5/5 seeds em todas as armas dos dois protocolos),
`missing_inputs` ficou vazio e o SHA-256 dos 153 arquivos `config.yaml`,
`metrics.json` e `checkpoint.pt` dos runs ficou idêntico antes e depois.

Macro-F1 restrita em `test` por ambiente, média ± desvio entre cinco seeds:

| Protocolo | Ambiente | A | B0 | B1 | C0 | C1 |
| --- | --- | --- | --- | --- | --- | --- |
| Le2i-CS | Coffee_room_01 | 0,589 ± 0,009 | 0,584 ± 0,009 | 0,590 ± 0,025 | 0,586 ± 0,019 | 0,588 ± 0,017 |
| Le2i-CS | Coffee_room_02 | 0,693 ± 0,042 | 0,663 ± 0,029 | 0,690 ± 0,030 | 0,670 ± 0,028 | 0,687 ± 0,060 |
| Le2i-CS | Home_01 | 0,449 ± 0,014 | 0,442 ± 0,027 | 0,452 ± 0,013 | 0,480 ± 0,013 | 0,475 ± 0,012 |
| Le2i-CS | Home_02 | 0,330 ± 0,034 | 0,459 ± 0,016 | 0,480 ± 0,026 | 0,329 ± 0,028 | 0,331 ± 0,025 |
| Le2i-CS | Lecture_room | 0,529 ± 0,039 | 0,421 ± 0,032 | 0,481 ± 0,071 | 0,387 ± 0,012 | 0,405 ± 0,055 |
| Le2i-CS | Office | 0,576 ± 0,017 | 0,642 ± 0,017 | 0,648 ± 0,009 | 0,642 ± 0,020 | 0,599 ± 0,012 |
| Le2i-CV | Home_01 | 0,347 ± 0,019 | 0,228 ± 0,065 | 0,271 ± 0,038 | 0,334 ± 0,010 | 0,344 ± 0,009 |
| Le2i-CV | Home_02 | 0,458 ± 0,033 | 0,220 ± 0,089 | 0,268 ± 0,057 | 0,426 ± 0,030 | 0,439 ± 0,034 |

Observações confirmadas:

- **Le2i-CS, `other→walk` em `Lecture_room`:** é a confusão dominante nas
  cinco seeds de todas as armas, com 43% a 55% do suporte de `other` em
  média (B0 e B1: 55%). `Lecture_room` responde por 30% a 37% dos erros de
  `test` em cada seed. Esse padrão da seed 42 se mantém entre seeds.
- **Le2i-CS, `Coffee_room_02`:** continua com a maior Macro-F1 média em
  `test` nas cinco armas. `Home_02` segue sem `fall`, `fallen` e `standing`,
  e B0/B1 ficam acima de A, C0 e C1 nele (0,46 e 0,48 contra 0,33).
- **Le2i-CS, demais confusões por ambiente:** fora de `Lecture_room` e do
  split inteiro, a confusão dominante da seed 42 de B0/B1 em `test` se
  repete em só 1 a 4 das cinco seeds; essas leituras da Etapa 1B dependem
  da seed.
- **Le2i-CV, deslocamento de B0/B1 para `other`:** em todas as seeds, B0
  prediz `other` para 22% a 80% das janelas de `test` fora de `other`, e B1
  para 26% a 70%, contra 6% a 11% em A e 9% a 13% em C0/C1. O deslocamento
  é reprodutível, mas a magnitude varia muito: a seed 42 é a mais extrema
  de B0 (80%; demais seeds de 22% a 47%), e a Macro-F1 de B0 em `test` vai
  de 0,13 (seed 42) a 0,33.
- **Le2i-CV, números da seed 42 citados na Etapa 1B:** B0 de 0,08 a 0,18 em
  todos os grupos (ambiente, sujeito) passa a médias de 0,10 a 0,28 entre
  seeds. `sitting→other` em `Home_02` é a confusão dominante de B0 em 4 de
  5 seeds, mas a fração do suporte de `sitting` vai de 22% a 81%
  (0,54 ± 0,24); em B1 ela é a dominante em 2 de 5 seeds. Em `Home_01`,
  `sitting→walk` é a dominante em 5 de 5 seeds de C1 e 3 de 5 de A.

Hipóteses não verificadas: em Le2i-CV, o deslocamento de B0/B1 para `other`
é uma propriedade reprodutível dessas configurações, não ruído de
inicialização, mas a escolha da confusão dominante e a intensidade dependem
da seed; a seed 42 superestima a intensidade em B0. Como o ambiente retido
também muda sujeitos, ações e suporte, isso não demonstra efeito causal das
features DINOv3. Em Le2i-CS, `other→walk` em `Lecture_room` é um erro
sistemático compartilhado por todas as armas, o que aponta mais para o
conteúdo do ambiente ou dos rótulos que para o backbone, também sem teste.

## Limitações

- Com cinco seeds por arma não há intervalo de confiança; o desvio-padrão é
  amostral e descritivo.
- As métricas são por janela com `eval_stride` 1; janelas vizinhas se
  sobrepõem e não são observações independentes.
- A Macro-F1 de grupos com classes restritas ausentes é mecanicamente menor
  e não é comparável à do split.
- Os resultados valem para as predições reexecutadas da Etapa 1B; a
  igualdade agregada com `metrics.json` não comprova a identidade das
  predições históricas por janela.
- A consolidação confia na procedência registrada nos relatórios; ela
  confere o SHA-256 dos relatórios, mas não relê checkpoints nem features.
