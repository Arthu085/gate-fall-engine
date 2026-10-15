# Análise — Contexto de janela dos erros `other→walk` (Etapa 3A)

`src/gatefall/eval/analysis/window_context_diagnosis.py` executa a
[investigação recomendada](cross-arm-diagnostic-synthesis.md#investigacao-recomendada)
pela síntese entre braços: atribuir, janela a janela, o contexto dos erros
`other→walk` de Le2i-CS em A, B0, B1, C0 e C1, seeds 42–46, em `val` e
`test`. `standing→walk` e `sitting→stand_up` entram só como controles
descritivos.

A análise é **exploratória, não canônica e somente leitura**. Ela não
retreina, não reextrai features, não altera checkpoints, receitas, janelas
nem limiares e não seleciona nem promove nada. O split `test` já estava
exposto e é usado só de forma descritiva; `val` e `test` são relatados
separadamente e nunca somados.

## Conclusão

- **Em `test`, `other→walk` não se concentra em janelas misturadas nem de
  fronteira.** 586 das 860 janelas `other` (68%) são misturadas. Os erros
  caem nelas em proporção próxima à dos acertos (63% a 71% contra 66% a 76%),
  e a taxa de erro em janelas puras (35% a 44%) é próxima da de janelas
  misturadas (35% a 43%). Nas 229 janelas a 23 quadros ou mais do início do
  trecho, em que a janela inteira fica dentro do trecho, a taxa de erro é de
  37% a 51%.
- **Pela leitura definida na síntese, a associação com o contexto da janela
  é fraca.** O erro persiste no meio dos trechos em todos os braços, e o
  próximo passo indicado é a leitura manual dos segmentos `other`,
  começando por `Lecture_room` (puras: 36% a 46% de erro) e pelas janelas
  puras de `coffee_room_02`, que erram mais que as misturadas (53% a 83%
  contra 6% a 23%).
- **`val` não permite responder a pergunta.** As 34 janelas `other` são
  todas misturadas, e `other→walk` tem de 0 a 2,4 erros em média por braço.
- **Os controles se comportam de outro modo.** Em `standing→walk` de
  `test`, os erros estão quase todos em janelas puras, com replicação de
  borda e sem pose no quadro final, o caso de `home_01/video_15` já descrito
  na síntese. Em `sitting→stand_up`, a taxa de erro é maior nas janelas puras
  que nas misturadas nos dois splits.

Nada aqui estabelece causa nem autoriza mudar o janelamento congelado.

## Resultado da execução real

Le2i-CS, cinco braços × seeds 42–46 (25 runs), média ± desvio entre seeds.
Todas as reconciliações com `metrics.json` e todas as checagens de
conservação passaram, sem entradas ausentes.

`test`, `other→walk` (860 janelas `other`: 274 puras, 586 misturadas):

| Braço | Erros (média) | Taxa de erro, puras | Taxa de erro, misturadas | Taxa de erro a 23+ quadros do início | Erros em misturadas | Acertos em misturadas | Seeds com taxa maior em misturadas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 327,4 | 44% ± 5% | 35% ± 2% | 51% | 63% | 76% | 0/5 |
| B0 | 354,0 | 38% ± 4% | 43% ± 4% | 42% | 70% | 66% | 5/5 |
| B1 | 356,4 | 38% ± 4% | 43% ± 5% | 43% | 71% | 66% | 5/5 |
| C0 | 314,0 | 35% ± 4% | 37% ± 0% | 37% | 70% | 71% | 3/5 |
| C1 | 325,6 | 37% ± 10% | 38% ± 3% | 38% | 69% | 72% | 3/5 |

- **Direção mista entre braços.** Em B0 e B1, a taxa em misturadas é maior
  nas cinco seeds, com diferença de 4 a 5 pontos; em A, é menor nas cinco;
  em C0 e C1, a diferença é de 1 a 2 pontos e muda de sinal entre seeds.
- **Ambiente × pureza.** Em `Lecture_room` (491 janelas), a taxa nas puras
  é de 36% a 46% e nas misturadas de 45% a 60%. Em `Coffee_room_02`, as 48
  puras erram de 53% a 83% e as 68 misturadas de 6% a 23%. Em `Office`, as
  puras erram menos (10% a 19%) que as misturadas (22% a 32%).
- **Trecho anterior.** 632 das 860 janelas vêm depois de `walk`, com taxa
  de 38% a 51%. As 27 janelas depois de `IGNORE_LABEL` erram de 65% a 99%,
  mas são poucas para pesar no total.
- **Borda e pose.** Só 50 janelas `other` têm replicação de borda e 11 não
  têm pose no quadro final; elas respondem por 2% a 5% e por 1% dos erros.

`val`, `other→walk`: 34 janelas, todas misturadas, de três trechos em
`Home_01` e `Home_02`. A erra 0 em média; B0, B1, C0 e C1, de 1,8 a 2,4,
todos em `home_02/video_51`. Não há janela pura para comparar.

Controles (`test`): em `standing→walk`, as 29 janelas puras erram de 81% a
100% e as 14 misturadas, de 3% a 40%; de 82% a 98% dos erros têm replicação
de borda e de 73% a 87% não têm pose no quadro final. Em `sitting→stand_up`,
a taxa nas puras é de 8% a 18% e nas misturadas de 5% a 7%. Em `val`, o
padrão de `sitting→stand_up` é o mesmo (puras 25% a 35%, misturadas 14% a
19%), e o de `standing→walk` varia de sinal entre braços.

## O que é reaproveitado

O diagnóstico não tem caminho de inferência próprio. Para cada run, ele usa
as funções do [diagnóstico estratificado](stratified-error-diagnosis.md):

- `validate_run`: braço declarado em `config.yaml`, estatísticas de pose do
  protocolo, SHA-256 de `checkpoint.pt` igual ao `checkpoint_sha256` de
  `metrics.json` e presença de `final.val` e `final.test`.
- `analyze_split_with_table`: inferência pelo avaliador de eventos do braço,
  alinhamento exato de `(video_id, k_end)` com `build_window_index`
  (`eval_stride` 1, sem descartar `IGNORE_LABEL`), conferência do rótulo
  verdadeiro e reconciliação das matrizes por ambiente, sujeito e vídeo com
  `confusion_matrix` de `metrics.json`.

Essa reconciliação é necessária, mas não prova que cada janela recebeu
historicamente o mesmo rótulo: as predições são as da reexecução validada.

## O que é medido

O contexto de cada janela vem só dos rótulos por quadro de `frames.parquet`
e de `person_found` do HDF5 de pose, com os índices de
`window_frame_indices`, inclusive os quadros replicados pela borda. Um
**trecho** é a sequência contígua máxima de quadros com o mesmo rótulo,
`IGNORE_LABEL` inclusive.

| Campo | Definição |
| --- | --- |
| `n_final_label_frames`, `final_label_fraction` | Quantos dos 24 índices da janela têm o rótulo do quadro final, e a fração correspondente |
| `purity` | `pure` se os 24 índices têm o rótulo final; `mixed` caso contrário. Quadros `IGNORE_LABEL` dentro da janela a tornam misturada |
| `preceding_label` | Rótulo do trecho anterior ao do quadro final; `video_start` se o trecho começa no quadro 0 |
| `frames_since_segment_start`, `frames_to_segment_end` | Distância, em quadros da grade de 10 fps, do quadro final ao início e ao fim do trecho anotado |
| `start_distance`, `end_distance` | As mesmas distâncias em faixas `00-05`, `06-11`, `12-22` e `23+`; em `23+` a janela inteira fica do mesmo lado da fronteira |
| `n_replicated_frames`, `edge_padding` | Índices repetidos pela replicação de borda (`max(0, 23 − k_end)`) e se há algum |
| `pose_final` | `person_found` no quadro final (`detected` ou `missing`) |

Em janelas rotuladas, `mixed` equivale a cruzar a fronteira inicial do
trecho, porque o trecho anterior tem por definição outro rótulo. As faixas
de distância descrevem a anotação, não o conteúdo dos quadros.

Para cada par `verdadeiro→predito`, as janelas rotuladas com o rótulo
verdadeiro são divididas em **erro** (o par), **acerto** e **outro erro**.
Os estratos são `purity`, `start_distance`, `end_distance`,
`preceding_label`, `edge_padding`, `pose_final`, `environment`, `video`,
`environment_purity` e `video_purity`. Em cada estrato são relatados
`n_true`, `n_error`, `n_correct`, `n_other_outcome`, a taxa de erro do
estrato (`error_rate`) e as frações dos erros e dos acertos do par que caem
nele (`error_share`, `correct_share`). O perfil de erros e de acertos traz a
fração em janelas misturadas, a fração média do rótulo final, a fração com
replicação de borda, a fração sem pose no quadro final e as medianas das
distâncias às fronteiras.

**Conservação.** Em todo estrato de toda dimensão, as somas de `n_true`,
`n_error`, `n_correct` e `n_other_outcome` precisam reproduzir a linha do
rótulo verdadeiro e a célula do par em `confusion_matrix` de
`metrics.json`; qualquer diferença faz o comando falhar.

**Agregação.** A unidade agregada é a seed de treino: média, desvio, mínimo
e máximo são calculados sobre os valores por seed de cada braço, e janelas
nunca são somadas entre seeds nem entre braços. O suporte por estrato não
depende do modelo e precisa ser idêntico em todos os runs. A consistência
entre runs conta em quantos dos runs a fração de erros em janelas
misturadas supera a de acertos e em quantos a taxa de erro em janelas
misturadas supera a de janelas puras.

## Saídas

`--output-dir` recebe quatro arquivos:

- `window_context_diagnosis.json`: `canonical: false`, definições, `runs`
  com seed, `checkpoint_sha256`, `input_sha256`, reconciliação e, por split
  e par, contagens, perfis, estratos e `conservation`; `arms` com as
  estatísticas entre seeds; `cross_run_consistency`; `missing_inputs`;
  `observations`; `limitations`; e `input_sha256` global, que inclui os HDF5
  de pose de `val` e `test`.
- `window_context_diagnosis.csv`: formato longo com `record` (`run` ou
  `arm_seed_aggregate`), `arm`, `seed`, `source`, `split`, `pair`,
  `dimension`, `stratum`, `metric` e `value`.
- `window_context_windows.csv`: uma linha por janela rotulada de `val` e
  `test`, com o contexto acima e uma coluna `pred_<braço>_seed<seed>` com o
  rótulo predito por run.
- `window_context_diagnosis.md`: resumo com perfis e tabelas por estrato
  para cada split e par, observações, entradas ausentes e limitações.

## Garantias de somente leitura

- Só Le2i-CS é aceito. `--output-dir` não pode coincidir, conter ou estar
  contido em `runs/reference/`, `src/`, `data/`, nos diretórios de frames e
  features ou em qualquer `--run`. O mesmo diretório ou o mesmo par (braço,
  seed) não pode ser fornecido duas vezes.
- Run sem `config.yaml`, `metrics.json`, `checkpoint.pt`, `frames.parquet`
  ou HDF5 de pose é registrado em `missing_inputs` com os arquivos exatos,
  sem resultado inventado; seeds 42–46 não analisadas também.
- O SHA-256 dos arquivos dos runs, de `frames.parquet` e dos HDF5 de pose é
  calculado antes da inferência e conferido antes da gravação. Os arquivos
  são gravados atomicamente e, sem `--force`, nada é sobrescrito.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.window_context_diagnosis selftest
uv run python -m gatefall.eval.analysis.window_context_diagnosis diagnose \
  --run le2i ARMA RUN_DIR [--run le2i ARMA RUN_DIR ...] \
  --output-dir PATH [--force]
```

`selftest` usa só dados sintéticos e cobre janelas puras e misturadas,
quadros replicados, fronteiras de trecho, rótulo do trecho anterior
(inclusive `IGNORE_LABEL` e início do vídeo), replicação de borda, pose no
quadro final, exclusão de janelas `IGNORE_LABEL`, falhas de alinhamento
entre contexto, pose e predições, conservação das contagens, runs, seeds e
pose ausentes e proteção da saída.

Execução com os 25 runs locais de Le2i-CS desta máquina:

```bash
uv run python -m gatefall.eval.analysis.window_context_diagnosis diagnose \
  --output-dir runs/local/window_context_diagnosis/le2i_cs_seeds_42_46 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_42 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_43 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_44 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_45 \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_46 \
  --run le2i B0 runs/local/le2i/b0_fusion \
  --run le2i B0 runs/local/le2i/b0_fusion_seed43 \
  ...
```

B0, B1, C0 e C1 seguem o padrão de B0: seed 42 em `b0_fusion`,
`b1_adaptive_gate`, `c0_fusion` e `c1_adaptive_gate`, e seeds 43–46 com o
sufixo `_seed43` … `_seed46`. São necessários os checkpoints,
`frames.parquet`, os HDF5 de pose e as features de cada braço. Na GTX 1650
desta máquina, os 25 runs levaram cerca de 1 min 45 s.

### Execução registrada (não canônica)

2026-10-15, branch `feat/window-context-diagnostic` (base
`analysis/gatefall-improvements`). Os SHA-256 dos 448 arquivos dos runs, de
`runs/reference/`, de `data/processed/le2i` e dos HDF5 de pose foram iguais
antes e depois da execução. Os relatórios não são versionados:

| Arquivo | SHA-256 |
| --- | --- |
| `window_context_diagnosis.json` | `d7b1b04155497176255c4063a95a230968faa164780a32a9cc3178b5e60b8c39` |
| `window_context_windows.csv` | `b23a0fd0853db5d5e416d3b29640ae9e0d52f13f9e7f8dd37b93d11d475dbb6c` |

Nessa execução, o código veio do worktree do branch e os dados do checkout
principal; por isso, as chaves de `input_sha256` do JSON são caminhos
absolutos.

## Limitações

- Associação não é causa. Fronteiras de trecho coincidem com transições de
  ação, posição no vídeo, replicação de borda e decisões de anotação; uma
  concentração dos erros em janelas misturadas não autoriza mudar o
  janelamento congelado.
- Janelas com `eval_stride` 1 se sobrepõem e não são observações
  independentes; contagens não equivalem a trechos, vídeos ou sujeitos.
- O desvio entre seeds descreve inicialização e otimização, não incerteza
  entre sujeitos.
- O rótulo do trecho anterior e as distâncias descrevem a anotação, não o
  conteúdo dos quadros; o conteúdo dos segmentos `other` não foi inspecionado.
- `pose_final` usa `person_found` bruto; não mede a qualidade da pose, a
  pose imputada nem a entrada visual de B0, B1, C0 e C1.
- Os controles têm suporte pequeno e concentrado em poucos vídeos e não
  testam a hipótese sobre `other→walk`.
- `test` foi exposto pelos diagnósticos anteriores; nada aqui pode ser
  confirmado por ele nem seleciona modelo, seed, limiar ou protocolo.
