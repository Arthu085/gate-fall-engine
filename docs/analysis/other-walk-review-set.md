# Análise — Conjunto de revisão dos erros `other→walk` (Etapa 3B)

`src/gatefall/eval/analysis/other_walk_review_set.py` prepara a leitura
manual indicada pela [Etapa 3A](window-context-diagnosis.md): seleciona, de
forma determinística, trechos `other` de Le2i-CS em que o erro
`other→walk` persiste entre braços e seeds, mais controles `other`
corretamente classificados dos mesmos ambientes, e monta folhas de contato
locais para uma revisão visual cega.

A preparação é **exploratória, não canônica e somente leitura**. Ela usa as
predições já gravadas em `window_context_windows.csv` da Etapa 3A, não
refaz inferência, não avalia modelo e não altera anotações, janelamento,
features, checkpoints, treino, métricas nem decisões experimentais. O split
`test` já estava exposto; `val` e `test` são contabilizados e numerados
separadamente.

A revisão das 15 folhas foi concluída depois da preparação; a
[consolidação](#revisao-concluida) separa a leitura cega preliminar das
observações humanas feitas após o desmascaramento.

## Critérios de seleção

| Etapa | Regra |
| --- | --- |
| Persistência | Janela `other` em que, em **todos** os braços (A, B0, B1, C0, C1), pelo menos 80% das seeds preveem `walk` (erro persistente) ou `other` (controle persistente); com cinco seeds, 4 de 5 por braço |
| Ambientes | Só `Lecture_room` e `Coffee_room_02`, os priorizados pela Etapa 3A; os demais entram só na cobertura. Controles só em (split, ambiente) com algum erro selecionado |
| Unidade | Um item por trecho anotado (`video_id`, `segment_start`, `segment_end`) e por papel |
| Representante | Entre as janelas persistentes do trecho: puras primeiro, depois maior fração mínima entre braços, depois maior distância à fronteira mais próxima do trecho, depois menor `k_end` |
| Sobreposição | Duas janelas selecionadas do mesmo vídeo ficam a pelo menos 24 quadros (`WINDOW_FRAMES`) uma da outra; se o representante sobrepõe um item já escolhido, vale o próximo candidato do trecho, e sem candidato o trecho é excluído. Erros são escolhidos antes dos controles |
| Limite | No máximo 8 trechos por (split, ambiente, papel), na ordem do representante |
| Identificador | `review_id` = `<split>-NN`, na ordem do SHA-256 de `<split>/<video_id>/<k_end>`, que não depende do papel |

Antes da seleção, o CSV da Etapa 3A é conferido por inteiro. Ele precisa
ter exatamente uma coluna `pred_<braço>_seed<seed>` para cada braço (A, B0,
B1, C0, C1) e cada seed de 42 a 46; seed ausente, seed fora desse conjunto,
coluna repetida ou seed repetida com outra grafia (por exemplo,
`seed042`) faz o comando falhar, de modo que um conjunto parcial de seeds
nunca satisfaz o critério de 4 de 5. Toda predição precisa ser um rótulo do
Le2i. As janelas do CSV são reconciliadas com as janelas rotuladas de `val`
e `test` que `build_window_index` reconstrói de `frames.parquet` com
`EVAL_STRIDE`, como na Etapa 3A: janela ausente, repetida ou inexistente,
split inesperado e `true_label`, ambiente ou sujeito divergentes fazem o
comando falhar sem gravar nada.

Em cada item selecionado, a janela é reconstruída a partir de
`frames.parquet` com `window_frame_indices`, como na Etapa 3A, e conferida
contra o CSV: `split`, ambiente e sujeito, rótulo `other` no quadro final,
número de quadros com o rótulo final, quadros replicados pela borda e
limites do trecho. Qualquer divergência faz o comando falhar.

## Cobertura da execução real

Le2i-CS, 25 runs (cinco braços × seeds 42–46) da Etapa 3A. Janelas `other`
persistentes e trechos que as contêm:

| Split | Ambiente | Janelas `other` / trechos | Erro persistente: janelas / trechos | Controle persistente: janelas / trechos | Selecionados: erro / controle |
| --- | --- | ---: | ---: | ---: | ---: |
| `val` | Home_01 | 15 / 2 | 0 / 0 | 6 / 2 | — |
| `val` | Home_02 | 19 / 1 | 0 / 0 | 0 / 0 | — |
| `test` | Coffee_room_01 | 43 / 2 | 2 / 1 | 16 / 2 | — |
| `test` | Coffee_room_02 | 116 / 3 | 11 / 1 | 27 / 3 | 1 / 3 |
| `test` | Home_02 | 16 / 1 | 0 / 0 | 0 / 0 | — |
| `test` | Lecture_room | 491 / 17 | 109 / 14 | 43 / 4 | 8 / 3 |
| `test` | Office | 194 / 7 | 15 / 4 | 50 / 4 | — |

- **15 itens, todos em `test`.** Em `Lecture_room`, 6 trechos de erro
  ficaram fora pelo limite de 8 e 1 trecho de controle por sobreposição com
  um erro do mesmo vídeo.
- **`val` não tem erro persistente** nem os ambientes priorizados; nada é
  selecionado ali.
- **Os controles são menos comparáveis em pureza que os erros.** Dos 9
  erros, 6 são janelas puras a 23 ou mais quadros do início do trecho; dos
  6 controles, só 1 é puro: nos dois ambientes, 62 das 70 janelas de
  controle persistente são misturadas, isto é, ficam a menos de 23 quadros
  do início do trecho.

## Estrutura da saída e revisão cega

`--output-dir` recebe dois diretórios:

- `worksheet/` (cego): `review_worksheet.csv`, uma linha por item com
  `review_id`, `split`, `video_id`, ambiente, sujeito, rótulo verdadeiro,
  `k_end`, quadros, `src_index` e tempos extremos da janela, a folha e os
  campos de revisão vazios; `review_frames.csv`, com os 24 quadros de cada
  janela (`position`, `frame_index`, `src_index`, `time_s`, rótulo anotado e
  `replicated`); `sheets/<review_id>.png`; e `instructions.md`.
- `key/` (chave de desmascaramento): `review_selection.json`, com
  critérios, colunas de predição, cobertura, itens, exclusões, entradas
  ausentes, limitações e SHA-256 das entradas; `review_key.csv`, com papel,
  contexto da Etapa 3A, votos por braço, frações de concordância e
  contagens do trecho; e `review_selection.md`, com a cobertura.

Cada folha mostra os 24 quadros da janela em resolução de origem, em grade
de 6 × 4, com índice da grade (`k`), quadro de origem (`src`), tempo,
rótulo anotado e a marca `(borda)` nos quadros replicados. Os quadros são
decodificados por `decode_frames` com o mapeamento `frame_index →
src_index` de `frames.parquet`. Nem a planilha nem as folhas mostram
predições, votos ou o papel do item.

### Instruções de revisão

1. Revise as folhas na ordem de `review_id`, sem abrir `key/`.
2. Preencha só os campos abaixo; deixe em branco o que não puder ser
   julgado.
3. Desmascare com `review_key.csv` só depois de concluir todas as linhas.

| Campo | Valores |
| --- | --- |
| `reviewer` | Texto livre |
| `visible_action` | Texto livre: ação visível ao longo dos 24 quadros |
| `apparent_walking` | `yes`, `no`, `partial`, `unclear` |
| `annotation_ambiguity` | `none`; `boundary`, quando a ação muda perto da fronteira anotada; `label`, quando `other` parece não descrever a ação visível; `unclear` |
| `person_visibility` | `full`, `partial`, `occluded`, `absent`, `unclear` |
| `reviewer_notes` | Texto livre |

## Garantias de somente leitura

- Só a planilha e as folhas são novas; nenhum arquivo de entrada é
  alterado. O SHA-256 de `window_context_windows.csv`, `frames.parquet`,
  `manifest.parquet` e dos vídeos brutos usados é calculado antes da
  seleção e conferido antes da gravação; o de cada vídeo precisa coincidir
  com o `sha256` do manifesto.
- `window_context_windows.csv` ou `frames.parquet` ausentes fazem o comando
  falhar sem gravar nada. Vídeo bruto ausente é registrado em
  `missing_inputs`, e o item fica sem folha, sem conteúdo inventado.
- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, `docs/`, no diretório de
  `frames.parquet`, no diretório do CSV da Etapa 3A ou em `data/raw/le2i`.
  Dentro do repositório, só `runs/local/` é aceito.
- Sem `--force`, nada é sobrescrito. Com `--force`, os arquivos de mesmo
  nome são regravados; folhas de execuções anteriores com outro
  `review_id` não são apagadas.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.other_walk_review_set selftest
uv run python -m gatefall.eval.analysis.other_walk_review_set prepare \
  --windows-csv runs/local/window_context_diagnosis/le2i_cs_seeds_42_46/window_context_windows.csv \
  --output-dir runs/local/other_walk_review_set/le2i_cs_seeds_42_46 [--force]
```

`prepare` precisa de `frames.parquet`, `manifest.parquet` e dos vídeos
brutos do Le2i; não usa GPU nem checkpoints. Nesta máquina, a execução leva
cerca de 15 s.

`selftest` usa só dados sintéticos e cobre a seleção determinística
(limiar inclusivo por braço, um item por trecho, troca do representante
sobreposto, limite por grupo, ambientes fora da prioridade, `val` e `test`
separados, independência da ordem de linhas e colunas), a planilha cega, o
alinhamento exato dos quadros com um vídeo AVI gerado pelo FFmpeg e
decodificado por `decode_frames`, inclusive com replicação de borda,
divergências entre o CSV e `frames.parquet`, a integridade da entrada
(seeds ausentes, extras ou repetidas, coluna repetida, braço desconhecido,
predição fora dos rótulos, tabela truncada, janela repetida ou inexistente,
split inesperado, rótulo verdadeiro ou sujeito divergentes), entradas
ausentes, SHA-256 de vídeo divergente do manifesto e a proteção da saída.

### Execução registrada (não canônica)

2026-10-15, branch `feat/other-walk-review-set` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados do checkout principal; por isso, as chaves de `input_sha256` do JSON
são caminhos absolutos. Os SHA-256 de `runs/reference/`,
`data/processed/le2i` e da saída da Etapa 3A foram iguais antes e depois.
Duas execuções produziram arquivos de texto idênticos. Depois da inclusão
das validações de integridade da entrada, uma nova execução sobre as mesmas
entradas produziu os mesmos 15 itens e os mesmos 21 arquivos, byte a byte.
As saídas não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `window_context_windows.csv` (entrada, Etapa 3A) | `b23a0fd0853db5d5e416d3b29640ae9e0d52f13f9e7f8dd37b93d11d475dbb6c` |
| `key/review_selection.json` | `1c6720b094254a4e9496bfee9088fbd82b805b82e52cbc615921cc9d73148a53` |
| `key/review_key.csv` | `3781b238b4060ffe6bfe34a6a7aca50f760c0ba8ae00cc25c602c199d64a4983` |
| `worksheet/review_worksheet.csv` | `8bd376a45062478d2fb80c0e2d515df28e4ea1b111d85bf1427e8b62afb98b8e` |
| `worksheet/review_frames.csv` | `592ca9b798061581e981ff505a1697372b01f99b000b8f8e89ab2d5635d7091f` |

## Revisão concluída { #revisao-concluida }

Consolidação exploratória e não canônica de `test`, registrada em
2026-10-15. Os arquivos locais da revisão ficam em
`runs/local/other_walk_review_set/le2i_cs_seeds_42_46/human_review/` e não
são versionados, assim como folhas, quadros e vídeos.

### Procedência e conferência

| Arquivo | SHA-256 |
| --- | --- |
| `key/review_key.csv` (chave, igual à da execução registrada) | `3781b238b4060ffe6bfe34a6a7aca50f760c0ba8ae00cc25c602c199d64a4983` |
| `human_review/.../etapa3B_revisao_humana_final.csv` | `bab8e5d070f23d4c1214d08019670d2d4bdc9d66e2fb2bc9fcf724aa59a04469` |

- O CSV final tem exatamente os 15 `review_id` da chave, sem repetição.
- As 51 colunas da chave (papel, split, vídeo, `k_end`, pureza, contexto
  da Etapa 3A, votos por braço) reaparecem no CSV com valores idênticos,
  item a item.
- O CSV acrescenta dois blocos de colunas, que não se misturam:
  - `visual_*`: leitura preliminar das folhas por um modelo de IA
    (`visual_reviewer` = ChatGPT), registrada como feita antes do
    desmascaramento. Ela não foi refeita nem conferida aqui.
  - `human_*`: relatos do usuário, todos com
    `human_blinding_status` = relato posterior à disponibilização da chave,
    isto é, **não cegos**. `human_annotation_ambiguity` ficou em branco nos
    15 itens.
- `human_apparent_walking = partial` significa passos curtos ou
  deslocamento pontual relatados, não caminhada contínua demonstrada; `no`
  significa ausência de caminhada relatada. A anotação `other` não foi
  corrigida nem reinterpretada.

### Observações humanas após o desmascaramento

| Caminhada relatada | Erros persistentes | Controles corretos | Total |
| --- | ---: | ---: | ---: |
| `partial` (passos curtos) | 5 (3 misturadas, 2 puras) | 5 (misturadas) | 10 |
| `no` | 4 (puras) | 1 (pura) | 5 |
| **Total** | **9** | **6** | **15** |

- **Quatro erros persistentes `other→walk` ocorrem em janelas puras de
  `other` sem caminhada relatada:** `test-04` (`coffee_room_02/video_68`,
  pegar algo na geladeira; sem pose no quadro final), `test-05`
  (`lecture_room/video_16`, pegar uma jaqueta de uma cadeira), `test-08`
  (`lecture_room/video_23`, abaixar-se para pegar algo do chão) e `test-09`
  (`lecture_room/video_16`, fechar uma porta). Nesses quatro, os 24 quadros
  da janela têm o rótulo `other`, e todos os braços previram `walk` em pelo
  menos 4 de 5 seeds.
- `test-12`, o único controle puro, também não tem caminhada relatada
  (pegar uma jaqueta de uma cadeira, atividade parecida com a de
  `test-05`), mas os dois itens não foram pareados de forma controlada.
- Os outros dez itens descrevem interação com objetos, portas ou móveis
  com passos curtos. Os cinco controles desse grupo são janelas
  misturadas, em que a mudança anotada de `walk` para `other` acontece
  dentro da janela.
- A leitura preliminar da IA e o relato humano coincidem em
  `apparent_walking` em 13 dos 15 itens. Divergem `test-09` (IA `partial`,
  humano `no`) e `test-14` (IA `yes`, humano `partial`).

### O que a revisão sustenta e o que não sustenta

- **Observado:** a confusão `other→walk` persiste também em janelas puras de
  `other` em que o revisor não relata caminhada. Ela não depende só de
  fronteira de trecho ou de mistura de rótulos na janela.
- **Não sustentado:** qual sinal de entrada ou regra de decisão produz a
  predição `walk`, erro de anotação, prevalência em `test` ou em outros
  ambientes, e qualquer diferença entre braços.
- A amostra foi escolhida a partir de erros persistentes e controles em dois
  ambientes de um `test` já exposto. A pureza está desbalanceada (6 de 9
  erros puros contra 1 de 6 controles), e itens do mesmo vídeo, seeds e
  janelas sobrepostas não são observações independentes. Não há cálculo de
  significância nem estimativa de taxa.
- Os relatos humanos vieram depois do desmascaramento e não confirmam de
  forma cega e independente a leitura preliminar.
- Estes casos motivam a [Etapa 3C](other-walk-train-composition.md), que só
  usa `train`. Eles não são alvo de otimização: nenhuma escolha de modelo,
  janela, limiar ou treino se apoia neles.

## Limitações

- Os 15 itens não são uma amostra representativa: os critérios favorecem
  erros persistentes e puros em dois ambientes de `test`, e o limite deixa
  6 trechos de `Lecture_room` fora.
- Trechos do mesmo vídeo ou sujeito não são observações independentes; o
  conjunto serve para descrever casos, não para estimar taxas.
- A persistência é contada entre seeds e braços que compartilham anotação,
  janela e partição; não mede incerteza entre sujeitos nem independência
  entre braços.
- As predições vêm da reexecução validada da Etapa 3A e herdam suas
  limitações.
- O mascaramento é parcial: ambiente, vídeo e rótulos anotados aparecem na
  planilha e nas folhas, e quem conhece os resultados da Etapa 3A pode
  reconhecer trechos.
- Os controles são menos comparáveis em pureza e posição no trecho que os
  erros.
- `test` foi exposto pelos diagnósticos anteriores; nada aqui seleciona
  modelo, seed, limiar ou protocolo nem autoriza mudar anotação ou
  janelamento.
