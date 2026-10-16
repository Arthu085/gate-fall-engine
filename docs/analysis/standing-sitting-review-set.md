# Análise — Conjunto de revisão visual de `standing→walk` e `sitting→stand_up` (Etapa 6B)

`src/gatefall/eval/analysis/standing_sitting_review_set.py` prepara a leitura
visual dos dois erros de desenvolvimento examinados na
[Etapa 6A](standing-sitting-diagnosis.md). A pergunta é se, nas observações
existentes, os erros parecem ambiguidade visual da ação, ambiguidade temporal
do rótulo ou falta de informação nos 24 quadros. A etapa só seleciona janelas
e monta folhas para revisão humana cega. Ela não propõe nem implementa mudança
de treino.

A preparação é **exploratória, não canônica e somente leitura**. Ela reusa as
predições por janela já gravadas pela Etapa 6A e não refaz inferência. Também
não altera anotações, features, partições, modelos, receitas de treino,
checkpoints, referências canônicas nem a evidência da Etapa 6A. Os quadros
vêm só de vídeos de `train` e `val`; o `test` oficial não é lido nem
selecionado.

!!! warning "Antes da revisão"
    Quem vai preencher a planilha não deve ler as seções [Critérios de
    seleção](#criterios-de-selecao) e [Cobertura da execução
    real](#cobertura-da-execucao-real) antes de concluir a revisão. A
    cobertura por estrato permite deduzir o papel de alguns itens.

## Evidência usada

| Evidência | Split | Modelos | Prefixo do `review_id` |
| --- | --- | --- | --- |
| `val` oficial | `val` (sujeito 5) | 25 runs: A, B0, B1, C0 e C1 × seeds 42–46 | `val-NN` |
| Grupos retidos das dobras por sujeito | `train` (sujeitos 0, 1, 3, 4, 6 e 8) | Controle do braço A, seed 42, de cada dobra da [Etapa 4B](../train/subject-folds.md) | `fold-NN` |

As duas evidências são selecionadas, contadas e numeradas à parte, e seus
números nunca são somados. No `val`, cada janela é contada uma vez, não uma
vez por seed. As dobras têm um único modelo, então não há persistência entre
seeds ou braços nelas.

## Critérios de seleção { #criterios-de-selecao }

As regras foram commitadas antes da execução sobre os dados reais e antes de
qualquer folha ser gerada.

| Etapa | Regra |
| --- | --- |
| Papel | **Erro**: em todo braço disponível, pelo menos 80% das seeds preveem a classe confundível do par (`walk` para `standing`; `stand_up` para `sitting`); com cinco seeds, 4 de 5 por braço. **Controle**: o mesmo com a classe verdadeira. Nas dobras, o único modelo decide |
| Estrato | (evidência, dobra, sujeito, classe verdadeira, categoria de janela da Etapa 6A: pose ausente, transição ou interior) |
| Unidade | Um item por trecho anotado, estrato e papel |
| Erros | Até 2 por estrato. Trechos ordenados pelo melhor representante: maior fração mínima entre braços e, em empate, SHA-256 de `<evidência>/<dobra>/<video_id>/<k_end>`. Os erros de todos os estratos são escolhidos antes dos controles |
| Controle pareado | Um por erro, obrigatoriamente da mesma evidência, dobra, sujeito, classe e categoria. Prefere o mesmo vídeo e, depois, outro vídeo do mesmo sujeito; dentro do nível, vale a mesma ordem dos erros. No máximo um controle por trecho e estrato |
| Sem par | O erro fica no conjunto e a chave registra `no_candidate` (nenhum controle no estrato) ou `overlap_or_segment_used` (candidatos sobrepostos ou de trechos já usados). Nenhum requisito é relaxado |
| Sobreposição | Duas janelas selecionadas do mesmo vídeo ficam a pelo menos 24 quadros (`WINDOW_FRAMES`) uma da outra; se o representante sobrepõe um item já escolhido, vale o próximo candidato do trecho |
| Identificador | `review_id` na ordem do SHA-256 de `<prefixo>/<video_id>/<k_end>`, que não depende do papel |

## Integridade das entradas

O comando falha sem gravar nada quando:

- `standing_sitting_windows.csv` não tem exatamente as colunas da Etapa 6A,
  tem evidência desconhecida, predição em `official_train`, rótulo, categoria,
  pureza ou inteiro inválidos, linha repetida, um dos 25 modelos do `val`
  ausente, dobra, braço ou seed inesperados, ou atributos de uma janela que
  divergem entre modelos ou janela ausente em algum modelo;
- `standing_sitting_diagnosis.json` não é da Etapa 6A, de Le2i-CS e não
  canônico, ou seus modelos e suas contagens por par (`n_true`, `n_error`,
  `n_correct`, `n_other_outcome`) não coincidem com as do CSV;
- as janelas de cada evidência não coincidem com as que `build_window_index`
  reconstrói, em `EVAL_STRIDE`, do `frames.parquet` filtrado em `train` e
  `val` (no caso das dobras, só com os sujeitos retidos); ou algum vídeo do
  CSV não está nesses splits;
- o split, o ambiente, o sujeito, o rótulo do quadro final, a pureza, as
  distâncias às fronteiras do trecho, os rótulos vizinhos ou a replicação de
  borda de uma janela divergem de `frames.parquet`, ou a categoria gravada não
  segue a regra da Etapa 6A.

## Cobertura da execução real { #cobertura-da-execucao-real }

Janelas distintas por evidência e estrato; "erro" e "controle" contam janelas
que atendem às regras de papel.

| Evidência | Sujeito | Classe | Categoria | Janelas | Erro: janelas / trechos | Controle: janelas / trechos | Selecionados erro / controle | Par no vídeo / no sujeito / sem par |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `val` | 5 | standing | transição | 72 | 21 / 2 | 0 / 0 | 2 / 0 | 0 / 0 / 2 |
| `val` | 5 | standing | interior | 24 | 9 / 1 | 0 / 0 | 1 / 0 | 0 / 0 / 1 |
| `val` | 5 | sitting | transição | 134 | 8 / 3 | 59 / 13 | 2 / 2 | 1 / 1 / 0 |
| `val` | 5 | sitting | interior | 136 | 23 / 3 | 56 / 10 | 2 / 2 | 0 / 2 / 0 |
| `retain_s0` | 0 | standing | transição | 304 | 139 / 16 | 75 / 9 | 2 / 2 | 0 / 2 / 0 |
| `retain_s0` | 0 | standing | interior | 192 | 66 / 7 | 74 / 4 | 2 / 2 | 0 / 2 / 0 |
| `retain_s0` | 0 | sitting | pose ausente | 20 | 0 / 0 | 0 / 0 | 0 / 0 | — |
| `retain_s0` | 0 | sitting | transição | 713 | 41 / 12 | 531 / 50 | 2 / 2 | 1 / 1 / 0 |
| `retain_s0` | 0 | sitting | interior | 1015 | 55 / 7 | 796 / 30 | 2 / 2 | 0 / 2 / 0 |
| `retain_s1` | 1 | standing | transição | 70 | 10 / 2 | 23 / 3 | 2 / 1 | 0 / 1 / 1 |
| `retain_s1` | 1 | sitting | transição | 169 | 13 / 6 | 140 / 15 | 2 / 2 | 0 / 2 / 0 |
| `retain_s1` | 1 | sitting | interior | 240 | 8 / 1 | 232 / 10 | 1 / 1 | 1 / 0 / 0 |
| `retain_s3-4-6-8` | 3 | sitting | transição | 78 | 9 / 2 | 58 / 6 | 2 / 2 | 1 / 1 / 0 |
| `retain_s3-4-6-8` | 3 | sitting | interior | 179 | 40 / 2 | 139 / 4 | 2 / 2 | 0 / 2 / 0 |
| `retain_s3-4-6-8` | 4 | sitting | transição | 5 | 3 / 1 | 2 / 1 | 1 / 0 | 0 / 0 / 1 |
| `retain_s3-4-6-8` | 4 | sitting | interior | 56 | 10 / 1 | 46 / 1 | 1 / 1 | 1 / 0 / 0 |
| `retain_s3-4-6-8` | 6 | standing | transição | 92 | 44 / 3 | 39 / 3 | 2 / 2 | 1 / 1 / 0 |
| `retain_s3-4-6-8` | 6 | standing | interior | 15 | 6 / 1 | 9 / 1 | 1 / 0 | 0 / 0 / 1 |
| `retain_s3-4-6-8` | 6 | sitting | transição | 79 | 1 / 1 | 72 / 3 | 1 / 1 | 0 / 1 / 0 |
| `retain_s3-4-6-8` | 6 | sitting | interior | 178 | 3 / 1 | 175 / 2 | 0 / 0 | — |

- **54 itens:** 11 do `val` (7 erros, 4 controles) e 43 das dobras (23
  erros, 20 controles), de 38 vídeos. Todas as folhas foram geradas.
- **`standing` no `val` não tem controle.** Nenhuma janela `standing` do
  `val` é prevista corretamente por pelo menos 4 de 5 seeds em todos os
  braços. Os 3 erros `standing` do `val` ficam sem par (`no_candidate`), e
  para esses itens a revisão não é cega para quem conhece esta cobertura.
- Nas dobras, 3 erros ficam sem par por sobreposição ou trecho já usado. Dos
  24 erros pareados nas duas evidências, 6 têm controle no mesmo vídeo e 18 em
  outro vídeo do mesmo sujeito.
- O limite de 2 erros por estrato deixa de fora 34 trechos de erro em
  `retain_s0`, 4 em `retain_s1`, 1 em `retain_s3-4-6-8` e 2 no `val`. Em
  `retain_s3-4-6-8`, sujeito 6, `sitting` interior, o único trecho de erro foi
  excluído por sobreposição.
- O sujeito 8 não tem janelas `standing` nem `sitting` no grupo retido, e as
  20 janelas `sitting` com pose ausente do sujeito 0 não têm erro.

## Estrutura da saída e revisão cega

`--output-dir` recebe dois diretórios:

- `worksheet/` (cego): `review_worksheet.csv`, uma linha por item com
  `review_id`, split, vídeo, ambiente, sujeito, rótulo verdadeiro, `k_end`,
  quadros, `src_index` e tempos extremos da janela, quadros replicados, a
  folha e os campos de revisão vazios; `review_frames.csv`, com os 24 quadros
  de cada janela (`position`, `frame_index`, `src_index`, `time_s`, rótulo
  anotado e `replicated`); `sheets/<review_id>.png`; e `instructions.md`.
- `key/` (chave de desmascaramento, permissão `0700` no diretório e `0600`
  nos arquivos): `review_selection.json`, com critérios, procedência dos 27
  modelos da Etapa 6A, `split_access`, cobertura, itens, pareamentos,
  exclusões, entradas e evidências ausentes, limitações, SHA-256 das entradas
  e de cada arquivo de `worksheet/`; `review_key.csv`, com evidência, dobra,
  papel, par, categoria, contexto do trecho, pose, votos por braço, frações de
  concordância, `review_id` do item pareado, nível de pareamento e motivo da
  falta de par; e `review_selection.md`, com a cobertura.

As folhas usam o leiaute da [Etapa 3B](other-walk-review-set.md): os 24
quadros da janela em resolução de origem, em grade de 6 × 4, decodificados
por `decode_frames` com o mapeamento `frame_index → src_index` de
`frames.parquet`, com índice da grade, quadro de origem, tempo, rótulo
anotado e a marca `(borda)` nos quadros replicados. Nem a planilha, nem os
quadros, nem as folhas mostram predições, votos, papel, categoria, evidência
ou pareamento.

### Instruções de revisão

1. Revise as folhas na ordem de `review_id`, só pelo que as imagens mostram,
   sem consultar predições, a chave, outras folhas do mesmo vídeo nem os
   documentos das Etapas 6A e 6B.
2. Preencha só os campos abaixo. Prefira `unclear` ou `insufficient` a um
   palpite e deixe em branco o que não puder ser julgado.
3. Desmascare com `key/` só depois de concluir todas as linhas.

| Campo | Valores |
| --- | --- |
| `reviewer` | Texto livre |
| `visible_action` | Texto livre: ação visível ao longo dos 24 quadros, sobretudo nos últimos |
| `apparent_motion` | `none` (parada); `in_place` (gestos ou ajustes sem deslocamento); `stepping` (passos ou deslocamento); `posture_change` (sentar, levantar ou abaixar); `unclear` |
| `confusable_reading` | `no`, `plausible`, `unclear`: se a ação nos últimos quadros poderia ser lida como a classe vizinha (`walk` para `standing`; `stand_up` para `sitting`) |
| `label_ambiguity` | `none`; `boundary`, quando a ação muda perto de uma fronteira anotada; `label`, quando o rótulo do último quadro parece não descrever a ação visível; `unclear` |
| `person_visibility` | `full`, `partial`, `occluded`, `absent`, `unclear`, nos últimos quadros |
| `review_confidence` | `high`, `medium`, `low`; `insufficient` quando os 24 quadros não permitem julgar a ação |
| `reviewer_notes` | Texto livre |

`confusable_reading` aponta para a ambiguidade visual da ação,
`label_ambiguity` para a ambiguidade temporal do rótulo e
`review_confidence = insufficient` para a falta de informação. A pergunta de
`confusable_reading` é a mesma para todos os itens da mesma classe e não
revela o papel.

## Garantias de somente leitura

- `frames.parquet` é lido com filtro de split; só linhas de `train` e `val`
  são materializadas. Só vídeos desses splits têm caminho resolvido, são
  hasheados ou decodificados, e a decodificação é envolvida por uma guarda que
  recusa qualquer outro vídeo. O manifesto é lido inteiro como metadado e
  filtrado logo em seguida. `split_access` registra os splits lidos e quantos
  vídeos de cada split foram decodificados.
- O SHA-256 do CSV e do JSON da Etapa 6A, de `frames.parquet`, do manifesto e
  dos vídeos usados é calculado antes da seleção e conferido antes da
  gravação; o de cada vídeo precisa coincidir com o `sha256` do manifesto.
- CSV ou JSON da Etapa 6A, ou `frames.parquet`, ausentes fazem o comando
  falhar sem gravar nada. Vídeo bruto ausente vai para `missing_inputs`, e o
  item fica sem folha. Dobra sem janelas na Etapa 6A vai para
  `missing_evidence`.
- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, `docs/`, no diretório da Etapa 6A, no
  diretório de `frames.parquet` ou em `data/raw/le2i`. Dentro do repositório,
  só `runs/local/` é aceito. Sem `--force`, nada é sobrescrito.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.standing_sitting_review_set selftest
uv run python -m gatefall.eval.analysis.standing_sitting_review_set prepare \
  --diagnosis-dir runs/local/standing_sitting_diagnosis/le2i_cs \
  --output-dir runs/local/standing_sitting_review_set/le2i_cs [--force]
```

`prepare` precisa da saída da Etapa 6A, de `frames.parquet`, do manifesto e
dos vídeos brutos do Le2i; não usa GPU nem checkpoints. Nesta máquina, a
execução leva cerca de 40 s.

`selftest` usa só dados sintéticos, com o CSV produzido pelo próprio código
de atributos da Etapa 6A. Cobre a seleção determinística (limiar inclusivo por
braço, um item por trecho e estrato, limite por estrato, independência da
ordem das linhas, execuções repetidas idênticas), o pareamento (mesmo vídeo
antes do mesmo sujeito, categoria pose ausente, erros sem controle por falta
de candidato ou sobreposição), a sobreposição, a separação entre `val` e
dobras, a dobra ausente, o isolamento de split (leitura filtrada, vídeo de
`test` nunca decodificado nem hasheado, saídas idênticas quando só linhas e
vídeo de `test` mudam, janela em vídeo de `test`, evidência desconhecida,
vídeo de dobra fora de `train`, guarda de decodificação), o alinhamento entre
rótulos, janela e quadros com um AVI gerado e decodificado pelo FFmpeg,
inclusive com replicação de borda, a separação e a permissão da chave, os
SHA-256 da planilha, a integridade do CSV e do JSON, entradas ausentes,
SHA-256 de vídeo divergente do manifesto e a proteção da saída.

### Execução registrada (não canônica)

2026-10-16, branch `feat/standing-sitting-review-set` (base
`analysis/gatefall-improvements`, commit `e45e356`), com o código do
worktree do branch e os dados e a saída da Etapa 6A do checkout principal, a
partir da raiz desse checkout; por isso, as chaves de `input_sha256` do JSON
são caminhos absolutos. A entrada foi a
[execução registrada da Etapa 6A](standing-sitting-diagnosis.md#execucao-registrada-nao-canonica).
Os SHA-256 de `runs/reference/`, `data/processed/le2i/` e da saída da Etapa
6A (32 arquivos) foram iguais antes e depois. Uma segunda execução, gravada
fora do repositório, produziu os mesmos 60 arquivos byte a byte. As saídas
não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `key/review_selection.json` | `8b5e64f9d31a88900f1d0cfcdb454ff1959d6e0356e63dcd7f88f48f92fd21a5` |
| `key/review_key.csv` | `ab41a1dc44ae6d32a7f539932b9b2acb77ee99a9bd02f3a78dc32229720710e1` |
| `worksheet/review_worksheet.csv` | `056f82d672272fff250e400d4b1e2eaa3a98e7a36edc4f37719df77ed914719a` |
| `worksheet/review_frames.csv` | `e488f9d94e5c8d6ddaa3526f609a24cde27ecd3a058bb2883d884fb38d5a4890` |

**Revisão humana pendente.** Nenhuma folha foi anotada, e nenhum achado
visual é relatado ou interpretado aqui.

## Limitações

- Os 54 itens não são uma amostra representativa: o limite por estrato e a
  ordem por hash tornam a seleção reprodutível, não proporcional aos erros.
- Itens do mesmo vídeo ou sujeito não são observações independentes; o
  conjunto descreve casos e não estima taxas. O `val` oficial tem um único
  sujeito.
- A persistência entre seeds e braços só existe no `val`; nas dobras, o papel
  vem de um único modelo e pode mudar com outra seed.
- As predições vêm das reexecuções validadas da Etapa 6A e herdam suas
  limitações.
- O mascaramento é parcial: ambiente, vídeo, sujeito e rótulos anotados
  aparecem na planilha e nas folhas. Quem conhece a Etapa 6A ou a cobertura
  acima pode inferir a categoria da janela ou, em `standing` do `val`, o
  papel.
- 18 dos 24 controles pareados vêm de outro vídeo do mesmo sujeito, não do
  mesmo vídeo, e 6 erros não têm controle.
