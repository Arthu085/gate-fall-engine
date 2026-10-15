# Análise — Composição de `other` e `walk` no treino (Etapa 3C)

`src/gatefall/eval/analysis/other_walk_train_composition.py` responde a uma
pergunta motivada pela [revisão concluída da Etapa 3B](other-walk-review-set.md#revisao-concluida):
no `train` oficial de Le2i-CS, os trechos anotados `other` e `walk` diferem
em composição temporal, sinais de pose disponíveis e cobertura de
movimentos curtos?

O diagnóstico é **descritivo, exploratório, não canônico e somente
leitura**. De `frames.parquet`, só as linhas `split == "train"` entram nos
cálculos; a filtragem acontece logo após a leitura das colunas de
identificação e rótulo. A pose vem do HDF5 bruto dos 133 vídeos de `train`.
Não abre features, predições
nem pose de `val` ou `test`, não refaz inferência, não treina e não altera
rótulos, janelamento, checkpoints, métricas nem decisões experimentais. Os
quatro erros da Etapa 3B motivam a pergunta, mas não são alvo de
otimização: nenhuma regra abaixo usa `test`.

## Unidades e descritores

| Unidade ou descritor | Definição |
| --- | --- |
| Trecho | Sequência contígua máxima de quadros da grade (10 fps) com o mesmo rótulo, `IGNORE_LABEL` inclusive, como na [Etapa 3A](window-context-diagnosis.md); só trechos `other` e `walk` são descritos |
| Trecho curto | Menos de `WINDOW_FRAMES` (24) quadros; não gera janela pura |
| Janela de treino | Linha de `build_window_index` em `TRAIN_STRIDE` (4) com rótulo `other` ou `walk` no quadro final, como no treino dos braços; o contexto (pureza, distância ao início do trecho, trecho anterior, replicação de borda, pose no quadro final) vem de `video_context` da Etapa 3A |
| Pose disponível | `person_found` do HDF5 bruto, sem imputação: pose no quadro final, fração dos 24 índices de `window_frame_indices` com pose (replicados inclusive) e confiança média dos keypoints detectados |
| Deslocamento da bbox | Distância entre os centros da bbox no primeiro índice da janela e no quadro final, dividida pela altura mediana da bbox nos quadros detectados desse intervalo; indefinido sem pose em uma das extremidades |
| Deslocamento baixo | Menos de 0,25 altura da bbox na janela; os demais intervalos são 0,25–0,50, 0,50–1,00 e 1,00 ou mais |

O deslocamento é um descritor do artefato de pose no plano da imagem. Ele
depende da direção do movimento em relação à câmera, de oclusão e do
enquadramento e **não identifica caminhada nem a atividade real**.

## Regras de decisão

As regras foram fixadas no código antes da execução sobre os dados reais e
não foram alteradas depois. Δ = fração em `other` − fração em `walk`.

| Regra | Fração comparada | Condição compatível com uma razão no treino |
| --- | --- | --- |
| `R1_temporal_composition` | Janelas puras | Δ ≤ −0,10: contextos puros de `other` menos cobertos que os de `walk` |
| `R2_low_displacement_pure` | Deslocamento baixo entre janelas puras com deslocamento definido | Δ < +0,10: deslocamento baixo não distingue `other` de `walk` |
| `R3_pose_availability` | Janelas com pose no quadro final | Δ ≤ −0,10: menos pose disponível em `other` |

Uma regra é **sustentada** quando a condição vale no agregado do treino, há
pelo menos 3 sujeitos elegíveis (10 ou mais janelas de cada rótulo no
denominador) e a mesma condição vale em pelo menos dois terços deles. Como
os braços usam perda ponderada por classe, as regras comparam frações dentro
de cada rótulo, não contagens absolutas. Se ao menos uma regra for
sustentada, o veredito é `consider_controlled_intervention`; caso
contrário, `no_model_modification`. Os limiares são convenções, não valores
estimados.

## Resultados observados

Execução registrada abaixo, sobre os seis sujeitos de `train`.

**Trechos anotados.**

| Medida | `other` | `walk` |
| --- | ---: | ---: |
| Trechos / quadros | 95 / 4677 | 180 / 6170 |
| Vídeos / sujeitos | 57 / 6 | 108 / 6 |
| Duração mediana (p25–p75), quadros | 24 (15–46) | 28 (19–46) |
| Trechos curtos | 46 (48%; 14% dos quadros) | 71 (39%; 18% dos quadros) |
| Trechos com ao menos uma janela pura | 47 | 108 |
| Trecho anterior mais comum | `walk` (55) | `IGNORE_LABEL` (47) e `other` (46) |

**Janelas de treino.**

| Medida | `other` | `walk` |
| --- | ---: | ---: |
| Janelas | 1170 | 1547 |
| Janelas puras | 64% | 50% |
| Quadros desde o início do trecho, mediana | 36 | 18 |
| Pose no quadro final | 100% | 98% |
| Fração média de quadros com pose | 100% | 91% |
| Confiança média dos keypoints detectados | 0,80 | 0,73 |
| Deslocamento mediano (p25–p75), janelas puras | 0,09 (0,03–0,20) | 0,37 (0,21–0,55) |
| Deslocamento mediano (p25–p75), janelas misturadas | 0,25 (0,11–0,45) | 0,25 (0,12–0,45) |
| Janelas puras com deslocamento baixo | 608 de 747 (81%) | 234 de 723 (32%) |
| Janelas puras com deslocamento de 0,25 ou mais | 139 de 747 (19%) | 489 de 723 (68%) |

**Regras.**

| Regra | `other` | `walk` | Δ | Sujeitos com a condição / elegíveis | Sustentada |
| --- | ---: | ---: | ---: | ---: | --- |
| `R1_temporal_composition` | 64% de 1170 | 50% de 1547 | +14 p.p. | 1 / 6 | não |
| `R2_low_displacement_pure` | 81% de 747 | 32% de 723 | +49 p.p. | 0 / 6 | não |
| `R3_pose_availability` | 100% de 1170 | 98% de 1547 | +1 p.p. | 0 / 6 | não |

Por sujeito, Δ de R1 vai de −13 p.p. (sujeito 3, o único com a condição) a
+29 p.p.; Δ de R2 vai de +23 p.p. (sujeito 8) a +68 p.p. (sujeito 1); Δ de
R3 fica entre 0 e +3 p.p.

**Veredito: `no_model_modification`.**

O relatório também traz uma tabela por ambiente, acrescentada depois da
primeira execução, só descritiva e fora das regras. Em `Lecture_room`, 68
das 191 janelas `other` são puras (36%), 67 com deslocamento definido, das
quais 88% com deslocamento baixo, contra 10% das 185 janelas `walk` puras
com deslocamento definido. Em `Coffee_room_02`, 90% das 476 janelas
`other` são puras, e 76% das puras com deslocamento definido têm
deslocamento baixo, contra 49% em `walk`.

## Interpretação

**Observado no treino:**

- `other` não tem menos janelas puras que `walk`: no agregado tem mais, e
  só um dos seis sujeitos vai na direção contrária.
- Nas janelas puras, o deslocamento da bbox separa os dois rótulos de forma
  consistente: `other` concentra deslocamento baixo em todos os sujeitos.
  Nas janelas misturadas, que cruzam a fronteira do trecho, as
  distribuições se sobrepõem.
- A pose está quase sempre disponível em `other`; quem perde mais pose
  dentro da janela é `walk`.
- Movimentos curtos aparecem nos dois rótulos: 19% das janelas `other`
  puras têm deslocamento de 0,25 ou mais, e 32% das janelas `walk` puras
  têm deslocamento baixo. Quase metade dos trechos `other` é curta demais
  para gerar janela pura.

**Hipóteses não testadas aqui:**

- Os três sinais pré-declarados não indicam lacuna de cobertura no treino
  que explique erros `other→walk` em janelas puras sem caminhada relatada.
  A causa pode estar fora do que este diagnóstico mede: mudança de sujeito
  ou de ambiente entre `train` e `test`, a heterogeneidade das atividades
  agrupadas em `other`, sinais que a TCN usa além do deslocamento da bbox,
  ou a entrada visual de B0, B1, C0 e C1.
- O número menor de janelas `other` puras em `Lecture_room` é uma
  observação por ambiente feita depois da primeira execução. Ambientes não
  são a unidade das regras, e a observação não sustenta intervenção.
- Nenhum valor aqui compara o treino com as janelas de `test` da Etapa 3B:
  a pose de `test` não foi lida.

**Recomendação:** não modificar modelo, treino, janelamento, rótulos nem
alarmes com base nesta evidência. Uma intervenção futura só caberia com
nova pergunta pré-declarada, aplicada igualmente a A, B e C e avaliada sem
o `test` já exposto.

## Garantias de somente leitura

- Só linhas `train` de `frames.parquet` entram nas tabelas; a pose é
  aberta apenas para vídeos de `train`, e pose ausente de um vídeo de
  `train` faz o comando falhar sem gravar nada.
- As janelas calculadas precisam coincidir exatamente, em `(video_id,
  k_end, rótulo)`, com `build_window_index` em `TRAIN_STRIDE`; K da pose
  precisa coincidir com a grade de cada vídeo.
- Conservação: os quadros dos trechos somam os quadros `train` de cada
  rótulo, e as janelas somam o total por trecho, por sujeito, por ambiente
  e em cada dimensão categórica. Qualquer divergência faz o comando falhar.
- O SHA-256 de `frames.parquet` e de cada HDF5 de pose de `train` é
  calculado antes e conferido antes da gravação.
- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, `docs/`, no diretório de
  `frames.parquet` ou na raiz da pose. Dentro do repositório, só
  `runs/local/` é aceito. Sem `--force`, nada é sobrescrito.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.other_walk_train_composition selftest
uv run python -m gatefall.eval.analysis.other_walk_train_composition diagnose \
  --output-dir runs/local/other_walk_train_composition/le2i_cs_train [--force]
```

`diagnose` precisa de `data/processed/le2i/frames.parquet` e da pose de
`train` em `data/features/le2i/pose/`; não usa GPU nem checkpoints. Nesta
máquina, a execução leva cerca de 5 s. A saída tem
`other_walk_train_composition.json` (resumos, tabelas por sujeito e por
ambiente, regras, veredito, conservação, limitações e SHA-256 das
entradas), `other_walk_train_composition.md`, `other_walk_train_segments.csv`
(um trecho por linha) e `other_walk_train_windows.csv` (uma janela de
treino por linha).

`selftest` usa só dados sintéticos e cobre limites e vizinhos dos trechos,
trechos curtos, alinhamento exato com `build_window_index`, replicação de
borda, fração de detecção, deslocamento exato e indefinido, K de pose e
`frame_index` inconsistentes, conservação, leitura só de `train` (a pose de
`val` e `test` nunca é aberta e os rótulos de `val` e `test` não alteram o
relatório), limiares e elegibilidade das regras, determinismo, preservação
sem `--force`, entradas inalteradas e proteção da saída.

### Execução registrada (não canônica)

2026-10-15, branch `feat/other-walk-train-composition` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados do checkout principal; por isso, as chaves de `input_sha256` do JSON
são caminhos absolutos. Foram lidos 134 arquivos: `frames.parquet` e 133
HDF5 de pose de `train`. Os SHA-256 de `runs/reference/`,
`data/processed/le2i` e das saídas das Etapas 3A e 3B foram iguais antes e
depois. Duas execuções produziram arquivos idênticos. As saídas não são
versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `other_walk_train_composition.json` | `b3e9091e0a16f711fdd0a705eb759a0dccf84c11d2d9b667c100ef0918073dd1` |
| `other_walk_train_composition.md` | `925790dc273535f7d1578936fb845e4d56ade6792fa11536f442c6fbbd4c0139` |
| `other_walk_train_segments.csv` | `1592ad0d0a991422cb64f35747ee9ed2ab815977325b678b9fe027512808efd1` |
| `other_walk_train_windows.csv` | `c4f003079b71530e112441d13e8b200695fe908559ddfb7c1589f486937fa40e` |

## Limitações

- Janelas de treino se sobrepõem, e janelas e trechos do mesmo vídeo ou
  sujeito não são independentes. A consistência entre seis sujeitos é
  descritiva, sem teste de significância.
- Os sinais de pose descrevem o artefato YOLO-Pose bruto. Detecção não mede
  qualidade, e o deslocamento da bbox não equivale a locomoção.
- Em janelas com replicação de borda, o deslocamento cobre menos de 24
  quadros; ele é indefinido em 17 janelas `other` e 292 janelas `walk` por
  falta de pose nas extremidades.
- Os limiares das regras são convenções fixadas antes da execução; outros
  limiares podem levar a outro veredito.
- A tabela por ambiente foi acrescentada depois da primeira execução e não
  entra nas regras.
- O diagnóstico não descreve o que cada braço usa para decidir e não testa
  nenhuma das hipóteses acima.
