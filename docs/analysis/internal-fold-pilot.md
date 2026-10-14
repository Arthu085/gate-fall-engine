# Análise — Piloto das dobras internas do Le2i-CV (Etapa 2D)

Este documento encerra a Etapa 2D: o piloto de nove treinos das
[dobras internas por ambiente](../train/internal-folds.md) do Le2i-CV, com A,
B0 e B1 na seed 42. É um registro **exploratório, não canônico e de resultado
negativo**. Ele não muda código de treino, protocolo, critério, splits
oficiais, runs, checkpoints nem resultados canônicos, e não cria nova
intervenção experimental.

## Decisão

**A Etapa 2D fica encerrada como resultado exploratório negativo.**

- O piloto não reproduz o deslocamento de B0/B1 para `other` em nenhuma das
  três dobras. Nas três, B0 e B1 predizem `other` com fração menor que A.
- Com esses runs da seed 42, o critério congelado da Etapa 2A não pode ser
  satisfeito por B0 nem por B1 em nenhuma dobra, quaisquer que sejam os
  resultados das seeds 43 a 46 (ver [viabilidade do critério](#viabilidade-do-criterio-de-cinco-seeds)).
- **Recomendação: não executar os 36 treinos restantes da avaliação** (seeds
  43 a 46). Eles não podem mudar a conclusão do critério e consumiriam GPU
  sem informação de decisão.
- A decisão da Etapa 2A sobre o
  [candidato 2](intervention-decision.md#candidato-2-dropout-de-modalidade-visual-em-b0b1-condicional)
  continua **não**: o dropout de modalidade visual era condicionado à
  reprodução do deslocamento nas dobras, e ela não ocorreu.

O piloto **não** é uma avaliação de cinco seeds executada e **não** explica a
falha observada no `test` oficial do Le2i-CV.

## Fontes

- `runs/local/le2i_cv_internal_folds/summary/pilot/internal_fold_summary.json`,
  gerado por `summarize --seed-set pilot`, que reconcilia `folds.json`, a
  proveniência das estatísticas e o `fold_run.json` de cada run antes de
  relatar qualquer número.
- `final.train` e `final.test` de `metrics.json` dos nove runs. No protocolo
  das dobras, as posições `val` e `test` do engine avaliam o mesmo ambiente
  retido, e o resumo confere que são idênticas. As matrizes de confusão de
  `Lecture_room` abaixo vêm desses arquivos.

Os artefatos ficam em `runs/local/`, que não é versionado. A seção
[artefatos preservados](#artefatos-preservados) registra os hashes.

## Resultados observados

Executado em 2026-10-14 com o protocolo da Etapa 2C: receita congelada,
estatísticas por dobra, checkpoint da última época e ambiente retido em
stride 1. Só o `train` oficial do Le2i-CV foi lido.

A fração predita como `other` é a soma da coluna `other` fora da linha
`other` da matriz de confusão do ambiente retido, dividida pelas janelas
rotuladas fora de `other`. A Macro-F1 de treino é a de `final.train`, no
treino de desenvolvimento da dobra. Valores de uma única seed, sem variação
entre seeds.

| Dobra (ambiente retido) | Arma | Janelas fora de `other` | Fração predita `other` | Macro-F1 treino dev. | Macro-F1 retido |
| --- | --- | ---: | ---: | ---: | ---: |
| `Coffee_room_01` | A | 5339 | 0,140 | 0,867 | 0,612 |
| `Coffee_room_01` | B0 | 5339 | 0,042 | 0,997 | 0,712 |
| `Coffee_room_01` | B1 | 5339 | 0,043 | 0,995 | 0,685 |
| `Coffee_room_02` | A | 3167 | 0,086 | 0,858 | 0,609 |
| `Coffee_room_02` | B0 | 3167 | 0,046 | 0,996 | 0,524 |
| `Coffee_room_02` | B1 | 3167 | 0,036 | 0,989 | 0,492 |
| `Lecture_room` | A | 4431 | 0,207 | 0,888 | 0,447 |
| `Lecture_room` | B0 | 4431 | 0,009 | 0,999 | 0,096 |
| `Lecture_room` | B1 | 4431 | 0,000 | 0,999 | 0,086 |

Fatos confirmados:

1. **Nenhum deslocamento para `other`.** Nas três dobras, a fração predita
   como `other` de B0 e de B1 é menor que a de A. Em `Lecture_room`, B1 não
   prediz `other` para nenhuma janela fora de `other`.
2. **B0/B1 quase memorizam o treino de desenvolvimento.** A Macro-F1 de
   treino fica entre 0,989 e 0,999, contra 0,858 a 0,888 em A, o mesmo padrão
   já registrado no Le2i-CV oficial na
   [Etapa 2A](intervention-decision.md#evidencia-confirmada-em-train-e-val).
3. **O ambiente retido não ordena as armas de forma estável.** B0/B1 superam
   A em `Coffee_room_01`, ficam abaixo em `Coffee_room_02` e colapsam em
   `Lecture_room`.

### Diagnóstico de `Lecture_room`

O ambiente retido tem 5688 janelas rotuladas, 2797 delas `walk` (49%) e
1257 `other`. Contagens de colunas preditas sobre todas as janelas:

| Arma | Predito `walk` | Predito `other` | Demais colunas | Fração `walk` entre as janelas fora de `other` |
| --- | ---: | ---: | ---: | ---: |
| A | 1164 | 1543 | 2981 | 0,234 |
| B0 | 5549 | 100 | 39 | 0,982 |
| B1 | 5680 | 5 | 3 | 0,999 |

- **B0 e B1 predizem `walk` quase exclusivamente.** B0 prediz `walk` em
  97,6% das janelas e B1 em 99,9%. Fora `walk`, B0 só usa `other` (100
  janelas), `stand_up` (37, todas de `fallen`) e `fall` (2). B1 só usa
  `other` (5) e `fall` (3).
- Toda classe não `walk` vai majoritariamente para `walk`: em B0, 310 de 312
  janelas `fall`, 418 de 455 `fallen`, todas as 108 `sit_down`, 481 de 491
  `sitting`, todas as 101 `stand_up`, 148 de 167 `standing` e 1198 de 1257
  `other`. Em B1, só 3 janelas `fall` e 5 `other` escapam de `walk`.
- A revocação de `walk` é 2785/2797 em B0 e 2797/2797 em B1. A Macro-F1 baixa
  vem das outras classes, não de `walk`.
- A, no mesmo ambiente, distribui as predições entre as classes, e sua
  principal confusão é `walk→other` (730 janelas) e `walk→sit_down` (544).
- A curva registrada por época no ambiente retido, que não seleciona
  checkpoint, fica entre 0,084 e 0,150 em B0 e entre 0,073 e 0,098 em B1 nas
  30 épocas, contra 0,294 a 0,473 em A. O colapso não é um efeito só da
  última época.
- `walk` é a classe mais frequente do treino de desenvolvimento dessa dobra
  (`Coffee_room_01` e `Coffee_room_02`): 925 de 2743 janelas rotuladas em
  stride 4.

O modo de falha de B0/B1 em `Lecture_room` é o colapso para `walk`, não o
deslocamento para `other` visto no `test` oficial. Ele não satisfaz o
critério e não foi pré-registrado como sinal de nenhuma intervenção.

### Explicações possíveis, não testadas

Nada nesta etapa separa as leituras abaixo. Elas não são resultados.

- **Atalho de cena.** O treino de desenvolvimento da dobra `Lecture_room`
  tem só as duas salas de café. B0/B1 quase memorizam esse treino, e o
  DINOv3 pode codificar a cena junto com a postura. A pose sozinha (A) não
  colapsa no mesmo ambiente.
- **Classe majoritária sob mudança de entrada.** Se a entrada visual de
  `Lecture_room` cai fora do que o treino cobre, a TCN pode recair na classe
  mais frequente do treino. Os pesos de classe vêm das janelas de
  desenvolvimento, o que torna essa leitura incompleta.
- **Composição da dobra.** O retido muda sujeitos, câmera e proporção de
  classes ao mesmo tempo. O treino de cada dobra tem cerca de dois terços das
  janelas do `train` oficial.
- **Variação de seed.** Com uma seed, não se sabe se o colapso se repete.

## Viabilidade do critério de cinco seeds

O critério congelado na Etapa 2A, sem alteração:

> O deslocamento para other é considerado reproduzido se, em pelo menos dois
> dos três ambientes retidos, a menor fração predita como other entre as
> seeds de B0 (ou de B1) superar a maior fração entre as seeds de A.

Para uma dobra e uma arma de fusão `F ∈ {B0, B1}`, seja `f_X(s)` a fração
predita como `other` da arma `X` na seed `s`, e `S = {42, …, 46}`. A dobra
satisfaz a comparação se `min_{s∈S} f_F(s) > max_{s∈S} f_A(s)`.

Como `42 ∈ S`, para quaisquer valores das seeds 43 a 46:

```text
min_{s∈S} f_F(s) ≤ f_F(42)    e    f_A(42) ≤ max_{s∈S} f_A(s)
```

Nas três dobras, o piloto mostra `f_F(42) < f_A(42)` para B0 e para B1:

| Dobra | `f_A(42)` | `f_B0(42)` | `f_B1(42)` |
| --- | ---: | ---: | ---: |
| `Coffee_room_01` | 0,140 | 0,042 | 0,043 |
| `Coffee_room_02` | 0,086 | 0,046 | 0,036 |
| `Lecture_room` | 0,207 | 0,009 | 0,000 |

Logo, `min f_F ≤ f_F(42) < f_A(42) ≤ max f_A`, e a comparação estrita falha
em todas as dobras para as duas armas. O número de dobras que a satisfazem é
zero para B0 e zero para B1, abaixo dos dois exigidos.

Alcance do argumento:

- Ele vale enquanto a avaliação reaproveitar estes runs da seed 42, como o
  protocolo faz com runs íntegros. Retreinar a seed 42 com `--force` gera
  outros runs e não faz parte desta conclusão.
- Ele mostra que o critério não pode ser satisfeito. Não substitui uma
  avaliação de cinco seeds executada e não diz nada sobre a variação entre
  seeds.
- Ele não prova a causa da falha no `test` oficial. Como a
  [Etapa 2A](intervention-decision.md#candidato-1-validacao-interna-por-ambiente-em-le2i-cv-instrumento)
  já previa, um resultado negativo nas dobras não distingue entre uma
  diferença real do `test` (resolução, fps, ambiente) e a mudança de treino
  das próprias dobras.

## Artefatos preservados

Os nove runs ficam em `runs/local/le2i_cv_internal_folds/`, sem alteração. O
SHA-256 de `config.yaml`, `metrics.json` e `checkpoint.pt` de cada run foi
recalculado e confere com o `fold_run.json` correspondente.

| Dobra | Arma | `checkpoint.pt` | `metrics.json` |
| --- | --- | --- | --- |
| `retain_Coffee_room_01` | A | `5ad6f23b…d780` | `327e8784…` |
| `retain_Coffee_room_01` | B0 | `3db01250…fa9f` | `b887bd97…` |
| `retain_Coffee_room_01` | B1 | `c32e8bdc…39ff` | `deb67a37…` |
| `retain_Coffee_room_02` | A | `97336f54…332d` | `13482648…` |
| `retain_Coffee_room_02` | B0 | `ce955dcb…e512` | `6261994e…` |
| `retain_Coffee_room_02` | B1 | `e2460cc0…171d` | `37db8086…` |
| `retain_Lecture_room` | A | `6d870ee4…db83` | `1fb9429b…` |
| `retain_Lecture_room` | B0 | `7c38d482…5ca1` | `3f9e2a54…` |
| `retain_Lecture_room` | B1 | `3376a4ad…57c3` | `9df40feb…` |

Entradas e composição, iguais em `folds.json` e no resumo:
`frames.parquet` `b65bedc3…b640`, `manifest.parquet` `4dcad0c5…a94a` e
`membership_sha256` `fd300467…5c99`. O resumo registra
`official_splits_read: ["train"]`: `val` e `test` oficiais não foram lidos.

## Limitações

- Uma seed por arma. As frações e as Macro-F1 não têm variação entre seeds e
  não devem ser lidas como estimativas.
- Cada dobra treina com dois ambientes em vez de três, com outra composição
  de classes, sujeitos e cenas. As magnitudes não são comparáveis às do
  Le2i-CV oficial.
- As frações são por janela em stride 1, sobre janelas sobrepostas, e a
  proporção de `other` muda entre ambientes retidos. Só comparações dentro
  da mesma dobra são válidas.
- A F1 por classe no ambiente retido depende de poucos vídeos em algumas
  dobras.
- Todo o `train` oficial é 320×240 a 25 fps. As dobras não testam a
  hipótese de resolução ou fps do `test` oficial.
- IDs brutos de sujeito se repetem entre ambientes. As dobras são disjuntas
  por ambiente, não comprovadamente por sujeito.
