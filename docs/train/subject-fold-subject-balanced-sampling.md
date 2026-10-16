# Treino — Amostragem balanceada por sujeito nas dobras por sujeito do Le2i-CS (Etapa 5A)

`src/gatefall/train/subject_folds/subject_balanced_sampling.py` acrescenta às
[dobras por sujeito da Etapa 4B](subject-folds.md) uma única intervenção
predeclarada de amostragem: as épocas de treino do braço A sorteiam as
janelas do treino de desenvolvimento com probabilidade igual por sujeito.
A pergunta é se reduzir o domínio de poucos sujeitos sobre as janelas de
treino melhora a generalização para sujeitos retidos. Cada run balanceado é
comparado ao controle da Etapa 4B da mesma dobra, seed e receita.

A intervenção é **exploratória e não canônica**, mesmo que o critério seja
atendido. Ela não muda as configurações A/B/C, os splits oficiais, as
features, as estatísticas oficiais nem os runs das Etapas 4B e 4C. O
resultado não é uma afirmação de significância estatística nem uma
validação do alvo de 0,800, e `val` e `test` oficiais nunca entram em
nenhuma seleção.

## O que fica igual ao controle

- As três dobras congeladas, `folds.json`, a seed 42 e a estatística de pose
  de cada dobra, ajustada só no treino de desenvolvimento limpo.
- A receita de `BASELINE_A_CONFIG`: arquitetura, AdamW, cosseno, 30 épocas,
  entropia cruzada ponderada por classe e checkpoint da última época.
  `config.yaml` só difere do controle em `run_name`
  (`baseline_a_subject_balanced`).
- As janelas, os rótulos e as features limpas do treino de desenvolvimento.
  Os pesos de classe continuam calculados sobre o treino limpo completo.
- A inicialização do modelo. O sorteio usa um gerador próprio e não consome
  o RNG global do NumPy, do PyTorch nem do `random`, então os pesos iniciais
  e o fluxo de dropout por lote são os do controle.
- A avaliação do grupo retido, nas posições `val` e `test` do engine, com as
  mesmas janelas limpas da Etapa 4B.

Não há balanceamento por classe, aumento de dados nem segunda intervenção.

## A intervenção

Em cada época, o treino sorteia com reposição exatamente o número original
de janelas do treino de desenvolvimento (stride 4). A probabilidade de uma
janela é

```text
p(janela) = 1 / (S · n_s)
```

em que `S` é o número de sujeitos de desenvolvimento da dobra e `n_s` o
número de janelas de treino do sujeito da janela. Cada sujeito soma `1 / S`
por sorteio. Sujeitos com poucas janelas são repetidos, e parte das janelas
de sujeitos dominantes não é sorteada em uma época.

- **Gerador isolado.** A época `e` usa `numpy.random.default_rng` semeado por
  (`RANDOM_STREAM_KEY`, seed, `e`). O sorteio é reprodutível e independente
  da inicialização do modelo.
- **Só o treino.** O sampler recebe só o índice das janelas de
  desenvolvimento. O retido e `val` e `test` oficiais não passam por ele.
- **Avaliação limpa do treino.** `final.train` é medido num loader
  sequencial sobre todas as janelas limpas do treino de desenvolvimento,
  nunca pelo sampler, e continua comparável ao do controle.

### Hook no engine

`run_training` de `gatefall.train.baseline_a.engine` aceita o argumento
opcional `train_sampler`. Sem ele, o comportamento é o anterior: um único
loader de treino embaralhado pelo gerador da seed, reaproveitado em
`final.train`. As configurações A, B e C canônicas, a Etapa 4B e a Etapa 4C
não passam o argumento. Com ele, o loader de treino usa o sampler e o mesmo
gerador da seed, e `final.train` usa um loader sequencial separado. Os
engines de B e C não mudam.

## Critério exploratório congelado

O critério foi fixado em `DECISION_RULE` e commitado antes dos treinos reais:

- o delta pareado de macro-F1 restrita no retido (balanceado menos
  controle) é de pelo menos +0,03 em pelo menos duas das três dobras; e
- nenhum delta fica abaixo de −0,02 na dobra restante.

Os deltas não são agregados entre dobras. `standing`, `other` e `fall` são
relatados à parte, com o suporte de cada dobra.

## Diretório e proveniência

```text
runs/local/le2i/subject_folds/
├── folds.json                                  # da Etapa 4B, só lido
├── retain_<sujeitos>/
│   ├── stats/{pose,provenance}.json            # da Etapa 4B, só lidos
│   └── runs/
│       ├── baseline_a/seed_42/                 # controle da Etapa 4B, só lido
│       ├── baseline_a_detection_dropout/seed_42/  # da Etapa 4C, não tocado
│       └── baseline_a_subject_balanced/seed_42/
│           ├── config.yaml, metrics.json, checkpoint.pt
│           └── fold_run.json
└── summary/
    ├── baseline/                               # da Etapa 4B, não regravado
    ├── detection_dropout/                      # da Etapa 4C, não regravado
    └── subject_balanced/subject_fold_subject_balanced_summary.{json,md}
```

O `fold_run.json` do run balanceado tem os campos do controle, com
`stage: "5A"` e `variant: "subject_balanced"`, e acrescenta:

- `intervention`: a regra de probabilidade, o número de sorteios, o fluxo
  aleatório, a avaliação limpa e a regra de decisão;
- `sampling`: o índice das janelas de treino (sujeito e rótulo, codificados
  por trechos), as janelas e a probabilidade por sujeito, o suporte limpo
  por classe, as janelas sorteadas por sujeito em cada época e no total, as
  janelas sorteadas por classe, as janelas distintas sorteadas por sujeito e
  o SHA-256 de todos os índices sorteados;
- `control`: o run do controle e o SHA-256 do seu `fold_run.json`.

`sampling` é calculado antes do treino com o mesmo sorteio. Depois do
treino, os índices que o sampler entregou ao loader precisam ter o mesmo
SHA-256, ou o `fold_run.json` não é publicado. `train --variant
subject_balanced` valida o controle de todas as dobras pedidas antes de
treinar qualquer uma. Um controle ausente ou inconsistente interrompe a
execução sem treiná-lo, repará-lo ou substituí-lo.

## Resumo pareado

`summarize --variant subject_balanced` é somente leitura e não abre frames,
manifesto nem features. Ele valida cada controle como o resumo da Etapa 4B e
cada run balanceado contra a configuração, a intervenção congelada e o
`fold_run.json` atual do controle. O índice de janelas gravado é
reconciliado com `folds.json` (janelas, sujeitos e suporte por classe), e o
sorteio inteiro é recalculado a partir dele: qualquer contagem ou hash
gravado que diverja é recusado. O resumo exige suporte do retido idêntico
nos dois runs e suporte de `final.train` igual ao do treino limpo, e relata,
por dobra, a macro-F1 restrita dos dois runs e o delta, a F1 e o delta de
cada classe restrita com o suporte, as janelas sorteadas por sujeito e por
classe, o critério congelado e as limitações. `pooled_score` e
`pooled_delta` são sempre nulos.

## Como executar

Checagens sintéticas, incluídas no `selftest` das dobras por sujeito:

```bash
uv run python -m gatefall.train.subject_folds selftest
```

Elas cobrem as probabilidades por janela e por sujeito, a reprodutibilidade
e o isolamento do sorteio, o critério nos limites, o caminho canônico do
engine sem o hook, a ordem de leitura igual à dos sorteios e a mesma
inicialização do controle, o isolamento por sujeito, a avaliação limpa de
`final.train`, a proteção dos controles, dos runs da Etapa 4C e dos resumos
anteriores, a validação da proveniência e o resumo pareado.

Com os controles da Etapa 4B já treinados, na raiz do checkout que contém os
dados e os runs:

```bash
uv run python -m gatefall.train.subject_folds plan --variant subject_balanced
uv run python -m gatefall.train.subject_folds train --variant subject_balanced
uv run python -m gatefall.train.subject_folds summarize --variant subject_balanced
```

Como na Etapa 4C, rode a variante com os mesmos caminhos relativos usados no
controle, a partir da mesma raiz.

## Execução registrada (não canônica)

2026-10-16, branch `feat/subject-balanced-sampling` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados e controles do checkout principal, a partir da raiz desse checkout, em
GPU CUDA (GTX 1650). O código, incluindo `DECISION_RULE`, foi commitado antes
dos treinos. `train --variant subject_balanced` validou os três controles e
treinou os três runs balanceados sem `--force`. Uma segunda chamada pulou os
três runs íntegros. `summarize --variant subject_balanced` validou controles e
runs balanceados e recalculou o sorteio de cada dobra. As entradas,
`folds.json` (`7db0ecc0…`) e a composição são as da
[execução da Etapa 4B](subject-folds.md#execucao-registrada-nao-canonica).

Macro-F1 restrita no grupo retido, braço A, seed 42, checkpoint da última
época. Cada linha é uma dobra; as linhas não devem ser agregadas nem
comparadas entre si.

| Dobra | Controle (4B) | Balanceado (5A) | Δ |
| --- | ---: | ---: | ---: |
| `retain_s0` | 0,5584 | 0,5416 | −0,0168 |
| `retain_s1` | 0,6576 | 0,6797 | +0,0221 |
| `retain_s3-4-6-8` | 0,6157 | 0,6163 | +0,0006 |

O critério congelado **não foi atendido**: nenhuma dobra chega a +0,03, e o
menor delta, −0,0168, fica dentro do limite de −0,02. A amostragem
balanceada por sujeito não mostrou ganho de generalização para sujeitos
retidos com esta receita e esta seed. O único delta positivo relevante,
em `retain_s1`, vem de uma dobra que retém um único sujeito, e o de
`retain_s0`, a dobra cujo treino perde o sujeito dominante, é negativo.

`standing`, `other` e `fall` (Δ F1, suporte em janelas no retido):

| Dobra | `standing` | `other` | `fall` |
| --- | ---: | ---: | ---: |
| `retain_s0` | −0,048 (496) | +0,012 (2200) | −0,000 (943) |
| `retain_s1` | +0,138 (70) | +0,031 (478) | −0,010 (461) |
| `retain_s3-4-6-8` | −0,009 (107) | −0,016 (1999) | +0,015 (589) |

Nenhuma das três classes muda de forma consistente entre as dobras. O maior
delta, de `standing` em `retain_s1`, vem de 70 janelas de cinco vídeos de um
único sujeito.

Janelas de treino por sujeito (stride 4) e fração sorteada nas 30 épocas:

| Dobra | Sujeito: janelas limpas (fração limpa → fração sorteada) |
| --- | --- |
| `retain_s0` | 1: 1000 (0,379 → 0,198); 3: 473 (0,179 → 0,199); 4: 467 (0,177 → 0,201); 6: 423 (0,160 → 0,202); 8: 275 (0,104 → 0,201) |
| `retain_s1` | 0: 2581 (0,612 → 0,199); 3: 473 (0,112 → 0,199); 4: 467 (0,111 → 0,202); 6: 423 (0,100 → 0,200); 8: 275 (0,065 → 0,201) |
| `retain_s3-4-6-8` | 0: 2581 (0,721 → 0,498); 1: 1000 (0,279 → 0,502) |

Ao longo das 30 épocas, toda janela de treino foi sorteada ao menos uma vez
nas três dobras. O sujeito 8 em `retain_s0` e `retain_s1`, com 275 janelas,
teve cada janela sorteada em média cerca de 1,9 e 3,1 vezes por época. A
amostragem também mudou a distribuição de classes vista no treino; por
exemplo, em `retain_s1`, `other` passou de 1052 janelas limpas para cerca de
1332 sorteadas por época, e `sit_down`, de 121 para cerca de 62. As
contagens por classe de cada dobra estão em
`summary/subject_balanced/subject_fold_subject_balanced_summary.md`.

Os SHA-256 dos 852 arquivos em `runs/`, `data/processed/`, `data/labels/`,
`data/features/le2i/pose/` e `src/gatefall/features/stats/` existentes antes
dos treinos foram iguais depois, incluindo `folds.json`, as estatísticas, os
controles da Etapa 4B, os runs da Etapa 4C, os resumos anteriores e o
diagnóstico da Etapa 4D. As saídas não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `retain_s0/.../baseline_a_subject_balanced/.../checkpoint.pt` | `4412a6692b2e19b5018acc039d16b0c0b6de21965e7c52aa3e9190527d9e4a11` |
| `retain_s1/.../baseline_a_subject_balanced/.../checkpoint.pt` | `e1b68a4b869f7a1aa664f96a84785c79a6ad77ae404d4972035df66f2c9109c9` |
| `retain_s3-4-6-8/.../baseline_a_subject_balanced/.../checkpoint.pt` | `a8acca6ab6be34d7228a9d27d33d5774c636d18d1d64b8d0d5b92ffbcb70c26a` |
| `summary/subject_balanced/subject_fold_subject_balanced_summary.json` | `a439010625d2b8b98301c91bd08db9ef51b9b7f82fbc3b6edda7d0e8bf8f5fd3` |

## Limitações

- O critério é uma regra de decisão predeclarada, não um teste de
  significância. Ele não valida o alvo de 0,800 nem estima o desempenho no
  `test` oficial, que não é lido. A variante segue não canônica.
- Três dobras de tamanhos muito diferentes e uma seed de treino. A variação
  entre seeds não é estimada, e deltas desta ordem podem vir só dela.
- Desequilíbrio entre sujeitos: em `retain_s3-4-6-8` só há dois sujeitos de
  desenvolvimento, e nas outras dobras o sujeito com menos janelas é
  repetido várias vezes por época. O efeito mistura menos domínio de poucos
  sujeitos com mais sobreajuste aos vídeos dos sujeitos pequenos.
- A amostragem por sujeito muda também a distribuição de classes vista no
  treino. Os pesos de classe continuam os do treino limpo completo, como no
  controle, e não compensam essa mudança.
- Janelas sobrepostas: no treino (stride 4), sorteios repetidos de janelas
  vizinhas reforçam a redundância; no retido (stride 1), as contagens por
  janela não são observações independentes.
