# Treino — Perda simulada de detecção nas dobras por sujeito do Le2i-CS (Etapa 4C)

`src/gatefall/train/subject_folds/detection_dropout.py` acrescenta às
[dobras por sujeito da Etapa 4B](subject-folds.md) uma única intervenção
predeclarada de robustez de pose: a perda simulada de detecção da pessoa no
treino de desenvolvimento do braço A. Cada run aumentado é comparado ao
controle da Etapa 4B da mesma dobra, seed e receita.

A intervenção é **exploratória e não canônica**. Ela não muda as
configurações A/B/C, os splits oficiais, os HDF5 de pose, o código de pose,
as estatísticas oficiais nem os runs da Etapa 4B. O resultado não é uma
afirmação de significância estatística nem uma validação do alvo de 0,800,
e `val` e `test` oficiais nunca entram em nenhuma seleção.

## O que fica igual ao controle

- As três dobras congeladas, `folds.json`, a seed 42 e a estatística de pose
  de cada dobra, ajustada só no treino de desenvolvimento limpo.
- A receita de `BASELINE_A_CONFIG`: arquitetura, AdamW, pesos de classe, 30
  épocas e checkpoint da última época. `config.yaml` só difere do controle
  em `run_name` (`baseline_a_detection_dropout`).
- A ordem de leitura do treino e a inicialização do modelo. O sorteio da
  intervenção usa um gerador próprio e não consome o RNG global do NumPy, do
  PyTorch nem do `random`, então o embaralhamento do `DataLoader` e os pesos
  iniciais são os do controle.
- A avaliação do grupo retido, nas posições `val` e `test` do engine, com as
  mesmas janelas limpas da Etapa 4B.

## A intervenção

Cada janela amostrada do treino de desenvolvimento (stride 4) recebe, de
forma independente em cada leitura, a intervenção com probabilidade 0,20.
Quando sorteada, exatamente três quadros consecutivos da janela de 24 viram
detecções ausentes, numa posição uniforme entre os deslocamentos válidos.

- **Ausência genuína.** A máscara é aplicada à pose bruta, numa cópia de
  `PoseArrays`, no formato que a extração grava quando o YOLO-Pose não
  encontra ninguém: `person_found` falso e keypoints e bbox zerados. As 134
  features são recalculadas pelo pipeline inalterado (imputação, dt efetivo
  e cinemática), então os quadros mascarados têm confiança zero, a pose
  segura a última observação real, a velocidade é zero durante a lacuna e a
  reaparição usa o dt acumulado. Se a máscara cobre a primeira detecção do
  vídeo, o trecho zerado antes da primeira observação se estende.
- **Contexto causal.** As features são recalculadas sobre o prefixo do vídeo
  até o último quadro da janela. O forward-fill e o dt efetivo enxergam a
  última observação anterior à janela, e nada posterior a ela entra. A
  janela resultante é idêntica à fatia das features do vídeo inteiro com os
  mesmos quadros mascarados.
- **Posições válidas.** Só são sorteáveis os deslocamentos cujos três
  quadros são quadros reais distintos do vídeo, sem a replicação de borda do
  início. Janelas que terminam antes do terceiro quadro do vídeo ficam
  limpas.
- **Gerador isolado.** O sorteio usa `numpy.random.default_rng` semeado por
  (`RANDOM_STREAM_KEY`, seed, índice da janela, visita da janela). O
  resultado é reprodutível e não depende da ordem de leitura.
- **Só o treino.** O retido e o cache de features limpas não são tocados. A
  pose bruta é lida com um loader restrito aos vídeos de desenvolvimento da
  dobra.

`final.train` do run aumentado é medido sobre janelas com a intervenção
sorteada na última leitura e não é comparável ao do controle. Só o retido
entra nos deltas.

## Critério exploratório congelado

O critério foi fixado em `DECISION_RULE` e gravado na proveniência de cada
run antes dos treinos reais:

- o delta pareado de macro-F1 restrita no retido (aumentado menos controle)
  é positivo em pelo menos duas das três dobras; e
- nenhum delta fica abaixo de −0,02.

Os deltas não são agregados entre dobras. `standing` e `fall` são relatados
à parte, com o suporte de cada dobra.

## Diretório e proveniência

```text
runs/local/le2i/subject_folds/
├── folds.json                                  # da Etapa 4B, só lido
├── retain_<sujeitos>/
│   ├── stats/{pose,provenance}.json            # da Etapa 4B, só lidos
│   └── runs/
│       ├── baseline_a/seed_42/                 # controle da Etapa 4B, só lido
│       └── baseline_a_detection_dropout/seed_42/
│           ├── config.yaml, metrics.json, checkpoint.pt
│           └── fold_run.json
└── summary/
    ├── baseline/                               # da Etapa 4B, não regravado
    └── detection_dropout/subject_fold_detection_dropout_summary.{json,md}
```

O `fold_run.json` do run aumentado tem os campos do controle, com
`stage: "4C"` e `variant: "detection_dropout"`, e acrescenta:

- `intervention`: probabilidade, número de quadros, janela, regra de
  posição, representação da ausência, contexto causal, fluxo aleatório e a
  regra de decisão;
- `control`: o run do controle e o SHA-256 do seu `fold_run.json`, que já
  liga `config.yaml`, `metrics.json` e `checkpoint.pt` do controle.

`train --variant detection_dropout` valida o controle de todas as dobras
pedidas antes de treinar qualquer uma, com a mesma validação do resumo da
Etapa 4B. Um controle ausente ou inconsistente interrompe a execução sem
treiná-lo, repará-lo ou substituí-lo. Um run aumentado cuja intervenção,
regra ou hash de controle diverja do esperado é recusado sem `--force`.

## Resumo pareado

`summarize --variant detection_dropout` é somente leitura. Ele valida cada
controle como o resumo da Etapa 4B e cada run aumentado contra a
configuração, a intervenção congelada e o `fold_run.json` atual do
controle. Em seguida, exige suporte do retido idêntico nos dois runs e
relata, por dobra, a macro-F1 restrita dos dois runs e o delta, a F1 e o
delta de cada classe restrita com o suporte, o critério congelado,
`standing` e `fall` e as limitações. `pooled_score` e `pooled_delta` são
sempre nulos.

## Como executar

Checagens sintéticas, incluídas no `selftest` das dobras por sujeito:

```bash
uv run python -m gatefall.train.subject_folds selftest
```

Elas cobrem a máscara da pose bruta e sua equivalência a uma perda lida de
HDF5, a causalidade das features e a coerência nas bordas da janela, a
avaliação limpa e idêntica à do controle, a mesma ordem de treino e a mesma
inicialização do controle, o sorteio determinístico e isolado, o isolamento
do treino, a proteção dos controles, a validação da proveniência e o resumo
pareado com o critério.

Com os controles da Etapa 4B já treinados, na raiz do checkout que contém os
dados e os runs:

```bash
uv run python -m gatefall.train.subject_folds plan --variant detection_dropout
uv run python -m gatefall.train.subject_folds train --variant detection_dropout
uv run python -m gatefall.train.subject_folds summarize --variant detection_dropout
```

`config.yaml` registra o caminho da estatística da dobra como foi passado.
Rode a variante com os mesmos caminhos relativos usados no controle, a
partir da mesma raiz; um caminho diferente faz a validação do controle
falhar, sem gravar nada.

## Execução registrada (não canônica)

2026-10-15, branch `feat/pose-detection-dropout` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados e controles do checkout principal, a partir da raiz desse checkout, em
GPU CUDA (GTX 1650). O código, incluindo `DECISION_RULE`, foi commitado antes
dos treinos. `train --variant detection_dropout` validou os três controles
e treinou os três runs aumentados sem `--force`. Uma segunda chamada pulou os
três runs íntegros. `summarize --variant detection_dropout` validou controles
e runs aumentados. As entradas, `folds.json` (`7db0ecc0…`) e a composição são
as da [execução da Etapa 4B](subject-folds.md#execucao-registrada-nao-canonica).

Macro-F1 restrita no grupo retido, braço A, seed 42, checkpoint da última
época. Cada linha é uma dobra; as linhas não devem ser agregadas nem
comparadas entre si.

| Dobra | Controle (4B) | Aumentado (4C) | Δ |
| --- | ---: | ---: | ---: |
| `retain_s0` | 0,5584 | 0,5611 | +0,0026 |
| `retain_s1` | 0,6576 | 0,6577 | +0,0001 |
| `retain_s3-4-6-8` | 0,6157 | 0,6174 | +0,0018 |

O critério congelado foi **atendido** formalmente: delta positivo nas três
dobras e menor delta +0,0001. Os deltas, porém, são todos menores que 0,003,
e o de `retain_s1` é positivo só na quarta casa decimal. Com uma seed por
dobra, eles estão bem abaixo da variação entre seeds esperada, que não foi
estimada aqui. A leitura honesta é que a intervenção **não piorou** o retido
limpo, não que o melhorou de forma material.

`standing` e `fall` (Δ F1, suporte em janelas no retido):

| Dobra | `standing` | `fall` |
| --- | ---: | ---: |
| `retain_s0` | −0,009 (496) | +0,009 (943) |
| `retain_s1` | +0,029 (70) | −0,019 (461) |
| `retain_s3-4-6-8` | +0,002 (107) | −0,000 (589) |

Nenhuma das duas classes muda de forma consistente entre as dobras. O maior
delta de `standing` vem de 70 janelas de um único sujeito. A F1 e o delta
das oito classes restritas de cada dobra estão em
`summary/detection_dropout/subject_fold_detection_dropout_summary.md`.

Os SHA-256 de `runs/reference/`, `data/processed/le2i/`, dos HDF5 de pose,
`src/gatefall/features/stats/`, `runs/local/le2i/` fora de
`subject_folds/`, `runs/local/le2i_cv/`, `runs/local/le2i_cv_internal_folds/`,
`runs/local/subject_fold_feasibility/` e, em `subject_folds/`, de
`folds.json`, das estatísticas, dos três controles e de `summary/baseline/`
foram iguais antes e depois. As saídas não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `retain_s0/.../baseline_a_detection_dropout/.../checkpoint.pt` | `49d7f1fec857f89faab395545365f2226374ab5c28aa90488e2ee24b06526b6d` |
| `retain_s1/.../baseline_a_detection_dropout/.../checkpoint.pt` | `28362314aee5c2dc6c1a5bd0e20c3fc157c11ac57ccf0d7ddc2061cbf991ae85` |
| `retain_s3-4-6-8/.../baseline_a_detection_dropout/.../checkpoint.pt` | `cf7e5d75a07da54349126ea26755654835a5eda09b7789f6e672ce3d3c83da8b` |
| `summary/detection_dropout/subject_fold_detection_dropout_summary.json` | `ee63dc5ec821fc0ba4e3a32c520cdb18a1f9a8f81eef79be8eca2cac973ad580` |

## Limitações

- O critério é uma regra de decisão predeclarada, não um teste de
  significância. Ele não valida o alvo de 0,800 nem estima o desempenho no
  `test` oficial, que não é lido.
- Três dobras de tamanhos muito diferentes e uma seed. A variação entre
  seeds não é estimada, e deltas desta ordem podem vir só dela.
- O retido é avaliado limpo: o resumo mede o efeito do aumento em dados
  reais, não a robustez a perdas simuladas no retido. Essa robustez é medida
  à parte, sem retreino, na [Etapa 4D](../analysis/subject-fold-dropout-robustness.md).
- A intervenção simula só lacunas curtas de três quadros, com probabilidade
  fixa. Ela não reproduz a distribuição real das perdas de detecção, que se
  concentram em `fall` e em Home_01.
- Contagens por janela usam janelas sobrepostas e não são observações
  independentes.
