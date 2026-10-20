# Treino do piloto GMDCSA24 com rótulos fracos (T e S, braço A)

`gatefall.train.gmdcsa24_weak_label` implementa a etapa de treino da
[autorização do passo 4](../analysis/gmdcsa24-weak-label-protocol.md#autorizacao-do-passo-4):
a junção das janelas de T ou S com o treino Le2i-CS, a padronização e os pesos
da perda de cada lado, e os portões de pré-treino exigidos em
[Portões antes de cada etapa](../analysis/gmdcsa24-weak-label-protocol.md#portoes-antes-de-cada-etapa).
O módulo não muda nenhuma decisão do protocolo: fontes, exclusões, T, S, a
margem de 0,5 s, as seeds, a receita congelada, o controle pareado e o
critério de decisão continuam os de lá.

**Estado:** implementação apenas. Neste PR não rodou nenhum `prepare` ou
`train` real, e nenhuma padronização de T ou S, peso da perda ou métrica de
modelo foi calculada sobre os dados reais. Sobre os dados reais rodaram só os
comandos de leitura `coverage` e `preflight`, sem montar features (ver
[Execução de leitura sobre os dados reais](#execucao-de-leitura-sobre-os-dados-reais)).

## Entradas

- **Adapter:** a saída de `data/processed/gmdcsa24/`, com as seis condições
  do `adapter validate`.
- **Pose:** a saída de `data/features/gmdcsa24/pose/`, com as seis condições
  do `pose_extract validate`. A extração dos 145 vídeos elegíveis foi
  informada pelo responsável no commit `5fc8204` (ver
  [Pose do GMDCSA24](../data/gmdcsa24-pose.md#extracao-informada-para-o-commit-5fc8204)).
- **Le2i-CS:** a grade de `data/processed/le2i/`, a pose de
  `data/features/le2i/pose/` e o JSON de padronização do controle, só leitura.
- **Controle:** o run do braço A da mesma seed, com `config.yaml`,
  `metrics.json`, `checkpoint.pt` e `alarm_protocol.yaml`.

A proveniência fica presa à extração validada: o adapter e a pose precisam ter
`code.commit = 5fc8204ae474d36d849e87000b5d4b52cd3d5177` e
`code.dirty = false`. O código de treino pode estar em outro commit; um commit
novo não exige refazer o adapter nem a extração. O checkout do treino precisa
estar limpo, e o commit dele fica registrado.

## Montagem de T e S

- O treino Le2i-CS é montado como no controle, com `build_window_index` em
  `TRAIN_STRIDE` sobre o split `train`: 5 219 janelas.
- As janelas do GMDCSA24 são as de `windows_t.parquet` ou
  `windows_s.parquet`, sem filtro novo: T = 1 973 e S = 1 562.
- O treino aumentado é o treino Le2i-CS seguido das janelas do tratamento:
  7 192 janelas em T e 6 781 em S. Qualquer outro total bloqueia o treino.
- As features são as do braço A (`build_pose_features`), lidas da pose do
  próprio dataset de cada janela. A leitura recusa um vídeo cujas features não
  têm o K registrado no índice de janelas.
- `val` é a do controle (`PoseWindowDataset` sobre o split `val` do Le2i-CS em
  `EVAL_STRIDE`), com as features Le2i de sempre. `test` não é montado.

## Padronização e pesos da perda

São os três itens que podem diferir do controle, calculados só sobre o treino
aumentado de cada lado:

- **Manifesto de treino:** `manifest.json` em
  `data/processed/gmdcsa24_training/{t,s}/`, com o SHA-256 do índice de treino
  combinado e do índice Le2i-CS, as contagens por classe, os hashes de `val` e
  `test` e a proveniência do adapter e da pose.
- **Padronização:** `compute_train_stats` sobre todas as janelas do treino
  aumentado, com o mesmo stride, as mesmas dimensões excluídas e o mesmo
  limite de desvio mínimo, gravada em `pose_stats.json` no mesmo diretório.
  O campo `frames_hash` desse JSON é o SHA-256 do índice de treino combinado,
  já que o treino não vem de uma única grade.
- **Pesos da `CrossEntropyLoss`:** a frequência inversa do engine do braço A,
  com peso zero para classe ausente (`lie_down` e `lying`). O manifesto
  registra os pesos, e o engine os recalcula do mesmo treino no início do
  treino.

O JSON do controle (`src/gatefall/features/stats/pose_le2i_cs.json`) e os
checkpoints do controle não mudam. O `prepare` mostra a diferença absoluta
média de média e desvio por bloco de features entre o JSON do tratamento e o
do controle.

## Portões de pré-treino

`preflight` e `train` conferem, para o tratamento e a seed pedidos:

| Condição | O que é conferido |
| --- | --- |
| checkout | Checkout do GateFall sem arquivo modificado ou não rastreado |
| proveniência | Integridade e proveniência do adapter, as seis condições do `pose_extract validate` (inclusive os hashes publicados de T, S e das candidatas) e `code` do adapter e da pose igual à extração validada |
| W1, W2, V5 | As do `adapter validate`, com as exclusões e os agregados publicados |
| V2 | A do adapter; hashes atuais dos índices de `val` e `test` do Le2i-CS iguais aos publicados; índice de `val` do treino igual ao publicado; treino só com `split = train` e sem vídeo de `val`/`test`; nenhum `video_id` do GMDCSA24 na grade do Le2i-CS; JSON do controle com as 5 219 janelas e o `frames_hash` da grade Le2i-CS; `alarm_protocol.yaml` do controle igual a `BASELINE_A_ALARM_PROTOCOL`; saída do tratamento fora dos caminhos do Le2i |
| V3 | Controle da mesma seed (42 a 46) válido para a receita congelada e com o `checkpoint_sha256` do [sumário multi-seed](../analysis/multiseed-summary.md#resultado-da-execucao-real); com `--control-retrained`, um checkpoint diferente só passa se `torch_version` e dispositivo do controle forem os desta máquina |
| V4 | Fingerprint da configuração do controle igual ao publicado e configuração do tratamento igual à do controle fora de `standardization_stats_path` e `standardization_stats_sha256` |
| tratamento | Janelas de T ou S, S contido em T, totais de 5 219, 1 973 ou 1 562 e 7 192 ou 6 781, `env` e `split` do GMDCSA24, nenhuma janela com borda replicada, `lie_down`, `lying` ou `IGNORE_LABEL`, e K de cada vídeo igual ao da pose |
| padronização e pesos | Manifesto preparado igual ao recalculado (proveniência, índices, contagens e pesos), sem campo a mais ou a menos; `standardization` com `path`, `sha256`, `window_count` e `frames_hash` iguais aos do JSON preparado e da padronização recalculada; `code` com `commit` completo, descendente da extração validada e ancestral do HEAD atual, e `dirty = false`; `created_at` válido; JSON igual à padronização recalculada sobre o treino aumentado |
| saídas | `--training-dir` fora das entradas (adapter, pose, grade, pose e JSON do Le2i-CS), do controle escolhido e da raiz dos runs do tratamento, comparando caminhos resolvidos e relações de ancestral e descendente |

`prepare` exige as condições que não dependem da seed (checkout,
proveniência, W1, W2, V2, V5 e tratamento) e o isolamento de `--training-dir`.
Qualquer falha bloqueia o comando sem gravar nada. Uma preparação existente
que passa em "padronização e pesos" é pulada, com ou sem `--force`, porque o
`treatment.json` dos runs registra o SHA-256 do manifesto preparado; `--force`
só recria uma preparação que reprova. Depois da
[decisão registrada](../eval/gmdcsa24-weak-label-val.md#registro) ou de qualquer
acesso a [`test`](../eval/gmdcsa24-weak-label-test.md#depois-de-acessar-test),
`prepare` e `train` recusam gravar, mesmo com `--force`; preparação conferida e
run completo continuam sendo pulados.

## Isolamento de `test`

O treino canônico do braço A avalia `test` no fim de cada run. O caminho do
piloto não avalia:

- `run_training` aceita `test_source=None`; nesse caso `test` não é montado,
  lido nem avaliado. O treino canônico continua passando `test`, e o
  checkpoint dele não muda.
- O `metrics.json` de um run de T ou S só tem `train` e `val`. O validador do
  run recusa `test` nesse caso, e o validador canônico recusa o run sem `test`,
  então avaliadores e sumários do braço A não o consomem por engano.
- `test` só aparece pelo hash do índice de janelas, calculado da grade, como o
  adapter já fazia. Nenhuma feature, predição ou métrica de `test` é calculada.

O módulo não tem comando de avaliação em `test`. A avaliação de eventos em
`val` e a decisão GO/NO-GO ficam em
[Eventos em `val` e decisão](../eval/gmdcsa24-weak-label-val.md), que também
não avalia `test`. As métricas descritivas de `test`, calculadas uma vez
depois da decisão em `val`, ficam para uma implementação própria.

## Saídas

```text
data/processed/gmdcsa24_training/{t,s}/
├── manifest.json
└── pose_stats.json
runs/local/gmdcsa24_weak_label/{t,s}/seed_<n>/
├── config.yaml
├── metrics.json      # só train e val
├── checkpoint.pt
└── treatment.json
```

As duas raízes são ignoradas pelo Git. Os comandos recusam saída dentro, ou
acima, de `data/processed/le2i*/`, da saída do adapter, de `data/features/`,
`data/labels/`, `data/raw/`, das estatísticas de padronização, de
`runs/reference/`, de `runs/local/le2i*/` e da revisão de proveniência.
`treatment.json` liga o run ao manifesto preparado, ao índice de treino, aos
pesos, ao controle pareado e ao commit do código. Nada disso é sincronizado
com o Google Drive.

`train` não tem `--run-dir`: cada tratamento e seed têm um único local de
execução, `runs/local/gmdcsa24_weak_label/<t|s>/seed_<n>/` na raiz do
repositório GateFall. `train` e `preflight` exigem que o diretório de execução
resolvido seja essa raiz, a do código importado, e não só que contenha as
entradas; um diretório ancestral do repositório é recusado mesmo que todas as
entradas resolvam sob ele. Também recusam qualquer entrada (adapter, pose,
`--training-dir`, grade, pose e JSON do Le2i-CS e controle) e o run canônico
cujo caminho resolvido, inclusive por link simbólico, fique fora da raiz.
Assim outra raiz de saída ignorada não reaproveita as mesmas entradas para
abrir um segundo run da mesma seed; isso só seria possível com outro clone do
repositório e cópia de todos os dados. Os selftests usam uma raiz sintética
injetada pela API Python, sem opção correspondente no CLI.

Antes de gravar, `prepare` e `train` também recusam, com caminhos resolvidos
(inclusive links simbólicos) e em qualquer relação de ancestral ou
descendente:

- `--training-dir` sobre as entradas, o controle escolhido ou
  `runs/local/gmdcsa24_weak_label/`;
- o run sobre as entradas, o controle escolhido, `--training-dir` ou
  `data/processed/gmdcsa24_training/`, com subdiretórios ou dentro de outro
  run.

## Um run por seed

`train` classifica o run canônico depois do preflight:

| Estado | Sem `--force` | Com `--force --recovery-reason` |
| --- | --- | --- |
| Ausente | Treina | Recusa: não há run a recuperar |
| Completo: run do braço A válido sem `test` e `treatment.json` igual ao recalculado | Pula | Recusa: o protocolo admite um único run por seed |
| Registro divergente: run válido com `treatment.json` ilegível ou diferente | Recusa | Recusa: exige inspeção manual |
| Incompleto: artefato ausente ou inválido, ou `treatment.json` ausente | Recusa | Refaz, se a identidade confere |

O `treatment.json` é conferido campo a campo: tratamento, seed, SHA-256 do
manifesto preparado e do JSON de padronização, índice de treino, pesos, hash
de `val`, `evaluated_splits`, checkpoint do run e do controle e aceitação do
controle, sem campo a mais. O `code` precisa ter `commit` completo,
descendente do commit da preparação e ancestral do HEAD atual, e
`dirty = false`.

A recuperação só refaz o run da mesma seed, tratamento e configuração: o
`config.yaml` existente precisa ser legível e igual à configuração do
tratamento, e um `treatment.json` legível precisa ter o mesmo tratamento e a
mesma seed. `--force` e `--recovery-reason` só valem juntos. O run refeito
grava em `treatment.json` o bloco `recovery`, com o motivo, a data e o
SHA-256 de cada arquivo substituído.

## Como executar

Na raiz de um checkout limpo, para T e depois para S:

```bash
uv run python -m gatefall.train.gmdcsa24_weak_label coverage
uv run python -m gatefall.train.gmdcsa24_weak_label prepare --treatment T
uv run python -m gatefall.train.gmdcsa24_weak_label preflight --treatment T --seed 42 \
  --control-run-dir runs/local/le2i/baseline_a_multiseed/seed_42
uv run python -m gatefall.train.gmdcsa24_weak_label train --treatment T --seed 42 \
  --control-run-dir runs/local/le2i/baseline_a_multiseed/seed_42
```

`train` roda o mesmo preflight antes de treinar e grava só em
`runs/local/gmdcsa24_weak_label/t/seed_42/`, o run canônico. Pelo
[critério de parada 3](../analysis/gmdcsa24-weak-label-protocol.md#criterios-de-parada),
`--force` só serve para repetir a mesma seed depois de uma falha técnica, com
o motivo registrado (ver [Um run por seed](#um-run-por-seed)):

```bash
uv run python -m gatefall.train.gmdcsa24_weak_label train --treatment T --seed 42 \
  --control-run-dir runs/local/le2i/baseline_a_multiseed/seed_42 \
  --force --recovery-reason "<falha técnica observada>"
```

## Execução de leitura sobre os dados reais

Em 2026-10-19, no commit deste PR, com o checkout limpo e na raiz do checkout
principal (dados reais), rodaram só comandos que não montam features:

- `coverage`: as seis condições do `pose_extract validate` passam.
- `preflight` para T e S nas seeds 42 a 46, com os controles de
  `runs/local/le2i/baseline_a_multiseed/seed_<n>/`: checkout, proveniência,
  W1, W2, V2, V3, V4, V5 e tratamento passam nos dez casos. Só
  "padronização e pesos" reprova, porque nenhum `prepare` rodou; por isso a
  comparação de V4 entre a configuração do tratamento e a do controle ainda
  não foi feita sobre os dados reais. Repetido depois das salvaguardas de
  ciclo de vida e de isolamento, com o mesmo resultado e "saídas" passando
  nos dez casos. Com a exigência da raiz do repositório, o mesmo preflight,
  executado do checkout principal com o ambiente de um worktree, reprova
  "saídas" para T e S na seed 42: a raiz do código importado é o worktree.
  `preflight` e `train` reais rodam da raiz do checkout cujo código está
  instalado no ambiente usado.

Nenhum arquivo foi gravado em `data/processed/gmdcsa24_training/` nem em
`runs/local/gmdcsa24_weak_label/`.

## Cobertura da detecção de pose

`coverage` relata quantos quadros da grade têm pessoa detectada, por vídeo e
por janela de T e S. É descritivo: não cria limiar, não exclui vídeo ou janela
e não muda T nem S. Resultado sobre a pose real:

| Medida | Valor |
| --- | ---: |
| Vídeos | 145 |
| Quadros da grade | 11 710 |
| Quadros com pessoa | 11 421 (97,53%) |
| Vídeos com 100% | 104 |
| Vídeos com algum quadro sem pessoa | 41 |
| Cobertura mínima por vídeo | 0,460 |

| Tratamento | Janelas | Fração média com pessoa | Todas com pessoa | Parciais | Sem pessoa |
| --- | ---: | ---: | ---: | ---: | ---: |
| T | 1 973 | 0,978 | 1 725 | 246 | 2 |
| S | 1 562 | 0,975 | 1 372 | 188 | 2 |

Vídeos com a menor cobertura:

| Vídeo | Quadros com pessoa | Cobertura | Janelas em T | Janelas em S |
| --- | ---: | ---: | ---: | ---: |
| `Subject_1/Fall/02` | 29 de 63 | 0,460 | 10 | 7 |
| `Subject_4/ADL/07` | 25 de 49 | 0,510 | 7 | 7 |
| `Subject_2/Fall/02` | 54 de 84 | 0,643 | 15 | 11 |
| `Subject_2/Fall/04` | 67 de 102 | 0,657 | 20 | 14 |
| `Subject_4/Fall/14` | 35 de 52 | 0,673 | 7 | 4 |
| `Subject_4/Fall/04` | 36 de 47 | 0,766 | 6 | 5 |
| `Subject_1/Fall/01` | 54 de 68 | 0,794 | 11 | 7 |
| `Subject_4/Fall/10` | 40 de 50 | 0,800 | 7 | 4 |

Os outros 33 vídeos com algum quadro sem pessoa têm cobertura de 0,818 a
0,992. Quadros sem pessoa recebem a imputação do braço A, como no Le2i.

## Selftest

`selftest` (na CI) grava saídas sintéticas de adapter e de pose com o clone e
o modelo falso dos selftests anteriores, monta um Le2i-CS sintético e treina
um controle e um run de T com a receita congelada. Ele cobre:

- pertencimento a T e S e os totais do treino aumentado;
- alinhamento entre janela e features, inclusive K divergente;
- padronização e pesos calculados só sobre o treino aumentado, próprios de T
  e de S, sem mudar o JSON do controle;
- isolamento de `val` e `test`, inclusive rótulo de `val` alterado e
  `video_id` do GMDCSA24 na `val`;
- recusa de proveniência adulterada (commit, `dirty`, tabela, manifesto da
  pose, manifesto preparado e JSON) e aceitação de outro commit de treino em
  checkout limpo;
- cada campo de `standardization`, `code` e `created_at` do manifesto
  preparado alterado ou ausente reprovando só "padronização e pesos", e a
  preparação inalterada passando;
- recusa de `--training-dir` e de run sobrepostos, por link simbólico, com
  outros runs ou fora do layout, sem gravar nem alterar artefato;
- `train` sem `--run-dir`, com o run canônico aceito na raiz do repositório e
  um segundo run da mesma seed recusado a partir de outra raiz ignorada ou de
  um ancestral do repositório, inclusive com as entradas alcançadas por link
  simbólico, e com o run canônico escapando da raiz por link simbólico;
- run completo nunca refeito com `--force`, `treatment.json` adulterado
  recusado com e sem `--force`, run incompleto recusado sem motivo e
  recuperado com motivo registrado, e recuperação de outra configuração ou
  seed recusada;
- controle pareado (checkpoint, seed, receita, fingerprint, alarme e
  `--control-retrained`) e V4;
- nenhuma feature de `test` lida em preflight e treino, run sem `test` e o
  controle canônico ainda avaliando `test`;
- cobertura descritiva sem mudar T nem S;
- execução a partir de um checkout de trabalho com `decision.json` e o
  marcador `test_descriptive/` registrados e ilegíveis, como o real depois da
  decisão: o mundo sintético roda com o repositório temporário como diretório
  de execução e não lê, grava nem apaga nada nesse checkout.

## O que fica para depois

- Os dez treinos sobre os dados reais. O responsável pelo projeto informou
  que o `prepare` real de T e S terminou, com manifestos e JSON de
  padronização registrados por SHA-256, e que os dez `preflight` passaram nas
  11 condições; essa evidência não foi reexecutada nos PRs.
- A avaliação de eventos em `val` dos dez runs e a decisão GO/NO-GO, com os
  comandos de [Eventos em `val` e decisão](../eval/gmdcsa24-weak-label-val.md).
- As métricas descritivas de `test`, calculadas uma vez depois da decisão.
