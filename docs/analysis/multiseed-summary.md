# Análise — Sumário multi-seed

`src/gatefall/eval/analysis/multiseed_summary.py` agrega treinos independentes de um
braço selecionado entre A, B0, B1, C0 e C1 no protocolo `le2i`, produzindo
estatísticas descritivas (n/mean/desvio-padrão amostral/min/max). Todas as
armas agregam classificação e evento. É somente leitura:
nenhum artefato dos runs de entrada é modificado.

## Fronteira com o bootstrap agrupado por sujeito

Este módulo é conceitualmente separado de
[Bootstrap agrupado por sujeito](grouped-bootstrap.md) e as duas noções de
variação nunca se misturam na mesma estatística:

- **Sumário multi-seed** agrega **treinos independentes**: cada seed produz
  seu próprio `run_dir` completo, com checkpoint, config e métricas
  próprios. A variação capturada é a variação entre execuções de treino
  independentes sob a mesma receita congelada (`AGENTS.md`, invariante 1).
- **Bootstrap agrupado por sujeito** reamostra sujeitos a partir de um
  **único checkpoint fixo**, sem retreinar nada. A variação capturada é
  incerteza de amostragem sobre a população de sujeitos avaliados, não
  variação de treino.

Nenhum dos dois seleciona, ranqueia ou promove nenhum run ou seed.

## Contrato de configuração

Todos os `--run-dir` devem pertencer ao mesmo braço e conter
`config.yaml`/`metrics.json`/`checkpoint.pt` íntegros. O validador de treino
do próprio braço verifica configuração, métricas, hashes e checkpoint. A,
B0, B1, C0 e C1 exigem também `alarm_protocol.yaml` igual ao protocolo
congelado e `event_metrics.json` validado com hashes de checkpoint,
métricas de treino e protocolo.

A ferramenta calcula um fingerprint sha256 da configuração de cada run com
`seed` removido. Para B0, B1, C0 e C1, remove também
`trainable_param_count`, campo de auditoria já permitido pelos respectivos
validadores. Todos os demais campos devem ser idênticos. Também rejeita menos
de duas seeds, `--run-dir` duplicado (mesmo path resolvido), seeds duplicadas
entre runs distintos e mistura de braços. `le2i-cv` permanece fora do escopo
para os braços de fusão.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.multiseed_summary selftest
uv run python -m gatefall.eval.analysis.multiseed_summary summarize [--dataset le2i] [--arm {A,B0,B1,C0,C1}] \
  --run-dir PATH [--run-dir PATH ...] --output-dir PATH [--force]
```

`--arm` usa `A` por padrão, preservando o comando anterior. `selftest` roda
checagens sintéticas, sem dataset real nem GPU. `summarize` exige
pelo menos dois `--run-dir` (a flag é repetível), valida cada run
integralmente e escreve `multiseed_summary.json`/`.csv` em `--output-dir`.
Sem `--force`, se algum dos dois arquivos já existir, o comando é pulado e a
mensagem de skip nomeia exatamente o(s) arquivo(s) encontrado(s).

## Métricas agregadas

- Classificação (`classification`): `macro_f1_restricted` e `f1_by_class`
  (por classe restrita) nos splits `train`, `val`, `test`.
- Por classe (`per_class`): `precision`/`recall`/`f1`/`tp`/`tn`/`fp`/`fn`/
  `support`, para as 10 classes do Le2i (não só as restritas), nos splits
  `train`, `val`, `test`.
- Projeção binária queda/caído (`binary_fall_fallen`): `tp`/`tn`/`fp`/`fn`/
  `precision`/`recall`/`specificity`/`f1`/`accuracy`, nos splits `train`,
  `val`, `test`. Esta projeção **não é remedida por inferência**: é derivada
  algebricamente da `confusion_matrix` 10x10 já persistida em
  `metrics.json`, contando como positivo qualquer célula cuja classe
  verdadeira ou predita esteja em `BINARY_POSITIVE_LABELS`
  (`{fall, fallen}`) — a mesma constante e a mesma lógica de projeção que
  `gatefall.train.baseline_a` e `grouped_bootstrap` usam a partir dos
  arrays de predição, agora expressa em termos da matriz de confusão
  (`gatefall.train.shared.metrics.binary_projection_from_confusion_matrix`).
- Evento (`events`, A/B0/B1/C0/C1): todo campo escalar de
  `event_metrics.json[splits][split]` (`sensitivity`, `fall_sensitivity`,
  `fall_or_fallen_sensitivity`, `false_alarms_per_hour`, `n_false_alarms`,
  `latency_seconds_mean`, `latency_seconds_median` e os demais campos
  escalares do bloco de evento — 23 métricas por split no total), nos
  splits `val`, `test`. Só `latency_seconds.per_event` fica de fora da
  agregação, por ser uma lista, não um escalar; ele é preservado bruto por
  seed.

## Esquema de saída

- `multiseed_summary.json`: `arm`, `config_fingerprint_sha256`, `n_seeds`,
  `seeds` (lista por seed com `seed`, `run_dir`, `checkpoint_sha256` e, por
  seed, os blocos completos e verbatim):
  - `classification[split]` (train/val/test): o bloco `final[split]`
    validado de `metrics.json`, na íntegra — `macro_f1_restricted`,
    `f1_by_class`, `support`, `confusion_matrix` 10x10 e o `per_class`
    completo (`tp`/`tn`/`fp`/`fn`/`support`/`precision`/`recall`/`f1`).
  - `binary_fall_fallen[split]` (train/val/test): derivado da
    `confusion_matrix` acima, sem rodar inferência de novo.
  - `events[split]` (val/test): o bloco de split validado de
    `event_metrics.json`, na íntegra, incluindo `latency_seconds.per_event`
    e as métricas binárias de janela.
  - `aggregate`: as mesmas famílias de métricas (`classification`,
    `per_class`, `binary_fall_fallen` e `events` quando aplicável), cada
    métrica escalar reduzida a `{n, mean, std, min, max}`.
- `multiseed_summary.csv`: achatado, uma linha por
  `(split, metric_group, entity, metric)`, colunas `split`, `metric_group`,
  `entity`, `metric`, `n`, `mean`, `std`, `min`, `max`. `metric_group` é um
  de `classification`, `per_class`, `binary`, `events`. `entity` carrega o
  nome da classe nas linhas `f1_by_class` (metric_group `classification`) e
  `per_class`; fica vazio nas demais. `f1_by_class` (por classe restrita,
  metric_group `classification`) é mantido deliberadamente distinto de
  `per_class.f1` (as 10 classes, metric_group `per_class`) — são recortes
  diferentes e não devem ser somados nem confundidos.
- Um `metrics.json` de run sem `confusion_matrix`/`per_class` em algum
  split faz o comando falhar explicitamente, em vez de produzir um sumário
  parcial.

### Regras de agregação

- `std` é o desvio-padrão amostral (`ddof=1`, `statistics.stdev`) e é
  `None` quando `n < 2`.
- Um valor de métrica ausente/indefinido em um run individual é **excluído**
  da agregação daquela métrica, nunca substituído por `0`. Por isso `n` em
  `aggregate` pode ser menor que `n_seeds` para uma métrica específica — os
  dois números têm significados diferentes e não devem ser confundidos.

## Resultado da execução real

Cinco seeds (42, 43, 44, 45, 46) treinadas de forma independente nesta
máquina (GPU GTX 1650, `torch 2.13.0+cu130`), 30 épocas cada, todas sob o
mesmo `BASELINE_A_ALARM_PROTOCOL` congelado e o mesmo fingerprint de
configuração fora da seed
(`b80b439f1878b1f14c2a879dd6bda6f0ce3dc5572229c0f248375609241b030a`). Como
`runs/local/**` está no `.gitignore`, esta seção é o único registro durável
desta execução real; os checkpoints sha256 por seed são:

| Seed | checkpoint_sha256 |
| --- | --- |
| 42 | `78278b1a6a6eb929c27ae071ebf364a2032877eb0adbcb4e4efd2a6a26787ab8` |
| 43 | `701ec9a941ccd3863931757c520cef4b496be2c64257bac43c0aaa16c014993e` |
| 44 | `a068534f0b476ff9d181794623764fab319732ee1d12a9b612385066e361aa36` |
| 45 | `940055adf3613f4e56ed7745da06ebea2200ed68f1922d2c03d5b62b2af4b4fa` |
| 46 | `6ebe278ce22da41afed26d22c061c4b86919e43efb0511c02607276e135e1265` |

O run da seed 42 desta execução reproduziu, checkpoint sha256 idêntico ao
acima, o checkpoint da referência congelada
`runs/reference/le2i/baseline_a/` — checagem de determinismo nesta mesma
máquina/stack, não uma garantia portável entre máquinas/GPUs/drivers. Como
verificação adicional: para a seed 42, o `binary_fall_fallen` derivado da
`confusion_matrix` (sem rodar inferência de novo) é idêntico, campo a campo
e nos três splits, ao `binary_fall_fallen` medido por inferência no
`classification_report.json` do run local que compartilha esse mesmo
checkpoint — evidência de que a derivação algébrica reproduz uma medição
real. Esse `classification_report.json` foi regenerado junto com o retreino
da referência e hoje traz também `class_support_table` e `macro_f1_policy`
(ver [Treino — Braço A (TCN)](../train/baseline-a.md)), campos que não
participam dessa conferência.

### Classificação multiclasse — `macro_f1_restricted`

| Split | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- |
| Treino | 5 | 0,8669 | 0,0056 | 0,8588 | 0,8731 |
| Validação | 5 | 0,6484 | 0,0136 | 0,6336 | 0,6636 |
| Teste | 5 | 0,6109 | 0,0137 | 0,5919 | 0,6251 |

### Projeção binária queda/caído — `binary_fall_fallen`

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Treino | precision | 5 | 0,9726 | 0,0045 | 0,9675 | 0,9798 |
| Treino | recall | 5 | 0,9530 | 0,0032 | 0,9492 | 0,9564 |
| Treino | f1 | 5 | 0,9627 | 0,0017 | 0,9604 | 0,9643 |
| Validação | precision | 5 | 0,8875 | 0,0332 | 0,8317 | 0,9153 |
| Validação | recall | 5 | 0,9284 | 0,0120 | 0,9073 | 0,9375 |
| Validação | f1 | 5 | 0,9072 | 0,0176 | 0,8815 | 0,9231 |
| Teste | precision | 5 | 0,8147 | 0,0232 | 0,7911 | 0,8426 |
| Teste | recall | 5 | 0,9008 | 0,0097 | 0,8885 | 0,9130 |
| Teste | f1 | 5 | 0,8553 | 0,0091 | 0,8453 | 0,8681 |

### Evento — `event_metrics`

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Val | sensitivity | 5 | 0,9846 | 0,0344 | 0,9231 | 1,0000 |
| Val | fall_sensitivity | 5 | 0,9846 | 0,0344 | 0,9231 | 1,0000 |
| Val | fall_or_fallen_sensitivity | 5 | 0,9846 | 0,0344 | 0,9231 | 1,0000 |
| Val | false_alarms_per_hour | 5 | 20,77 | 22,57 | 0,00 | 51,92 |
| Val | n_false_alarms | 5 | 1,2 | 1,3 | 0 | 3 |
| Val | latency_seconds_mean | 5 | 0,440 | 0,055 | 0,400 | 0,500 |
| Val | latency_seconds_median | 5 | 0,460 | 0,055 | 0,400 | 0,500 |
| Teste | sensitivity | 5 | 0,9182 | 0,0593 | 0,8636 | 1,0000 |
| Teste | fall_sensitivity | 5 | 0,8909 | 0,0610 | 0,8182 | 0,9545 |
| Teste | fall_or_fallen_sensitivity | 5 | 0,9182 | 0,0593 | 0,8636 | 1,0000 |
| Teste | false_alarms_per_hour | 5 | 87,55 | 21,44 | 58,37 | 110,89 |
| Teste | n_false_alarms | 5 | 15,0 | 3,7 | 10 | 19 |
| Teste | latency_seconds_mean | 5 | 0,500 | 0,000 | 0,500 | 0,500 |
| Teste | latency_seconds_median | 5 | 0,360 | 0,055 | 0,300 | 0,400 |

### Por classe — `per_class.f1`

`lie_down` e `lying` ficam fora de `RESTRICTED_CLASSES` e têm suporte nulo
ou quase nulo nesta execução; o F1 igual a 0 para essas duas classes é
estrutural (não há amostra suficiente para medir), não uma regressão do
modelo.

| Split | Classe | Suporte | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Val | walk | 708 | 5 | 0,8281 | 0,0173 | 0,7976 | 0,8399 |
| Val | fall | 249 | 5 | 0,8110 | 0,0232 | 0,7756 | 0,8323 |
| Val | fallen | 215 | 5 | 0,8402 | 0,0248 | 0,8118 | 0,8661 |
| Val | sit_down | 124 | 5 | 0,6545 | 0,0395 | 0,6009 | 0,6867 |
| Val | sitting | 270 | 5 | 0,7227 | 0,0273 | 0,6857 | 0,7541 |
| Val | lie_down | 5 | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Val | lying | 0 | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Val | stand_up | 378 | 5 | 0,7493 | 0,0156 | 0,7299 | 0,7708 |
| Val | standing | 96 | 5 | 0,3461 | 0,0306 | 0,3141 | 0,3929 |
| Val | other | 34 | 5 | 0,2354 | 0,0558 | 0,1525 | 0,2892 |
| Teste | walk | 2412 | 5 | 0,8401 | 0,0015 | 0,8384 | 0,8417 |
| Teste | fall | 455 | 5 | 0,6975 | 0,0143 | 0,6842 | 0,7171 |
| Teste | fallen | 442 | 5 | 0,8067 | 0,0133 | 0,7875 | 0,8188 |
| Teste | sit_down | 267 | 5 | 0,6648 | 0,0386 | 0,6113 | 0,7125 |
| Teste | sitting | 716 | 5 | 0,7569 | 0,0464 | 0,7027 | 0,8033 |
| Teste | lie_down | 0 | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Teste | lying | 0 | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Teste | stand_up | 421 | 5 | 0,6392 | 0,0530 | 0,5731 | 0,6957 |
| Teste | standing | 43 | 5 | 0,0302 | 0,0239 | 0,0000 | 0,0592 |
| Teste | other | 860 | 5 | 0,4514 | 0,0425 | 0,3957 | 0,5064 |

Esta tabela não promove nem ranqueia nenhuma seed.

Estas tabelas não promovem nenhuma seed nem recomendam mudança de seed
padrão; 42 permanece o default de `gatefall.train.baseline_a train`. Os
arquivos completos — os 340 registros de `aggregate` achatados no CSV e os
valores brutos por seed (`classification`, `binary_fall_fallen`, `events`
verbatim, incluindo `per_class`/`f1_by_class` para todas as 10 classes) —
estão em `runs/local/le2i/baseline_a_multiseed/multiseed_summary.json` e
`.csv`.

## Resultado da execução real — B0, B1, C0 e C1

Os sumários locais de B0, B1, C0 e C1 registram cinco treinos
independentes por braço, com seeds 42–46 e `n_seeds=5`. A execução real
foi feita pelo usuário fora da CI; os `selftest` da CI usam apenas runs
sintéticos. Como `runs/local/**` não é versionado, os fingerprints, hashes
e agregados abaixo preservam a evidência textual desta execução.

Em cada braço, o checkpoint SHA-256 da seed 42 e suas métricas de
classificação nos splits `train`, `val` e `test` conferem com
`runs/reference/le2i/<arma>/metrics.json` e
`classification_report.json`. A seed 42 permanece a referência canônica
de todos os braços; nenhuma referência versionada foi substituída.

A variação entre seeds descreve inicialização e otimização sob a mesma
receita. Ela não substitui a incerteza amostral do bootstrap agrupado por
sujeito. Os resultados não ranqueiam nem promovem seeds, e o teste não é
usado para ajustar a receita ou selecionar modelos.

Nesta execução de cinco seeds, B1 tem média de `macro_f1_restricted` maior
que B0 na validação e no teste. C1 tem média maior que C0 na validação, mas
menor no teste; a comparação C0/C1 é mista entre splits. Essas observações
descritivas não alteram a referência canônica nem determinam seleção de braço.

### B0

Fingerprint SHA-256 da configuração normalizada: `394d023c9206417ad7027736e602d94477621b11d80aeebbb7b23580a211b0cc`.

| Seed | checkpoint_sha256 |
| --- | --- |
| 42 | `da002ab281c80be6d30175077775c9da7d97b3e009a03167bdb382f3d257d847` |
| 43 | `c62332053276f6d1bc5e71e830c8dfe8f06b469714e56cf64ba51c59be4e91a3` |
| 44 | `531f68d1e78465f7c01d3375e18d0fa27f40bbe7856894616953f36d6475241e` |
| 45 | `699eae132b426435246d2f0311587d56fcfbf3d623beef652580059deadca0f3` |
| 46 | `3f12500592fee463fd21d76e4be06a2396022dc64e77951b7f491ebe61d7d3a9` |

#### Classificação — `macro_f1_restricted`

| Split | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- |
| Treino | 5 | 0,9972 | 0,0002 | 0,9970 | 0,9974 |
| Validação | 5 | 0,6802 | 0,0203 | 0,6606 | 0,7034 |
| Teste | 5 | 0,6558 | 0,0143 | 0,6348 | 0,6751 |

#### Eventos — protocolo de alarme congelado

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Validação | sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_or_fallen_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | false_alarms_per_hour | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Validação | latency_seconds_mean | 5 | 0,3600 | 0,0548 | 0,3000 | 0,4000 |
| Validação | latency_seconds_median | 5 | 0,3000 | 0,0000 | 0,3000 | 0,3000 |
| Teste | sensitivity | 5 | 0,9455 | 0,0498 | 0,8636 | 1,0000 |
| Teste | fall_sensitivity | 5 | 0,9455 | 0,0498 | 0,8636 | 1,0000 |
| Teste | fall_or_fallen_sensitivity | 5 | 0,9455 | 0,0498 | 0,8636 | 1,0000 |
| Teste | false_alarms_per_hour | 5 | 46,6926 | 8,2542 | 40,8560 | 58,3658 |
| Teste | latency_seconds_mean | 5 | 0,3200 | 0,0447 | 0,3000 | 0,4000 |
| Teste | latency_seconds_median | 5 | 0,3000 | 0,0000 | 0,3000 | 0,3000 |

Os valores completos por seed e agregados estão em
`runs/local/le2i/baseline_b0_multiseed/multiseed_summary.json` e `.csv`.

### B1

Fingerprint SHA-256 da configuração normalizada: `b868f5274e5ef258e120ce82c9da44523937072b65151951c1499d9cc3895c1b`.

| Seed | checkpoint_sha256 |
| --- | --- |
| 42 | `74e66a5c7687617124e04b37b72d6effc218587b1caafa6c0cff2ed4fc88fe21` |
| 43 | `0dca0509176ae17071f451f584c7ede34f28b8cb22c34f422cd1799a898d223b` |
| 44 | `1a04365bbabb219cfd65ff2c550f2cde393bb95a99a4e9f8c49e1337234f35f2` |
| 45 | `4230c37c5fab0bbf5669df1abfd90d65c23793291d1a95c2eaf4b22970fc1107` |
| 46 | `9b3b08d3c7c88c87d5d0d860971adde4dc348224aa9ca1b7f46c1705fd01aae3` |

#### Classificação — `macro_f1_restricted`

| Split | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- |
| Treino | 5 | 0,9940 | 0,0007 | 0,9928 | 0,9946 |
| Validação | 5 | 0,6885 | 0,0182 | 0,6645 | 0,7073 |
| Teste | 5 | 0,6722 | 0,0089 | 0,6607 | 0,6844 |

#### Eventos — protocolo de alarme congelado

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Validação | sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_or_fallen_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | false_alarms_per_hour | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Validação | latency_seconds_mean | 5 | 0,3400 | 0,0548 | 0,3000 | 0,4000 |
| Validação | latency_seconds_median | 5 | 0,3200 | 0,0447 | 0,3000 | 0,4000 |
| Teste | sensitivity | 5 | 0,9636 | 0,0203 | 0,9545 | 1,0000 |
| Teste | fall_sensitivity | 5 | 0,9636 | 0,0203 | 0,9545 | 1,0000 |
| Teste | fall_or_fallen_sensitivity | 5 | 0,9636 | 0,0203 | 0,9545 | 1,0000 |
| Teste | false_alarms_per_hour | 5 | 43,1907 | 8,8516 | 35,0195 | 52,5292 |
| Teste | latency_seconds_mean | 5 | 0,3000 | 0,0000 | 0,3000 | 0,3000 |
| Teste | latency_seconds_median | 5 | 0,2800 | 0,0447 | 0,2000 | 0,3000 |

Os valores completos por seed e agregados estão em
`runs/local/le2i/baseline_b1_multiseed/multiseed_summary.json` e `.csv`.

### C0

Fingerprint SHA-256 da configuração normalizada: `7ce58d7ea967633a340396757acfd1728cc4dc2799291e15c9779f8999583623`.

| Seed | checkpoint_sha256 |
| --- | --- |
| 42 | `6cebe3737d850b6d76a5c02579b93210a56f26d8bde7cc0cb9e3148d19d0ed07` |
| 43 | `99ce19cd7093a9af805d32fa958521423a28277613242793e8103ffd785240b4` |
| 44 | `b96bffcbb9fb42a23c0a7b6507d87ebaddf6b8db7775f3c3b044182573cb1b55` |
| 45 | `1750a4061f9c0536b9536d626ac03a457749d9f2510871e917f78c4983eb02fd` |
| 46 | `0460489e3a6d23f5158c0f53f1c44761a5c5e24f53e2d6e7943fa02071712e5e` |

#### Classificação — `macro_f1_restricted`

| Split | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- |
| Treino | 5 | 0,9856 | 0,0017 | 0,9838 | 0,9882 |
| Validação | 5 | 0,6617 | 0,0140 | 0,6448 | 0,6784 |
| Teste | 5 | 0,6391 | 0,0071 | 0,6314 | 0,6461 |

#### Eventos — protocolo de alarme congelado

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Validação | sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_or_fallen_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | false_alarms_per_hour | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Validação | latency_seconds_mean | 5 | 0,3400 | 0,0548 | 0,3000 | 0,4000 |
| Validação | latency_seconds_median | 5 | 0,3000 | 0,0707 | 0,2000 | 0,4000 |
| Teste | sensitivity | 5 | 0,9727 | 0,0407 | 0,9091 | 1,0000 |
| Teste | fall_sensitivity | 5 | 0,9636 | 0,0380 | 0,9091 | 1,0000 |
| Teste | fall_or_fallen_sensitivity | 5 | 0,9727 | 0,0407 | 0,9091 | 1,0000 |
| Teste | false_alarms_per_hour | 5 | 35,0195 | 11,6732 | 23,3463 | 52,5292 |
| Teste | latency_seconds_mean | 5 | 0,3800 | 0,0447 | 0,3000 | 0,4000 |
| Teste | latency_seconds_median | 5 | 0,3000 | 0,0000 | 0,3000 | 0,3000 |

Os valores completos por seed e agregados estão em
`runs/local/le2i/baseline_c0_multiseed/multiseed_summary.json` e `.csv`.

### C1

Fingerprint SHA-256 da configuração normalizada: `4e4320e02731a9803cc38c87f993236956a6921e76e55782ae8d2219f17640d0`.

| Seed | checkpoint_sha256 |
| --- | --- |
| 42 | `e4a6584d8b0135154b0a5ec93956157ab9b1a9b85cc31e4e20d0a2d88a673aca` |
| 43 | `2110f15158ced198be5388273248b165e34f05ec9f45225e6b1dbf9e764107d5` |
| 44 | `b901da9629079656335079d8cd090a415067fe00a0433bf9daed12d7e2bafadb` |
| 45 | `be56122520212226ec1dae97a1cd88ea13237e34e7cae37875a66b0fb0241dca` |
| 46 | `6b08186255ea41f4105876d99c70b2e8a3ddd45f9e9d08da6d43c12c72789050` |

#### Classificação — `macro_f1_restricted`

| Split | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- |
| Treino | 5 | 0,9779 | 0,0027 | 0,9743 | 0,9811 |
| Validação | 5 | 0,6813 | 0,0094 | 0,6710 | 0,6957 |
| Teste | 5 | 0,6254 | 0,0179 | 0,6031 | 0,6448 |

#### Eventos — protocolo de alarme congelado

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Validação | sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | fall_or_fallen_sensitivity | 5 | 1,0000 | 0,0000 | 1,0000 | 1,0000 |
| Validação | false_alarms_per_hour | 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| Validação | latency_seconds_mean | 5 | 0,4000 | 0,0000 | 0,4000 | 0,4000 |
| Validação | latency_seconds_median | 5 | 0,3600 | 0,0548 | 0,3000 | 0,4000 |
| Teste | sensitivity | 5 | 0,9545 | 0,0455 | 0,9091 | 1,0000 |
| Teste | fall_sensitivity | 5 | 0,9545 | 0,0455 | 0,9091 | 1,0000 |
| Teste | fall_or_fallen_sensitivity | 5 | 0,9545 | 0,0455 | 0,9091 | 1,0000 |
| Teste | false_alarms_per_hour | 5 | 39,6887 | 10,4408 | 29,1829 | 52,5292 |
| Teste | latency_seconds_mean | 5 | 0,4200 | 0,0447 | 0,4000 | 0,5000 |
| Teste | latency_seconds_median | 5 | 0,3000 | 0,0000 | 0,3000 | 0,3000 |

Os valores completos por seed e agregados estão em
`runs/local/le2i/baseline_c1_multiseed/multiseed_summary.json` e `.csv`.
