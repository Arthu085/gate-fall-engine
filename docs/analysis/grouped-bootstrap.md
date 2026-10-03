# Análise — Bootstrap agrupado por sujeito (intervalo de confiança)

`src/gatefall/eval/analysis/grouped_bootstrap.py` é uma ferramenta de diagnóstico
independente de estágio: produz intervalos de confiança percentil por
bootstrap para as métricas de classificação e de evento congeladas de
qualquer braço (A, B0, B1, C0 ou C1; ver
[Avaliação — Braço A](../eval/baseline-a-events.md) e as páginas de avaliação
dos demais braços), reusando uma única passada de inferência local por split.
O subcomando `compare` produz, além disso, o intervalo **pareado** da
diferença adaptativa − baseline para exatamente B1 − B0 e C1 − C0 (ver
[Comparação pareada](#comparacao-pareada-b1-b0-e-c1-c0)).

É uma análise **pós-hoc e somente leitura**: não retreina, não ajusta, não
seleciona modelo, não promove checkpoint, não extrai feature e não altera
protocolo. Não faz parte do pipeline padrão nem do lifecycle dos
avaliadores de eventos (`gatefall.eval.baseline_{a,b0,b1,c0,c1}`, com
lock/journal) e nunca toca `alarm_protocol.yaml`, `event_metrics.json`,
`metrics.json`, `config.yaml` ou `checkpoint.pt`.

## Contrato: só incerteza descritiva, nenhuma seleção

Este bootstrap **não implementa teste de hipótese em nível de janela, nem
p-valor, nem seleção/ranking/promoção de modelo ou protocolo**.
`BASELINE_A_ALARM_PROTOCOL` (`trigger_consecutive=3`,
`refractory_period_s=5.0`) permanece a única configuração congelada de alarme
de todos os braços; esta ferramenta apenas expõe incerteza em torno das
métricas já congeladas, calculadas exatamente como nos avaliadores de
eventos/`split_event_report`. Os intervalos do split de teste aqui produzidos são
estritamente descritivos e **não foram usados para nenhum ajuste ou
seleção**.

Janelas de avaliação em `EVAL_STRIDE=1` se sobrepõem fortemente e não são
observações independentes — reamostrá-las diretamente infla artificialmente
a precisão do intervalo. A unidade de reamostragem aqui é o **sujeito**
(cluster): cada réplica sorteia sujeitos com reposição e arrasta consigo
todos os vídeos e todas as janelas daquele sujeito. O JSON de saída registra
esse contrato explicitamente em `method.note`.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.grouped_bootstrap selftest
uv run python -m gatefall.eval.analysis.grouped_bootstrap analyze \
  [--arm {A,B0,B1,C0,C1}] [--dataset {le2i,le2i-cv}] [--run-dir PATH] \
  [--n-replicates 10000] [--confidence-level 0.95] [--seed 42] [--force]
uv run python -m gatefall.eval.analysis.grouped_bootstrap compare \
  --adaptive-arm {B1,C1} --baseline-arm {B0,C0} [--dataset {le2i,le2i-cv}] \
  [--adaptive-run-dir PATH] [--baseline-run-dir PATH] --output-dir PATH \
  [--n-replicates 10000] [--confidence-level 0.95] [--seed 42] [--force]
```

`selftest` roda checagens sintéticas, sem modelo nem GPU. `analyze` roda uma
única passada de inferência local por split (val e test) sobre o checkpoint
já treinado do braço escolhido (`--arm`, padrão `A`), cacheia as
predições/identidades e bootstrapa esse cache — o modelo nunca é
reexecutado por réplica. Sem `--run-dir`, o run local padrão do braço no
protocolo pedido é usado (`runs/local/le2i/baseline_b1`,
`runs/local/le2i_cv/baseline_c0` etc.). Sem `--force`, se
`grouped_bootstrap.json`/`.csv` já existirem, o comando é pulado e a mensagem
de skip nomeia exatamente o(s) arquivo(s) encontrado(s) (só o JSON, só o
CSV, ou ambos).

`compare` roda a mesma passada única por split em cada um dos dois runs e
grava `paired_bootstrap_<adaptativa>_minus_<baseline>.json`/`.csv` (por
exemplo `paired_bootstrap_b1_minus_b0.json`) em `--output-dir`, obrigatório.
Os run_dirs padrão são os runs locais de cada braço. Sem `--force`, saída
existente é preservada e o comando é pulado antes da inferência.

### Preparação compartilhada por braço

A preparação de cada braço é a do próprio avaliador de eventos
(`load_event_evaluation` em `gatefall.eval.baseline_{a,b0,b1,c0,c1}.cli`):
o mesmo validador de treino contra a configuração congelada do braço e
protocolo, os mesmos loaders de estatísticas, features e qualidade, o mesmo
modelo e o mesmo caminho de inferência. Nenhuma arquitetura nem avaliador
foi reimplementado aqui. Diferenças preservadas:

- **A** continua exigindo a configuração congelada inteira, inclusive a
  seed (comportamento histórico desta análise).
- **B0/B1/C0/C1** aceitam `seed` e `trainable_param_count` divergentes,
  exatamente como os avaliadores de eventos.
- O run deve declarar o braço pedido; run de outro braço, de referência
  (`runs/reference/`) ou do outro protocolo (`runs/local/le2i_cv/` com
  `--dataset le2i` e vice-versa) é recusado antes da inferência.

**Execuções concorrentes não são suportadas**, pela mesma limitação aceita de
`gatefall.eval.analysis.alarm_protocol_sensitivity` e `gatefall.eval.analysis.qualitative
render`: esta ferramenta é deliberadamente independente do lifecycle de
`gatefall.eval.baseline_a`, nunca abre nem toca o
`EventEvaluationLock` nem o journal canônicos. O comportamento de pular sem
`--force` é conveniência de idempotência, não garantia de concorrência.

## Método

### Unidade de reamostragem: sujeito

O mapeamento vídeo→sujeito vem de `adapter.load_frames()` (coluna `subject`
da grade de reamostragem temporal), filtrado pelo split e deduplicado por
`(video_id, subject)`. A construção valida e levanta `ValueError` nomeando
os infratores quando: (a) algum `video_id` mapeia para mais de um `subject`
distinto; (b) algum `video_id` presente nas predições do split está ausente
do mapeamento, ou vice-versa.

Para cada split, `n_unique_clusters = len(subjects)` sujeitos distintos
formam a população de reamostragem. Cada réplica sorteia
`n_unique_clusters` sujeitos com reposição
(`rng.choice(subjects, size=n_unique_clusters, replace=True)`) e inclui,
para cada sujeito sorteado, **todos** os vídeos e **todas** as janelas
daquele sujeito — nunca janelas individuais.

### IDs virtuais por sorteio

Quando um mesmo sujeito é sorteado mais de uma vez na mesma réplica, cada
cópia sorteada precisa permanecer estatisticamente distinta antes de entrar
em `split_event_report` (cujo agrupamento por `video_id` senão fundiria as
cópias e quebraria a contiguidade `k_end == prev_k + 1` exigida por
`extract_label_segments`/`detect_alarms_for_video`). Por isso cada janela
emitida na posição de sorteio `draw_position` recebe
`virtual_video_id = f"{video_id}__draw{draw_position:06d}"`, preservando a
sequência de `k_end` original de cada vídeo dentro de cada cópia.

### Recomputação de exposição

`total_windows`, `usable_windows` (idêntico a `total_windows` por
construção — nenhuma janela é descartada na reamostragem) e
`labeled_windows` (`true_label != IGNORE_LABEL`) são recomputados a partir
das janelas efetivamente sorteadas em cada réplica, não herdados do split
base. Isso garante que `false_alarms_per_hour`/
`false_alarms_per_hour_labeled_time` usem o tempo de vídeo replicado
correto (um sujeito sorteado duas vezes conta o dobro do seu tempo de
exposição).

### Réplicas indefinidas

Métricas cujo denominador pode zerar numa réplica **nunca substituem por
zero** — a réplica é marcada indefinida para aquela métrica e excluída do
cálculo do intervalo:

| Métrica | Indefinida quando |
| --- | --- |
| `sensitivity`, `fall_sensitivity`, `fall_or_fallen_sensitivity` | `n_fall_events == 0` |
| `detected_events_alarm_within_fall_rate`, `latency_mean_s`, `latency_median_s` | `n_detected_events == 0` |
| `false_alarms_per_hour_labeled_time` | `labeled_windows == 0` |
| `precision` | `tp + fp == 0` |
| `recall` | `tp + fn == 0` |
| `specificity` | `tn + fp == 0` |
| `f1` | `2*tp + fp + fn == 0` (o denominador do próprio F1; precision ou recall isoladamente terem denominador zero não torna o F1 indefinido — nesse caso F1=0.0 é um valor válido, não um fallback) |
| `accuracy` | `n_samples == 0` |
| `window_binary_sensitivity` | `tp + fn == 0` (mesma máscara/`positive_labels` da métrica) |
| `window_binary_specificity` | `tn + fp == 0` (mesma máscara/`positive_labels` da métrica) |
| `false_alarms_per_hour` | `total_windows == 0` |
| `macro_f1_restricted` | nunca (F1=0 de classe sem suporte já faz parte da definição da métrica) |

Cada métrica registra `valid_replicates`/`undefined_replicates` (soma sempre
igual a `n_replicates`). Quando `valid_replicates == 0`, `ci_lower`/
`ci_upper` são `null`.

### Determinismo

Seed padrão 42, `n_replicates` padrão 10000, confiança padrão 0.95. Duas
sequências de RNG independentes derivam de
`np.random.SeedSequence(seed).spawn(2)` na ordem fixa (val, depois test),
cada uma alimentando um `Generator` avançado por chamadas repetidas de
`.choice()`. A mesma seed produz saída idêntica bit a bit — coberto por
selftest.

### Ponto de estimativa

O `point_estimate` de cada métrica vem das mesmas funções aplicadas uma
única vez sobre as predições base **não reamostradas**, não da média do
bootstrap — reproduzindo exatamente os valores canônicos de
`metrics.json`/`event_metrics.json`.

## Comparação pareada (B1 − B0 e C1 − C0)

`compare` estima a incerteza amostral da diferença entre o braço adaptativo
e o seu baseline de concatenação, **somente** para B1 − B0 e C1 − C0, nesta
direção. Qualquer outro par (invertido, cruzado entre B e C, ou envolvendo
A) é recusado.

### Método

Para cada split e cada réplica, o sorteio ordenado de sujeitos é gerado
**uma única vez** e aplicado de forma idêntica aos dois runs: mesmos
sujeitos, mesmas multiplicidades e mesmas posições de sorteio, logo os
mesmos IDs virtuais `{video_id}__draw{posição}` em ambos — cópias repetidas
de um sujeito ficam pareadas evento a evento. Cada métrica é calculada
separadamente nas duas réplicas e só então se registra
`delta = adaptativa − baseline`. O IC percentil vem da distribuição desses
deltas por réplica; **não** é obtido subtraindo limites de ICs calculados
independentemente, que ignorariam a correlação entre os dois braços sobre
os mesmos sujeitos.

O ponto de estimativa do delta é a métrica do run adaptativo menos a do
baseline sobre as predições não reamostradas. Uma réplica só é válida para
uma métrica quando a métrica é válida nos dois braços (regras de
indefinição da tabela acima); réplicas indefinidas são contadas em
`undefined_replicates` e nunca convertidas em zero. Quando a métrica é
indefinida no ponto em algum braço, o ponto daquele braço e o delta são
`null`.

A seed padrão (42) deriva os mesmos fluxos de RNG por split da análise de
run único; o sorteio pareado de uma réplica coincide com o sorteio da
mesma réplica em `analyze` de cada braço com a mesma seed.

### Leitura do delta

Delta positivo significa que o braço adaptativo tem valor maior que o
baseline naquela métrica — favorável em sensibilidade, F1 ou
especificidade, desfavorável em `false_alarms_per_hour` e latências. O
intervalo descreve a variação amostral **entre sujeitos** de um único par
de checkpoints treinados com a mesma seed. Não é teste de hipótese, não
produz p-valor nem rótulo de superioridade e não incorpora variação entre
seeds de treino, que é assunto do [Sumário multi-seed](multiseed-summary.md);
as duas noções nunca são combinadas. Com 1 sujeito em val e 2 em test no
Le2i-CS, valem as mesmas ressalvas de resolução inferencial da execução
real abaixo.

### Validação antes do bootstrap

Antes de gravar qualquer arquivo, `compare` falha quando:

- o par não é B1 − B0 ou C1 − C0 na direção adaptativa − baseline;
- algum run é de referência, do outro protocolo, do braço errado, ou os dois
  run_dirs coincidem/se aninham, ou `--output-dir` coincide com/está dentro
  de um run de entrada ou de `runs/reference/`;
- algum run falha no validador de treino do próprio braço (configuração
  congelada, checkpoint e `metrics.json` íntegros);
- as seeds de treino divergem, ou algum campo de configuração presente nos
  dois braços diverge — fora `run_name`, `arm` e `trainable_param_count`.
  Isso cobre receita, `num_classes`, `eval_stride`, estatísticas de
  pose/visual (path e sha256) e, em C1 − C0, as features e a proveniência
  SAM 3;
- algum run não tem `alarm_protocol.yaml` igual a
  `BASELINE_A_ALARM_PROTOCOL` e `event_metrics.json` íntegro e amarrado por
  hash ao checkpoint e ao `metrics.json` (mesma validação do
  [Sumário multi-seed](multiseed-summary.md));
- depois da inferência e antes do bootstrap, os dois runs divergem em
  sujeitos ou mapeamento sujeito→vídeos, na sequência ordenada de
  `(video_id, k_end)`, nos rótulos verdadeiros ou nas contagens de janelas
  de algum split.

## Esquema de saída

### Run único (`analyze`)

Dois arquivos no próprio `--run-dir`:

- `grouped_bootstrap.json`:

```json
{
  "arm": "A", "dataset": "le2i", "run_name": "...", "training_seed": 42,
  "checkpoint_path": "...", "checkpoint_sha256": "...",
  "training_metrics_path": "...", "training_metrics_sha256": "...",
  "alarm_protocol": {
    "fall_label": 1, "fallen_label": 2, "positive_labels": [1, 2],
    "trigger_consecutive": 3, "refractory_period_s": 5.0,
    "association_end_offset_s": 2.0,
    "fallback_association_uses_fall_end": true, "eval_stride": 1,
    "target_fps": "...", "latency_decimal_places": 1,
    "pre_fall_diagnostic_window_s": 1.0,
    "pre_fall_alarms_count_as_false_alarms": true
  },
  "method": {
    "cluster_unit": "subject", "n_replicates": 10000,
    "confidence_level": 0.95, "seed": 42, "ci_method": "percentile",
    "note": "..."
  },
  "splits": {
    "val": {
      "n_unique_clusters": 1,
      "classification": {
        "macro_f1_restricted": {
          "point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
          "valid_replicates": 10000, "undefined_replicates": 0
        },
        "precision": {"...": "..."}, "recall": {"...": "..."},
        "specificity": {"...": "..."}, "f1": {"...": "..."},
        "accuracy": {"...": "..."}
      },
      "events": {
        "sensitivity": {"...": "..."}, "fall_sensitivity": {"...": "..."},
        "fall_or_fallen_sensitivity": {"...": "..."},
        "detected_events_alarm_within_fall_rate": {"...": "..."},
        "false_alarms_per_hour": {"...": "..."},
        "false_alarms_per_hour_labeled_time": {"...": "..."},
        "window_binary_sensitivity": {"...": "..."},
        "window_binary_specificity": {"...": "..."},
        "latency_mean_s": {"...": "..."}, "latency_median_s": {"...": "..."}
      }
    },
    "test": {"...": "mesma forma"}
  }
}
```

  A chave `alarm_protocol` grava `BASELINE_A_ALARM_PROTOCOL.to_dict()`
  (`src/gatefall/eval/shared/alarm_protocol.py`) por inteiro, tornando o artefato
  autodescritivo quanto ao protocolo de alarme congelado usado para calcular
  as métricas de evento — sem essa chave, interpretar
  `false_alarms_per_hour`/latências exigiria abrir `alarm_protocol.yaml`
  separadamente. É somente leitura: nenhum arquivo canônico é modificado
  para produzi-la.

- `grouped_bootstrap.csv`: achatado, uma linha por
  `(split, metric_group, metric)`, colunas `split`, `metric_group`,
  `metric`, `point_estimate`, `ci_lower`, `ci_upper`, `valid_replicates`,
  `undefined_replicates`.

`arm`, `dataset` e `training_seed` foram acrescentados ao JSON; as demais
chaves, o CSV e os valores do braço A permanecem como antes.

### Comparação pareada (`compare`)

Dois arquivos em `--output-dir`, nunca dentro dos runs de entrada:

- `paired_bootstrap_<adaptativa>_minus_<baseline>.json`:

```json
{
  "comparison": "B1 - B0", "dataset": "le2i", "training_seed": 42,
  "adaptive": {
    "arm": "B1", "run_name": "...", "run_dir": "...", "training_seed": 42,
    "config_path": "...", "config_sha256": "...",
    "checkpoint_path": "...", "checkpoint_sha256": "...",
    "training_metrics_path": "...", "training_metrics_sha256": "...",
    "event_metrics_path": "...", "event_metrics_sha256": "...",
    "alarm_protocol_path": "...", "alarm_protocol_sha256": "..."
  },
  "baseline": {"...": "mesma forma, arma B0"},
  "alarm_protocol": {"...": "BASELINE_A_ALARM_PROTOCOL.to_dict()"},
  "method": {
    "cluster_unit": "subject", "n_replicates": 10000,
    "confidence_level": 0.95, "seed": 42, "ci_method": "percentile",
    "note": "...", "paired": true, "delta": "adaptive - baseline"
  },
  "compatibility": {
    "label_names": ["..."],
    "shared_config": {"seed": 42, "num_classes": 10, "eval_stride": 1, "...": "..."},
    "splits": {
      "val": {
        "n_subjects": 1, "subject_to_videos": {"...": ["..."]},
        "n_videos": 0, "total_windows": 0, "labeled_windows": 0,
        "support_sha256": "..."
      },
      "test": {"...": "mesma forma"}
    }
  },
  "splits": {
    "val": {
      "n_unique_clusters": 1,
      "classification": {
        "macro_f1_restricted": {
          "baseline_point_estimate": 0.0, "adaptive_point_estimate": 0.0,
          "delta_point_estimate": 0.0, "delta_ci_lower": 0.0,
          "delta_ci_upper": 0.0, "valid_replicates": 10000,
          "undefined_replicates": 0
        },
        "...": "mesmas métricas do run único"
      },
      "events": {"...": "mesmas métricas do run único"}
    },
    "test": {"...": "mesma forma"}
  }
}
```

  `support_sha256` é o sha256 da sequência ordenada de `video_id`, `k_end` e
  rótulo verdadeiro do split, idêntica nos dois braços por validação.

- `paired_bootstrap_<adaptativa>_minus_<baseline>.csv`: uma linha por
  `(split, metric_group, metric)`, colunas `split`, `metric_group`,
  `metric`, `baseline_point_estimate`, `adaptive_point_estimate`,
  `delta_point_estimate`, `delta_ci_lower`, `delta_ci_upper`,
  `valid_replicates`, `undefined_replicates`.

Os dois arquivos de cada análise são gravados em temporários e promovidos
com `os.replace` adjacentes (melhor esforço de pareamento, não transação);
falha de validação nunca deixa saída parcial.

## Resultado da execução real

Só o braço A foi executado sobre dados reais. As análises de run único de
B0/B1/C0/C1 e as comparações pareadas B1 − B0 e C1 − C0 ainda não foram
executadas; esta página não registra resultados para elas.

Execução sobre `runs/local/le2i/baseline_a` com os parâmetros padrão
(`n_replicates=10000`, `confidence_level=0.95`, `seed=42`). Esse run local é o
que foi promovido a referência congelada `runs/reference/le2i/baseline_a/`:
`config.yaml`, `metrics.json` e `alarm_protocol.yaml` são byte a byte
idênticos entre os dois, `event_metrics.json` difere apenas em
`checkpoint_path` e `alarm_protocol_path`, reescritos na promoção, e o
`checkpoint_sha256` registrado é `78278b1a…` em ambos, de modo que os pontos
de estimativa abaixo são os valores canônicos da referência. Val e test vêm
de uma única passada de inferência por split; `checkpoint.pt`,
`config.yaml`, `metrics.json`, `event_metrics.json` e `alarm_protocol.yaml`
canônicos foram confirmados byte a byte idênticos antes e depois da
execução (hash SHA-256). A coluna de test é estritamente descritiva e
**não foi usada para nenhum ajuste ou seleção**.

**Aviso de resolução inferencial — leia antes de interpretar a tabela abaixo.**
O split val do Le2i real tem apenas **1 sujeito** e o split test real tem
apenas **2 sujeitos**. Isso não é um detalhe secundário: é a limitação mais
importante desta execução.

- **Val (n=1 sujeito):** o bootstrap sempre resorteia o mesmo (e único)
  cluster, então o intervalo colapsa no próprio ponto de estimativa em toda
  métrica — comportamento esperado do método, não um defeito. **Isso não
  significa incerteza zero.** Significa que a variabilidade entre sujeitos é
  matematicamente inestimável a partir de um único cluster; o intervalo
  colapsado é a ausência de informação sobre essa variabilidade, não a
  ausência da própria variabilidade.
- **Test (n=2 sujeitos):** com apenas dois clusters distintos, o bootstrap
  tem resolução inferencial severamente limitada — só existem poucas
  combinações possíveis de reamostragem com reposição de 2 elementos. Os
  intervalos do split test **devem ser lidos como descritivos**, não como um
  intervalo de confiança de 95% bem calibrado em nível populacional. Eles
  ilustram a variação por reamostragem observável já com estes dois sujeitos
  específicos, não uma inferência confiável sobre a população geral de
  sujeitos com quedas.

| Split | Métrica | Ponto | IC 95% | Válidas/Total |
| --- | --- | --- | --- | --- |
| val (n=1 sujeito) | macro_f1_restricted | 0.658 | [0.658, 0.658] | 10000/10000 |
| val | precision | 0.910 | [0.910, 0.910] | 10000/10000 |
| val | recall (sensitivity) | 0.933 | [0.933, 0.933] | 10000/10000 |
| val | specificity | 0.973 | [0.973, 0.973] | 10000/10000 |
| val | f1 | 0.921 | [0.921, 0.921] | 10000/10000 |
| val | accuracy | 0.964 | [0.964, 0.964] | 10000/10000 |
| val | sensitivity (evento) | 1.000 | [1.000, 1.000] | 10000/10000 |
| val | fall_sensitivity | 1.000 | [1.000, 1.000] | 10000/10000 |
| val | fall_or_fallen_sensitivity | 1.000 | [1.000, 1.000] | 10000/10000 |
| val | detected_events_alarm_within_fall_rate | 1.000 | [1.000, 1.000] | 10000/10000 |
| val | false_alarms_per_hour | 0.0 | [0.0, 0.0] | 10000/10000 |
| val | false_alarms_per_hour_labeled_time | 0.0 | [0.0, 0.0] | 10000/10000 |
| val | window_binary_sensitivity | 0.933 | [0.933, 0.933] | 10000/10000 |
| val | window_binary_specificity | 0.973 | [0.973, 0.973] | 10000/10000 |
| val | latency_mean_s | 0.4 | [0.4, 0.4] | 10000/10000 |
| val | latency_median_s | 0.4 | [0.4, 0.4] | 10000/10000 |
| test (n=2 sujeitos) | macro_f1_restricted | 0.623 | [0.576, 0.623] | 10000/10000 |
| test | precision | 0.843 | [0.780, 0.993] | 10000/10000 |
| test | recall (sensitivity) | 0.895 | [0.889, 0.908] | 10000/10000 |
| test | specificity | 0.968 | [0.965, 0.996] | 10000/10000 |
| test | f1 | 0.868 | [0.831, 0.949] | 10000/10000 |
| test | accuracy | 0.957 | [0.955, 0.964] | 10000/10000 |
| test | sensitivity (evento) | 1.000 | [1.000, 1.000] | 10000/10000 |
| test | fall_sensitivity | 0.955 | [0.933, 1.000] | 10000/10000 |
| test | fall_or_fallen_sensitivity | 1.000 | [1.000, 1.000] | 10000/10000 |
| test | detected_events_alarm_within_fall_rate | 0.955 | [0.933, 1.000] | 10000/10000 |
| test | false_alarms_per_hour | 58.4 | [0.0, 67.6] | 10000/10000 |
| test | false_alarms_per_hour_labeled_time | 51.3 | [0.0, 60.3] | 10000/10000 |
| test | window_binary_sensitivity | 0.895 | [0.889, 0.908] | 10000/10000 |
| test | window_binary_specificity | 0.968 | [0.965, 0.996] | 10000/10000 |
| test | latency_mean_s | 0.5 | [0.4, 0.6] | 10000/10000 |
| test | latency_median_s | 0.3 | [0.3, 0.5] | 10000/10000 |

Todas as réplicas foram válidas em ambos os splits (`valid_replicates =
10000` em toda métrica) porque tanto val quanto test têm pelo menos um
evento de queda e pelo menos uma janela rotulada em qualquer combinação de
sujeitos sorteados nesta amostra. Nenhuma dessas observações constitui
seleção, ranking ou recomendação de modelo/protocolo — o contrato desta
ferramenta permanece inalterado.

A regra de validade do F1 binário foi corrigida para considerar indefinida
apenas a réplica cujo próprio denominador (`2*tp + fp + fn`) é zero (ver
tabela de réplicas indefinidas acima). A implementação corrigida foi
reexecutada sobre a execução real local do Arm A após o commit `003d591`,
com `n_replicates=10000`, `confidence_level=0.95` e `seed=42`. Essa
reexecução reproduziu os valores de métrica já documentados, incluindo o
F1 binário, e toda métrica em ambos os splits val e test teve
`10000/10000` réplicas válidas e zero réplicas indefinidas.
