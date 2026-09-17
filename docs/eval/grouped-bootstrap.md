# Avaliação — Bootstrap agrupado por sujeito (intervalo de confiança)

`src/gatefall/eval/grouped_bootstrap.py` é uma ferramenta de diagnóstico
independente de estágio: produz intervalos de confiança percentil por
bootstrap para as métricas de classificação e de evento congeladas do braço A
(ver [Avaliação — Braço A](baseline-a-events.md)), reusando uma única
passada de inferência local por split. Não faz parte do pipeline padrão nem
do lifecycle de `gatefall.eval.baseline_a_events` (lock/journal); é
estritamente somente leitura contra `checkpoint.pt`/`config.yaml`/
`metrics.json` do run e nunca toca `alarm_protocol.yaml`,
`event_metrics.json`, `metrics.json` ou `checkpoint.pt`.

## Contrato: só incerteza descritiva, nenhuma seleção

Este bootstrap **não implementa teste de hipótese em nível de janela, nem
p-valor, nem seleção/ranking/promoção de modelo ou protocolo**.
`BASELINE_A_ALARM_PROTOCOL` (`trigger_consecutive=3`,
`refractory_period_s=5.0`) permanece a única configuração congelada do braço
A; esta ferramenta apenas expõe incerteza em torno das métricas já
congeladas, calculadas exatamente como em `baseline_a.py`/
`split_event_report`. Os intervalos do split de teste aqui produzidos são
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
uv run python -m gatefall.eval.grouped_bootstrap selftest
uv run python -m gatefall.eval.grouped_bootstrap analyze \
  [--dataset le2i] [--run-dir PATH] \
  [--n-replicates 10000] [--confidence-level 0.95] [--seed 42] [--force]
```

`selftest` roda checagens sintéticas, sem modelo nem GPU. `analyze` roda uma
única passada de inferência local por split (val e test) sobre o checkpoint
já treinado, cacheia as predições/identidades e bootstrapa esse cache — o
modelo nunca é reexecutado por réplica. Sem `--force`, se
`grouped_bootstrap.json`/`.csv` já existirem, o comando é pulado e a mensagem
de skip nomeia exatamente o(s) arquivo(s) encontrado(s) (só o JSON, só o
CSV, ou ambos).

**Execuções concorrentes não são suportadas**, pela mesma limitação aceita de
`gatefall.eval.alarm_protocol_sensitivity` e `gatefall.eval.qualitative
render`: esta ferramenta é deliberadamente independente do lifecycle de
`gatefall.eval.baseline_a_events`, nunca abre nem toca o
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
| `f1` | `tp + fp == 0` ou `tp + fn == 0` |
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

## Esquema de saída

Dois arquivos em `runs/local/{dataset}/{run_name}/`:

- `grouped_bootstrap.json`:

```json
{
  "run_name": "...", "checkpoint_path": "...", "checkpoint_sha256": "...",
  "training_metrics_path": "...", "training_metrics_sha256": "...",
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

- `grouped_bootstrap.csv`: achatado, uma linha por
  `(split, metric_group, metric)`, colunas `split`, `metric_group`,
  `metric`, `point_estimate`, `ci_lower`, `ci_upper`, `valid_replicates`,
  `undefined_replicates`.

## Resultado da execução real

Execução sobre `runs/local/le2i/baseline_a` com os parâmetros padrão
(`n_replicates=10000`, `confidence_level=0.95`, `seed=42`). Val e test vêm
de uma única passada de inferência por split; `checkpoint.pt`,
`config.yaml`, `metrics.json`, `event_metrics.json` e `alarm_protocol.yaml`
canônicos foram confirmados byte a byte idênticos antes e depois da
execução (hash SHA-256). A coluna de test é estritamente descritiva e
**não foi usada para nenhum ajuste ou seleção**.

O split val do Le2i tem `n_unique_clusters = 1` (um único sujeito), então o
bootstrap sempre resorteia o mesmo (e único) cluster: o intervalo colapsa no
próprio ponto de estimativa em toda métrica — comportamento esperado do
método, não um defeito.

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
