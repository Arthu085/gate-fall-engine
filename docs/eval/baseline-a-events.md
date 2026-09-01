# Avaliação — Braço A (protocolo de alarme por evento)

`src/gatefall/eval/` implementa um protocolo de detecção de alarme em nível
de evento sobre o checkpoint já treinado do braço A (ver [Treino — Braço A
(TCN)](../train/baseline-a.md)). Diferente da macro-F1 restrita por janela
usada no treino, aqui a unidade de avaliação é o evento de queda: quantos
eventos reais foram detectados, com que latência, e quantos alarmes
dispararam sem um evento correspondente.

## Como executar

```bash
uv run python -m gatefall.eval.baseline_a_events selftest
```

Roda checagens sintéticas da FSM de gatilho/refratário e da associação
alarme-evento (`events_selftest.py`), sem tocar no checkpoint nem no
dataset real.

```bash
uv run python -m gatefall.eval.baseline_a_events evaluate --dataset le2i \
  --run-dir runs/local/le2i/baseline_a
```

Carrega `checkpoint.pt`, `config.yaml` e `metrics.json` do run local completo,
roda o protocolo sobre `val` e `test` e publica `alarm_protocol.yaml` e
`event_metrics.json` no mesmo diretório. Destinos em `runs/reference/` são
rejeitados.

A avaliação usa lock exclusivo entre processos implementado com
`fcntl.flock`, journal, temporários, hashes do config, checkpoint, métricas de
treino e protocolo e uma promoção conjunta. Por depender de `fcntl`, esse
lifecycle requer um runtime POSIX, como Linux ou WSL. Sem `--force`, preserva
um par de saídas válido; saídas parciais ou inconsistentes falham de forma
explícita. Com `--force`, substitui o par local somente depois de validar os
dois novos arquivos e restaura o par anterior se a promoção falhar.

## Protocolo de alarme

Definido em `alarm_protocol.py` (`AlarmProtocol`, instanciado como
`BASELINE_A_ALARM_PROTOCOL`) e persistido em
`alarm_protocol.yaml` do run avaliado. A cópia histórica versionada fica em
`runs/reference/le2i/baseline_a/alarm_protocol.yaml`.

### Predição positiva e gatilho

Uma janela é positiva se o rótulo predito estiver em `positive_labels`
(`[1, 2]`, isto é, `fall_label` ou `fallen_label`). Um alarme dispara
quando `trigger_consecutive` (3) predições positivas ocorrem em janelas
**consecutivas** (`k_end` contíguo — quebra em qualquer gap na sequência).

A FSM usa comparação estrita (`==`, não `>=`) no ponto em que a contagem
consecutiva atinge `trigger_consecutive`: um run positivo longo colapsa em
exatamente um candidato a alarme (disparado no primeiro `k_end` que atinge
o limiar), não um candidato por quadro depois do limiar.

### Refratário

Após um alarme disparar, qualquer novo gatilho dentro de
`refractory_period_s` (5,0 s) do último alarme é suprimido e não é
retentado depois — o relógio do refratário só reinicia no próximo alarme
que efetivamente soa.

### Evento de queda e janela de associação

Um evento de queda é extraído dos rótulos verdadeiros: cada segmento
contíguo rotulado `fall_label` (1) começa um evento, com
`start_time_s = fall.start_k / target_fps`. A janela de associação
(`[start_time_s, association_end_time_s]`) determina até quando um alarme
ainda conta como detecção desse evento:

- **Caso normal**: se existe um segmento `fallen_label` (2) que começa
  depois do fim do segmento `fall`, `association_end_time_s` é o fim
  desse segmento `fallen` mais `association_end_offset_s` (2,0 s).
- **Fallback** (`fallback_association_uses_fall_end=True`): se nenhum
  segmento `fallen` segue o `fall` (por exemplo, o vídeo termina em
  seguida ao impacto), `association_end_time_s` cai de volta para o fim
  do próprio segmento `fall` mais `association_end_offset_s`. Com
  `fallback_association_uses_fall_end=False`, esse mesmo caso (fall sem
  fallen seguinte) levanta `ValueError` em vez de aplicar o fallback —
  não há degradação silenciosa.

A extração de eventos de queda por split é conferida contra a contagem de
segmentos `fall_label` na anotação bruta do Le2i (antes do descarte de
janelas `IGNORE_LABEL`); uma divergência levanta `ValueError`, sinalizando
que uma janela ignorada partiu um run `fall` real em dois eventos.

A FSM de gatilho e a extração de eventos assumem `k_end` contíguo
(`k_end == k_end_anterior + 1`); por isso `split_event_report` exige
`protocol.eval_stride == 1`; uma violação gera erro explícito mesmo com Python
otimizado, pois o protocolo não suporta stride diferente de 1.

Um alarme é associado ao evento cujo `trigger_time_s` cai dentro dessa
janela; entre múltiplos matches, a associação usa o de menor
`trigger_time_s` para calcular a latência. Um evento sem nenhum alarme
associado é uma perda (`missed`); um alarme sem nenhum evento associado é
um falso alarme.

### Inferência sobre a grade completa de janelas

`evaluate` carrega `val`/`test` com `drop_ignored=False`
com a fonte genérica `PoseWindowDataset`, rodando o modelo
sobre **todas** as janelas do split, incluindo as antes descartadas por
`IGNORE_LABEL`. A avaliação exige explicitamente que `usable_windows ==
total_windows` nesse modo, já que nada é descartado. Isso significa que um
alarme disparado dentro de um trecho sem rótulo confiável (`IGNORE_LABEL`)
ainda conta como falso alarme — não há mais um "buraco" na cobertura
temporal da avaliação. A extração de eventos de queda continua vindo dos
rótulos verdadeiros e não é afetada por essa mudança.

### Denominador de falsos alarmes por hora

`false_alarms_per_hour` usa como denominador **`total_video_time_hours`**
(`total_windows / target_fps / 3600`), isto é, o tempo total de vídeo do
split, incluindo janelas antes ignoradas.

Um segundo campo, `false_alarms_per_hour_labeled_time`, usa como
denominador `labeled_time_hours` (`labeled_windows / target_fps / 3600`),
isto é, apenas o tempo coberto por janelas com rótulo verdadeiro
não-`IGNORE_LABEL` (`labeled_windows` vem de `build_window_index(...,
stride=EVAL_STRIDE, drop_ignored=True)` em `baseline_a_events.py`, contando
no stride 1 do protocolo, não no stride de treino). Seu numerador não é
`n_false_alarms` inteiro, mas apenas os falsos alarmes cuja janela de
gatilho (`trigger_k`) carrega um rótulo verdadeiro não-`IGNORE_LABEL` —
um falso alarme disparado dentro de um trecho sem rótulo confiável é
contado no numerador de `false_alarms_per_hour` (que cobre toda a grade),
mas excluído do numerador de `false_alarms_per_hour_labeled_time`. Por
isso os dois valores podem divergir de forma não trivial quando parte dos
falsos alarmes cai em trechos `IGNORE_LABEL`: no split de teste da
execução real (ver tabela abaixo), 2 dos 10 falsos alarmes disparam em
janelas `IGNORE_LABEL` e são excluídos apenas do numerador da taxa
secundária, o que basta para separar as duas taxas mesmo com denominadores
próximos.

## Schema de `alarm_protocol.yaml`

Serialização direta dos campos de `AlarmProtocol`: `fall_label`,
`fallen_label`, `positive_labels`, `trigger_consecutive`,
`refractory_period_s`, `association_end_offset_s`,
`fallback_association_uses_fall_end`, `eval_stride`, `target_fps`,
`latency_decimal_places` (casas decimais usadas para arredondar latências
individuais e agregadas), `pre_fall_diagnostic_window_s` (1,0 s) e
`pre_fall_alarms_count_as_false_alarms` (`true`) — os dois últimos
documentam explicitamente a regra usada por `n_pre_fall_false_alarms`
abaixo: um alarme disparado até `pre_fall_diagnostic_window_s` segundos
antes do início de um evento `fall` real segue contando como falso alarme
no protocolo (`pre_fall_alarms_count_as_false_alarms=True`), mas é
reportado à parte como diagnóstico, pois pode indicar um gatilho precoce
legítimo em vez de um falso positivo genuíno.

## Schema de `event_metrics.json`

`run_name`, `checkpoint_path` e `alarm_protocol_path` identificam a
execução. `splits.val` e `splits.test` trazem, cada um:

- `usable_windows` / `total_windows`: janelas usadas pelo modelo vs.
  total de janelas do split. Com `drop_ignored=False` na inferência (ver
  acima), os dois valores coincidem — ambos representam a grade completa,
  incluindo janelas `IGNORE_LABEL`.
- `labeled_windows`: quantidade de janelas do split (no stride 1 do
  protocolo) cujo rótulo verdadeiro não é `IGNORE_LABEL`, obtida via
  `build_window_index(..., drop_ignored=True)`. Distinto de
  `usable_windows`/`total_windows`: estes contam a grade completa rodada
  pelo modelo, enquanto `labeled_windows` conta só o subconjunto com
  rótulo confiável — por isso `labeled_windows <= total_windows`.
- `total_video_time_hours` / `labeled_time_hours`: os tempos por trás de
  `false_alarms_per_hour` e `false_alarms_per_hour_labeled_time`,
  respectivamente (ver acima).
- `n_fall_events`, `n_detected_events`, `n_missed_events`, `sensitivity`
  (`n_detected_events / n_fall_events`, nível de evento).
- `n_alarms_total`, `n_false_alarms`, `n_pre_fall_false_alarms`
  (subconjunto diagnóstico de `n_false_alarms` cujo `trigger_time_s` cai
  em `[event.start_time_s - pre_fall_diagnostic_window_s,
  event.start_time_s)` de algum evento `fall` real — ver
  `pre_fall_alarms_count_as_false_alarms` acima).
- `false_alarms_per_hour` (denominador `total_video_time_hours`, numerador
  `n_false_alarms` inteiro; principal) e
  `false_alarms_per_hour_labeled_time` (denominador `labeled_time_hours`,
  numerador restrito aos falsos alarmes cuja janela de gatilho tem rótulo
  verdadeiro não-`IGNORE_LABEL`; secundário — ver acima).
- `window_binary_sensitivity` / `window_binary_specificity`: métricas
  binárias em **nível de janela** (`{fall, fallen}` vs. resto),
  calculadas sobre todas as janelas do split exceto as com rótulo
  `IGNORE_LABEL` (`window_level_binary_metrics`). Distintas do
  `sensitivity` acima, que é em nível de evento — o prefixo
  `window_binary_` marca essa diferença de unidade.
- `latency_seconds`: `per_event` (latência de cada evento detectado, em
  segundos, arredondada a `latency_decimal_places`), mais `mean` e
  `median` (`null` se nenhum evento foi detectado).

## Resultado da execução real

Execução registrada em `runs/reference/le2i/baseline_a/event_metrics.json`, sobre o
checkpoint da última época treinado em [Treino — Braço A
(TCN)](../train/baseline-a.md):

| Split | Eventos | Detectados | Sensibilidade (evento) | Falsos alarmes/h (total) | Falsos alarmes/h (tempo rotulado) | Falsos alarmes pré-queda | Sensibilidade (janela) | Especificidade (janela) | Latência média |
| ----- | ------- | ---------- | ----------------------- | ------------------ | ------------------------- | ------------------------- | ------------------------ | -------------------------- | --------------- |
| Validação | 13 | 12 | 92,3% | 17,3 | 17,3 | 0 | 91,6% | 97,3% | 0,4 s |
| Teste | 22 | 21 | 95,5% | 58,4 | 51,3 | 1 | 90,0% | 97,1% | 0,5 s |

A taxa de falsos alarmes por hora é maior no teste que na validação
(58,4 vs. 17,3 no denominador de tempo total), consistente com a queda de
macro-F1 do treino para o teste já documentada em [Treino — Braço A
(TCN)](../train/baseline-a.md): o split de
teste é cross-subject, então mais confusões entre classes próximas de
`fall`/`fallen` viram alarmes espúrios sobre subjects não vistos. Os
contadores de evento (13/12 na validação, 22/21 no teste) não mudaram
com a inclusão das janelas antes ignoradas na inferência — apenas o
denominador de falsos alarmes por hora e a contagem de falsos alarmes em
si mudaram.

No split de teste, `false_alarms_per_hour_labeled_time` (51,3) fica
sensivelmente abaixo de `false_alarms_per_hour` (58,4): apesar de
`labeled_time_hours` (0,156 h) já ser menor que `total_video_time_hours`
(0,171 h), 2 dos 10 falsos alarmes do split disparam dentro de trechos
`IGNORE_LABEL` e são excluídos do numerador da taxa secundária (ver
"Denominador de falsos alarmes por hora" acima) — por isso a taxa
secundária não sobe proporcionalmente à redução do denominador. Na
validação os dois valores praticamente coincidem (17,3 vs. 17,3), pois
apenas 1 janela do split é `IGNORE_LABEL` e o único falso alarme não cai
nela.
