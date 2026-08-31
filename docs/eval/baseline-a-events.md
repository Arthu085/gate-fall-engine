# Avaliação — Arma A (protocolo de alarme por evento)

`src/gatefall/eval/` implementa um protocolo de detecção de alarme em nível
de evento sobre o checkpoint já treinado da arma A (ver [Treino — Arma A
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
uv run python -m gatefall.eval.baseline_a_events evaluate --force
```

Carrega `runs/baseline_a/checkpoint.pt` e `runs/baseline_a/config.yaml`,
roda o protocolo de alarme sobre os splits `val` e `test` do Le2i e grava
`runs/baseline_a/alarm_protocol.yaml` e `runs/baseline_a/event_metrics.json`.
Assim como no treino, `runs/baseline_a/event_metrics.json` já está
versionado neste repositório: uma execução sem `--force` apenas imprime a
mensagem de "já existe" e retorna sem erro. `alarm_protocol.yaml` é sempre
regravado, pois descreve a configuração fixa do protocolo, não um resultado
de execução.

## Protocolo de alarme

Definido em `alarm_protocol.py` (`AlarmProtocol`, instanciado como
`BASELINE_A_ALARM_PROTOCOL`) e persistido em
`runs/baseline_a/alarm_protocol.yaml`.

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
  do próprio segmento `fall` mais `association_end_offset_s`.

Um alarme é associado ao evento cujo `trigger_time_s` cai dentro dessa
janela; entre múltiplos matches, a associação usa o de menor
`trigger_time_s` para calcular a latência. Um evento sem nenhum alarme
associado é uma perda (`missed`); um alarme sem nenhum evento associado é
um falso alarme.

### Denominador de falsos alarmes por hora

`false_alarms_per_hour` usa como denominador `evaluated_time_hours`
(`usable_windows / target_fps / 3600`), isto é, o tempo efetivamente
avaliado — apenas as janelas utilizáveis pelo `PoseWindowDataset`, não o
tempo total de vídeo do split (`total_video_time_hours`, calculado sobre
`total_windows` incluindo janelas ignoradas). Os dois valores aparecem em
`event_metrics.json` para permitir a comparação.

## Schema de `alarm_protocol.yaml`

Serialização direta dos campos de `AlarmProtocol`: `fall_label`,
`fallen_label`, `positive_labels`, `trigger_consecutive`,
`refractory_period_s`, `association_end_offset_s`,
`fallback_association_uses_fall_end`, `eval_stride`, `target_fps` e
`latency_decimal_places` (casas decimais usadas para arredondar latências
individuais e agregadas).

## Schema de `event_metrics.json`

`run_name`, `checkpoint_path` e `alarm_protocol_path` identificam a
execução. `splits.val` e `splits.test` trazem, cada um:

- `usable_windows` / `total_windows`: janelas usadas pelo modelo vs.
  total de janelas do split (incluindo ignoradas).
- `evaluated_time_hours` / `total_video_time_hours`: os dois tempos por
  trás do denominador de `false_alarms_per_hour` (ver acima).
- `n_fall_events`, `n_detected_events`, `n_missed_events`, `sensitivity`
  (`n_detected_events / n_fall_events`).
- `n_alarms_total`, `n_false_alarms`, `false_alarms_per_hour`.
- `latency_seconds`: `per_event` (latência de cada evento detectado, em
  segundos, arredondada a `latency_decimal_places`), mais `mean` e
  `median` (`null` se nenhum evento foi detectado).

## Resultado da execução real

Execução registrada em `runs/baseline_a/event_metrics.json`, sobre o
checkpoint da última época treinado em [Treino — Arma A
(TCN)](../train/baseline-a.md):

| Split | Eventos | Detectados | Sensibilidade | Falsos alarmes/h | Latência média |
| ----- | ------- | ---------- | -------------- | ------------------ | --------------- |
| Validação | 13 | 12 | 92,3% | 17,3 | 0,4 s |
| Teste | 22 | 21 | 95,5% | 57,7 | 0,5 s |

A taxa de falsos alarmes por hora é maior no teste que na validação
(57,7 vs. 17,3), consistente com a queda de macro-F1 do treino para o
teste já documentada em [Treino — Arma A
(TCN)](../train/baseline-a.md#resultado-da-execução-real): o split de
teste é cross-subject, então mais confusões entre classes próximas de
`fall`/`fallen` viram alarmes espúrios sobre subjects não vistos.
