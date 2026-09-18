# Avaliação — Sensibilidade do protocolo de alarme (análise pós-hoc)

`src/gatefall/eval/alarm_protocol_sensitivity.py` é uma ferramenta de
diagnóstico independente de estágio: varre `trigger_consecutive` e
`refractory_period_s` em torno do protocolo de alarme congelado da arma A
(ver [Avaliação — Braço A](baseline-a-events.md)) e recomputa as métricas de
evento para cada combinação, reusando `split_event_report`. Não faz parte
do pipeline padrão nem do lifecycle de
`gatefall.eval.baseline_a_events` (lock/journal); é estritamente somente
leitura contra `checkpoint.pt`/`config.yaml`/`metrics.json` do run e nunca
toca `alarm_protocol.yaml` ou `event_metrics.json`.

## Contrato: nenhuma seleção de protocolo

Esta análise **não seleciona, não ranqueia, não recomenda e não promove**
nenhum protocolo substituto. `BASELINE_A_ALARM_PROTOCOL`
(`trigger_consecutive=3`, `refractory_period_s=5.0`) permanece a única
configuração congelada usada pelo braço A; esta ferramenta apenas descreve
como as métricas de evento se comportariam sob outras combinações. As
métricas do split de teste aqui produzidas são estritamente descritivas —
nenhuma decisão de modelo ou de protocolo pode ser tomada a partir delas.
O JSON de saída registra esse contrato explicitamente em `metadata`
(`selection_performed: false`, `test_split_is_descriptive_only: true`).

## Como executar

```bash
uv run python -m gatefall.eval.alarm_protocol_sensitivity selftest
uv run python -m gatefall.eval.alarm_protocol_sensitivity analyze \
  [--dataset le2i] [--run-dir PATH] \
  [--refractory-grid 0.0,1.0,2.0,5.0,10.0] [--force]
```

`selftest` roda checagens sintéticas, sem modelo nem GPU. `analyze` roda
uma única passada de inferência local por split (val e test) sobre o
checkpoint já treinado e recomputa `split_event_report` para cada célula da
grade — o modelo nunca é reexecutado por combinação. Sem `--force`, se
`alarm_protocol_sensitivity.json`/`.csv` já existirem, o comando é pulado e a
mensagem de skip nomeia exatamente o(s) arquivo(s) encontrado(s) (só o JSON,
só o CSV, ou ambos).

**Execuções concorrentes não são suportadas.** Duas invocações de `analyze`
simultâneas sobre o mesmo `run_dir` passam a checagem de existência antes de
qualquer uma escrever, correm em paralelo e podem terminar com um JSON de
uma execução ao lado de um CSV de outra. Essa ferramenta é deliberadamente
independente do lifecycle de `gatefall.eval.baseline_a_events`: ela nunca
abre nem toca o `EventEvaluationLock` nem o journal canônicos — a mesma
limitação aceita de `gatefall.eval.qualitative render`. O comportamento de
pular sem `--force` é uma conveniência de idempotência (evita reescrever um
resultado já presente), não uma garantia de concorrência; não há lock
próprio e nenhum será adicionado.

## Grade varrida

- `trigger_consecutive`: fixo em `1, 2, 3, 4, 5` (não configurável por CLI).
- `refractory_period_s`: configurável via `--refractory-grid`, uma lista de
  floats separada por vírgula, na ordem em que deve aparecer no relatório
  (a ordem passada nunca é reordenada por valor). Padrão determinístico:
  `0.0, 1.0, 2.0, 5.0, 10.0`.

Todos os demais campos de `AlarmProtocol` (rótulos, offsets de associação,
fallback, fps, casas decimais de latência etc.) permanecem fixos nos
valores de `BASELINE_A_ALARM_PROTOCOL`; cada protocolo da grade é derivado
por `dataclasses.replace(BASELINE_A_ALARM_PROTOCOL, trigger_consecutive=N,
refractory_period_s=R)`, variando só esses dois campos.

A linha com `trigger_consecutive=3` e `refractory_period_s=5.0` é marcada
com `is_frozen_protocol: true` — é exatamente o protocolo congelado em
produção, e sua reprodução exata de `split_event_report` é coberta por
selftest.

## Saída

Dois arquivos em `runs/local/{dataset}/{run_name}/`:

- `alarm_protocol_sensitivity.json`: relatório completo, indentado, sem
  reordenar chaves. Campos de topo: `run_name`, `checkpoint_path`,
  `checkpoint_sha256`, `training_metrics_path`, `training_metrics_sha256`,
  `frozen_protocol` (`BASELINE_A_ALARM_PROTOCOL.to_dict()`), `grid`
  (`trigger_consecutive` e `refractory_period_s` explorados), `metadata`
  (contrato de não seleção) e `rows` — uma entrada por combinação, na ordem
  do laço externo (`trigger_consecutive` ascendente 1..5) e do laço interno
  (`refractory_period_s` exatamente na ordem passada), cada uma com
  `trigger_consecutive`, `refractory_period_s`, `is_frozen_protocol` e
  `splits.val`/`splits.test` — a saída íntegra e não modificada de
  `split_event_report` para aquele protocolo.
- `alarm_protocol_sensitivity.csv`: achatado, uma linha por
  `(trigger_consecutive, refractory_period_s, split)`, na mesma ordem do
  JSON (val antes de test). Colunas: `trigger_consecutive`,
  `refractory_period_s`, `is_frozen_protocol`, `split`, `n_fall_events`,
  `n_detected_events`, `sensitivity`, `n_false_alarms`,
  `false_alarms_per_hour`, `latency_mean_s`, `latency_median_s`,
  `fall_sensitivity`, `fall_or_fallen_sensitivity`. Latência ausente
  (nenhum evento detectado) fica vazia na célula.

## Validação de `--refractory-grid`

Roda imediatamente após o parsing da CLI, antes de qualquer acesso a
modelo/filesystem/`validate_local_run_dir`. Rejeita (`ValueError`): grade
vazia, qualquer valor não finito (`NaN`/`inf`) e qualquer valor negativo.

Uma grade customizada que omite o valor literal `5.0` simplesmente não
produz nenhuma linha marcada `is_frozen_protocol: true` — não é um erro. O
bloco `frozen_protocol` de topo (`BASELINE_A_ALARM_PROTOCOL.to_dict()`) é
sempre registrado no JSON de saída independentemente da grade de
`refractory_period_s` usada.

## Resultado da execução real

Execução sobre `runs/local/le2i/baseline_a` (grade padrão, `trigger_consecutive`
1..5 x `refractory_period_s` 0.0/1.0/2.0/5.0/10.0). Esse run local é o que foi
promovido a referência congelada `runs/reference/le2i/baseline_a/`:
`config.yaml`, `metrics.json` e `alarm_protocol.yaml` são byte a byte
idênticos entre os dois, `event_metrics.json` difere apenas em
`checkpoint_path` e `alarm_protocol_path`, reescritos na promoção, e o
`checkpoint_sha256` registrado é `78278b1a…` em ambos. Os números abaixo
descrevem a referência, não um candidato local à parte. Val e test vêm de uma
única passada de inferência por split; cada célula da tabela reusa
`split_event_report` variando só `trigger_consecutive`/`refractory_period_s`
— o modelo nunca é reexecutado por célula. A coluna `Split=test` é
estritamente descritiva e **não foi usada para nenhuma seleção**, conforme o
contrato de não seleção da seção acima.

A linha ancorada é a congelada (`N=3`, `R=5.0`, marcada `sim` em
"Congelada"), que reproduz exatamente `event_metrics.json` canônico nos dois
splits:

| N | R (s) | Congelada | Split | n_fall | n_detect | Sensib. | FA | FA/h | Lat. média (s) | Lat. mediana (s) | Sens. fall | Sens. fall∪fallen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 5 | sim | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 5 | sim | test | 22 | 22 | 100.0% | 10 | 58.4 | 0.5 | 0.3 | 95.5% | 100.0% |

Grade completa (25 combinações x 2 splits = 50 linhas), na mesma ordem do
CSV/JSON:

| N | R (s) | Congelada | Split | n_fall | n_detect | Sensib. | FA | FA/h | Lat. média (s) | Lat. mediana (s) | Sens. fall | Sens. fall∪fallen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 |  | val | 13 | 13 | 100.0% | 7 | 121.2 | 0.2 | 0.2 | 100.0% | 100.0% |
| 1 | 0 |  | test | 22 | 21 | 95.5% | 75 | 437.7 | 0.3 | 0.2 | 95.5% | 95.5% |
| 1 | 1 |  | val | 13 | 12 | 92.3% | 7 | 121.2 | 0.2 | 0.2 | 92.3% | 92.3% |
| 1 | 1 |  | test | 22 | 20 | 90.9% | 51 | 297.7 | 0.4 | 0.2 | 86.4% | 90.9% |
| 1 | 2 |  | val | 13 | 11 | 84.6% | 7 | 121.2 | 0.2 | 0.2 | 84.6% | 84.6% |
| 1 | 2 |  | test | 22 | 16 | 72.7% | 42 | 245.1 | 0.4 | 0.2 | 68.2% | 72.7% |
| 1 | 5 |  | val | 13 | 10 | 76.9% | 7 | 121.2 | 0.2 | 0.2 | 76.9% | 76.9% |
| 1 | 5 |  | test | 22 | 12 | 54.5% | 35 | 204.3 | 0.4 | 0.4 | 54.5% | 54.5% |
| 1 | 10 |  | val | 13 | 10 | 76.9% | 6 | 103.8 | 0.2 | 0.2 | 76.9% | 76.9% |
| 1 | 10 |  | test | 22 | 12 | 54.5% | 30 | 175.1 | 0.4 | 0.4 | 54.5% | 54.5% |
| 2 | 0 |  | val | 13 | 13 | 100.0% | 2 | 34.6 | 0.3 | 0.3 | 100.0% | 100.0% |
| 2 | 0 |  | test | 22 | 22 | 100.0% | 21 | 122.6 | 0.4 | 0.2 | 100.0% | 100.0% |
| 2 | 1 |  | val | 13 | 13 | 100.0% | 2 | 34.6 | 0.3 | 0.3 | 100.0% | 100.0% |
| 2 | 1 |  | test | 22 | 22 | 100.0% | 18 | 105.1 | 0.4 | 0.2 | 100.0% | 100.0% |
| 2 | 2 |  | val | 13 | 12 | 92.3% | 2 | 34.6 | 0.3 | 0.3 | 92.3% | 92.3% |
| 2 | 2 |  | test | 22 | 22 | 100.0% | 16 | 93.4 | 0.4 | 0.2 | 100.0% | 100.0% |
| 2 | 5 |  | val | 13 | 12 | 92.3% | 2 | 34.6 | 0.3 | 0.3 | 92.3% | 92.3% |
| 2 | 5 |  | test | 22 | 21 | 95.5% | 15 | 87.5 | 0.4 | 0.2 | 95.5% | 95.5% |
| 2 | 10 |  | val | 13 | 12 | 92.3% | 2 | 34.6 | 0.3 | 0.3 | 92.3% | 92.3% |
| 2 | 10 |  | test | 22 | 20 | 90.9% | 11 | 64.2 | 0.4 | 0.2 | 90.9% | 90.9% |
| 3 | 0 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 0 |  | test | 22 | 22 | 100.0% | 13 | 75.9 | 0.5 | 0.3 | 95.5% | 100.0% |
| 3 | 1 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 1 |  | test | 22 | 22 | 100.0% | 12 | 70.0 | 0.5 | 0.3 | 95.5% | 100.0% |
| 3 | 2 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 2 |  | test | 22 | 22 | 100.0% | 11 | 64.2 | 0.5 | 0.3 | 95.5% | 100.0% |
| 3 | 5 | sim | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 5 | sim | test | 22 | 22 | 100.0% | 10 | 58.4 | 0.5 | 0.3 | 95.5% | 100.0% |
| 3 | 10 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.4 | 0.4 | 100.0% | 100.0% |
| 3 | 10 |  | test | 22 | 21 | 95.5% | 7 | 40.9 | 0.5 | 0.3 | 90.9% | 95.5% |
| 4 | 0 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.5 | 0.5 | 100.0% | 100.0% |
| 4 | 0 |  | test | 22 | 21 | 95.5% | 10 | 58.4 | 0.5 | 0.4 | 95.5% | 95.5% |
| 4 | 1 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.5 | 0.5 | 100.0% | 100.0% |
| 4 | 1 |  | test | 22 | 21 | 95.5% | 10 | 58.4 | 0.5 | 0.4 | 95.5% | 95.5% |
| 4 | 2 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.5 | 0.5 | 100.0% | 100.0% |
| 4 | 2 |  | test | 22 | 21 | 95.5% | 9 | 52.5 | 0.5 | 0.4 | 95.5% | 95.5% |
| 4 | 5 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.5 | 0.5 | 100.0% | 100.0% |
| 4 | 5 |  | test | 22 | 21 | 95.5% | 8 | 46.7 | 0.5 | 0.4 | 95.5% | 95.5% |
| 4 | 10 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.5 | 0.5 | 100.0% | 100.0% |
| 4 | 10 |  | test | 22 | 20 | 90.9% | 6 | 35.0 | 0.6 | 0.4 | 90.9% | 90.9% |
| 5 | 0 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.6 | 0.6 | 100.0% | 100.0% |
| 5 | 0 |  | test | 22 | 21 | 95.5% | 8 | 46.7 | 0.6 | 0.5 | 95.5% | 95.5% |
| 5 | 1 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.6 | 0.6 | 100.0% | 100.0% |
| 5 | 1 |  | test | 22 | 21 | 95.5% | 8 | 46.7 | 0.6 | 0.5 | 95.5% | 95.5% |
| 5 | 2 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.6 | 0.6 | 100.0% | 100.0% |
| 5 | 2 |  | test | 22 | 21 | 95.5% | 7 | 40.9 | 0.6 | 0.5 | 95.5% | 95.5% |
| 5 | 5 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.6 | 0.6 | 100.0% | 100.0% |
| 5 | 5 |  | test | 22 | 21 | 95.5% | 6 | 35.0 | 0.6 | 0.5 | 95.5% | 95.5% |
| 5 | 10 |  | val | 13 | 13 | 100.0% | 0 | 0.0 | 0.6 | 0.6 | 100.0% | 100.0% |
| 5 | 10 |  | test | 22 | 20 | 90.9% | 6 | 35.0 | 0.7 | 0.5 | 90.9% | 90.9% |

`n_fall_events` é 13 (val) / 22 (test) em toda célula da grade — invariante
ao protocolo, pois depende só dos rótulos verdadeiros. Tendências
qualitativas observadas na grade, descritas sem ranquear, recomendar ou
indicar preferência entre configurações:

- `trigger_consecutive` baixo combinado com refratário longo perde detecções
  — em `N=1, R=10.0` a sensibilidade cai para 10/13 no val e 12/22 no test —
  porque a janela de refratário suprime o alarme que seria associado a um
  evento posterior.
- `trigger_consecutive` mais alto reduz falsos alarmes e aumenta a latência
  (latência média no val sobe de 0.4s em `N=3` para 0.6s em `N=5`).
- No test, o número de falsos alarmes cai monotonicamente à medida que o
  refratário cresce, dentro de cada valor fixo de `N`.

Nenhuma dessas observações constitui seleção, ranking ou recomendação de
configuração — o contrato de não seleção desta ferramenta permanece
inalterado.
