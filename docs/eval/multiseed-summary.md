# Avaliação — Sumário multi-seed

`src/gatefall/eval/multiseed_summary.py` agrega treinos independentes do
braço A que diferem apenas na seed, produzindo estatísticas descritivas
(n/mean/desvio-padrão amostral/min/max) sobre as métricas congeladas de
classificação e de evento. É somente leitura: nenhum artefato dos runs de
entrada (`config.yaml`, `metrics.json`, `checkpoint.pt`,
`alarm_protocol.yaml`, `event_metrics.json`) é modificado.

## Fronteira com o bootstrap agrupado por sujeito

Este módulo é conceitualmente separado de
[Bootstrap agrupado por sujeito](grouped-bootstrap.md) e as duas noções de
variação nunca se misturam na mesma estatística:

- **Sumário multi-seed** agrega **treinos independentes**: cada seed produz
  seu próprio `run_dir` completo, com checkpoint, config e métricas
  próprios. A variação capturada é a variação entre execuções de treino
  independentes sob a mesma receita congelada (`CLAUDE.md`, invariante 1).
- **Bootstrap agrupado por sujeito** reamostra sujeitos a partir de um
  **único checkpoint fixo**, sem retreinar nada. A variação capturada é
  incerteza de amostragem sobre a população de sujeitos avaliados, não
  variação de treino.

Nenhum dos dois seleciona, ranqueia ou promove nenhum run ou seed.

## Contrato de configuração

Todos os `--run-dir` informados devem corresponder a runs locais já
treinados e avaliados do braço A (`config.yaml`/`metrics.json`/
`checkpoint.pt`/`alarm_protocol.yaml`/`event_metrics.json` completos e
íntegros, validados com a mesma checagem de hash usada em
`gatefall.eval.baseline_a_events`). `validate_training_run` foi estendido
com o parâmetro `fields_allowed_to_differ`; este módulo é um dos únicos três
chamadores que passa `{"seed"}` — `qualitative`,
`alarm_protocol_sensitivity` e `grouped_bootstrap` continuam exigindo
igualdade estrita de configuração, e nenhuma checagem de integridade ou
hash foi enfraquecida.

A ferramenta calcula um fingerprint sha256 da configuração de cada run com
o campo `seed` removido e exige que todos os runs informados compartilhem o
mesmo fingerprint — ou seja, que toda a configuração fora da seed seja
idêntica. Também rejeita: menos de duas seeds, `--run-dir` duplicado (mesmo
path resolvido) e seeds duplicadas entre runs distintos. O
`alarm_protocol.yaml` de cada run deve ser igual ao protocolo congelado
`BASELINE_A_ALARM_PROTOCOL`.

## Como executar

```bash
uv run python -m gatefall.eval.multiseed_summary selftest
uv run python -m gatefall.eval.multiseed_summary summarize [--dataset le2i] \
  --run-dir PATH [--run-dir PATH ...] --output-dir PATH [--force]
```

`selftest` roda checagens sintéticas, sem modelo nem GPU. `summarize` exige
pelo menos dois `--run-dir` (a flag é repetível), valida cada run
integralmente e escreve `multiseed_summary.json`/`.csv` em `--output-dir`.
Sem `--force`, se algum dos dois arquivos já existir, o comando é pulado e a
mensagem de skip nomeia exatamente o(s) arquivo(s) encontrado(s).

## Métricas agregadas

- Classificação: `macro_f1_restricted` nos splits `train`, `val`, `test`.
- Evento: `sensitivity`, `fall_sensitivity`, `fall_or_fallen_sensitivity`,
  `false_alarms_per_hour`, `n_false_alarms`, `latency_seconds_mean`,
  `latency_seconds_median` nos splits `val`, `test`.

## Esquema de saída

- `multiseed_summary.json`: `arm`, `config_fingerprint_sha256`, `n_seeds`,
  `seeds` (lista por seed com `seed`, `run_dir`, `checkpoint_sha256` e os
  valores brutos de classificação/evento) e `aggregate` (as mesmas métricas,
  cada uma reduzida a `{n, mean, std, min, max}`).
- `multiseed_summary.csv`: achatado, uma linha por
  `(split, metric_group, metric)`, colunas `split`, `metric_group`,
  `metric`, `n`, `mean`, `std`, `min`, `max`.

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
(`b80b439f1878b1f14c2a879dd6bda6f0ce3dc5572229c0f248375609241b030a`). O run
da seed 42 desta execução reproduziu, checkpoint sha256 idêntico
(`78278b1a6a6eb929c27ae071ebf364a2032877eb0adbcb4e4efd2a6a26787ab8`), o
checkpoint do candidato local já existente — checagem de determinismo
nesta mesma máquina/stack, não uma garantia portável entre
máquinas/GPUs/drivers.

| Split | Métrica | n | Média | Desvio-padrão | Mín | Máx |
| --- | --- | --- | --- | --- | --- | --- |
| Treino | macro_f1_restricted | 5 | 0,8669 | 0,0056 | 0,8588 | 0,8731 |
| Validação | macro_f1_restricted | 5 | 0,6484 | 0,0136 | 0,6336 | 0,6636 |
| Teste | macro_f1_restricted | 5 | 0,6109 | 0,0137 | 0,5919 | 0,6251 |
| Val (evento) | sensitivity | 5 | 0,9846 | 0,0344 | 0,9231 | 1,0000 |
| Val (evento) | false_alarms_per_hour | 5 | 20,77 | 22,57 | 0,00 | 51,92 |
| Teste (evento) | sensitivity | 5 | 0,9182 | 0,0593 | 0,8636 | 1,0000 |
| Teste (evento) | false_alarms_per_hour | 5 | 87,55 | 21,44 | 58,37 | 110,89 |

Esta tabela não promove nenhuma seed nem recomenda mudança de seed padrão;
42 permanece o default de `gatefall.train.baseline_a train`. Os arquivos
completos, com todas as métricas de evento e os valores brutos por seed,
estão em `runs/local/le2i/baseline_a_multiseed/multiseed_summary.json` e
`.csv`.
