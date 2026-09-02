# Avaliação — Diagnóstico qualitativo (render de quadros de alarme)

`src/gatefall/eval/qualitative.py` é uma ferramenta de diagnóstico
independente de estágio: renderiza, para cada evento de queda detectado,
um PNG do quadro real de vídeo decodificado no instante do gatilho do
alarme, com o esqueleto/bbox do YOLO-Pose sobreposto. Não faz parte do
protocolo de avaliação (ver [Avaliação — Braço A](baseline-a-events.md));
existe só para inspeção visual manual dos alarmes já computados.

## O que lê e o que nunca toca

`render` lê apenas artefatos já publicados de um run local completo:
`config.yaml`, `alarm_protocol.yaml` e `event_metrics.json` (ver
[Avaliação — Braço A](baseline-a-events.md)). É estritamente somente
leitura contra esses artefatos: nunca escreve, sobrescreve ou toca no
lock/journal de `gatefall.eval.baseline_a_events`, nem em
`checkpoint.pt`, `config.yaml`, `metrics.json`, `alarm_protocol.yaml` ou
`event_metrics.json`. A única escrita do comando é em
`runs/local/{dataset}/{run_name}/figures/`.

## Por que recomputa predições localmente

`event_metrics.json` guarda métricas agregadas do protocolo de alarme,
não as predições por janela em si — não há como recuperar dali os
quadros concretos de gatilho de cada alarme. Por isso `render` roda uma
passada de inferência local somente leitura (sem lock, sem reescrever
nenhum artefato) sobre a grade completa de janelas do split, e reusa
`gatefall.eval.events` (`fall_events_for_video`,
`detect_alarms_for_video`, `associate_events_and_alarms`) para a
associação evento-alarme, em vez de reimplementar essa lógica. Como
conferência de consistência, `run_render` exige que o `n_detected_events`
recomputado localmente bata com o valor já publicado em
`event_metrics.json` para o mesmo split; uma divergência levanta
`ValueError`.

## Decodificação de vídeo em uma única passada

Os quadros de cada vídeo são decodificados em uma única chamada a
`decode_frames` por vídeo, agrupando todos os `src_index` distintos
necessários para os alarmes daquele vídeo. Não há busca aleatória
(*seek*) por evento: decodificar quadro a quadro sob demanda seria caro
para vídeo bruto, então todos os alvos de um vídeo são resolvidos numa
única leitura sequencial dos índices de origem necessários.

## Pose imputada

Quando a pose no `trigger_k` do alarme tem `person_found=False` (pose
zero-preenchida por imputação, ver [Contrato temporal — dataset de
janelas de pose](../data/temporal-contract.md#dataset-de-janelas-de-pose)),
o esqueleto e a bbox não são desenhados sobre a origem zerada — a
legenda troca para "pose imputada" no lugar do desenho.

## Saída, nome de arquivo e `--force`

PNGs vão para `runs/local/{dataset}/{run_name}/figures/`, um arquivo por
par `(video_id, trigger_k)`: `{video_id com "/" trocado por
"__"}__k{trigger_k:06d}.png`. Sem `--force`, um arquivo já existente é
preservado e contado como pulado; com `--force`, é sobrescrito
atomicamente (escrita em `.tmp` seguida de `os.replace`). O desenho do
esqueleto/bbox e a codificação do PNG usam Pillow (`PIL.Image`,
`PIL.ImageDraw`, `PIL.ImageFont`); a decodificação de vídeo continua
exclusivamente via `gatefall.data.video_io.decode_frames`, nunca por
alguma API de vídeo do Pillow.

Com `--include-false-alarms`, o comando escreve também um PNG por
gatilho de alarme falso, além dos PNGs de evento detectado já descritos
acima. Esses arquivos usam o prefixo `falsealarm__` no nome (em vez do
padrão acima) e a legenda mostra `(ALARME FALSO)` no lugar da latência,
já que não há evento associado a um alarme falso. A flag é aditiva e
desligada por padrão: sem ela, os arquivos e a contagem de PNGs de
evento detectado são exatamente os mesmos de antes.

## Como executar

```bash
uv run python -m gatefall.eval.qualitative selftest
```

Roda checagens sintéticas (pose imputada não desenha, `decode_frames`
chamado uma vez por vídeo, desenho altera pixels, escolha do alarme mais
cedo, nome de arquivo estável) sem vídeo real, sem GPU e sem tocar em
nenhum artefato do projeto.

```bash
uv run python -m gatefall.eval.qualitative render --dataset le2i \
  --run-dir runs/local/le2i/baseline_a --split both
```

Recomputa predições sobre `val` e `test` (`--split val`, `--split test`
ou `--split both`, o padrão) e renderiza os PNGs correspondentes aos
eventos detectados. `--force` sobrescreve figuras já existentes.

## Ausência intencional do pipeline; `render` fora da CI

Este módulo está deliberadamente fora da lista de estágios de
`gatefall.pipeline` (`build_pipeline()`). Dentro do próprio módulo,
`selftest` é totalmente sintético e roda na suíte de selftests do
`.github/workflows/ci.yml` junto com os demais módulos. Só o subcomando
`render` fica fora da CI: ele depende de vídeo bruto decodificado, que a
CI não tem disponível. Essa exclusão de `render` é intencional, não um
esquecimento — ele requer o dataset real e um run já treinado e
avaliado.
