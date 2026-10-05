# Análise — Diagnóstico qualitativo (render de quadros de alarme)

`src/gatefall/eval/analysis/qualitative.py` é uma ferramenta de diagnóstico
independente de estágio: renderiza, para cada evento de queda detectado,
um PNG do quadro real de vídeo decodificado no instante do gatilho do
alarme, com o esqueleto/bbox do YOLO-Pose sobreposto. Atende os braços A,
B0, B1, C0 e C1. Não faz parte do protocolo de avaliação (ver
[Avaliação — Braço A](../eval/baseline-a-events.md)); existe só para
inspeção visual manual dos alarmes já computados.

## Braços suportados

`--arm {A,B0,B1,C0,C1}` escolhe o braço (A é o padrão). Sem `--run-dir`, o
run é resolvido por `default_run_dir_for_arm` (`runs/local/le2i/baseline_a`,
`baseline_b0`, `baseline_b1`, `baseline_c0` ou `baseline_c1`). A preparação
da inferência é a mesma de `gatefall.eval.analysis.grouped_bootstrap`
(`load_arm_evaluation`), que delega ao `load_event_evaluation()` do avaliador
de eventos de cada braço — A com a seed congelada, como antes; B0/B1/C0/C1
com as mesmas validações de config, estatísticas, features e checkpoint
usadas em [B0](../eval/baseline-b0-events.md),
[B1](../eval/baseline-b1-events.md), [C0](../eval/baseline-c0-events.md) e
[C1](../eval/baseline-c1-events.md). Assim, as predições recomputadas usam
exatamente as entradas e o modelo da avaliação de eventos do braço. Um
`config.yaml` de outro braço é recusado antes da inferência.

Em todos os braços, o protocolo usado é o `BASELINE_A_ALARM_PROTOCOL`
congelado: um `alarm_protocol.yaml` divergente interrompe o comando.

## O que lê e o que nunca toca

`render` lê apenas artefatos já publicados de um run local completo:
`config.yaml`, `alarm_protocol.yaml` e `event_metrics.json` (ver
[Avaliação — Braço A](../eval/baseline-a-events.md)), além das features
HDF5 que o avaliador do braço já lê. É estritamente somente leitura contra
esses artefatos: nunca escreve, sobrescreve ou toca no lock/journal de
`gatefall.eval.baseline_*`, nem em `checkpoint.pt`, `config.yaml`,
`metrics.json`, `alarm_protocol.yaml`, `event_metrics.json` ou nos HDF5 de
pose, DINOv3 e SAM 3. A única escrita do comando é em `{run_dir}/figures/`.

## Por que recomputa predições localmente

`event_metrics.json` guarda métricas agregadas do protocolo de alarme,
não as predições por janela em si — não há como recuperar dali os
quadros concretos de gatilho de cada alarme. Por isso `render` roda uma
passada de inferência local somente leitura (sem lock, sem reescrever
nenhum artefato) sobre a grade completa de janelas do split, e reusa
`gatefall.eval.shared.events` (`fall_events_for_video`,
`detect_alarms_for_video`, `associate_events_and_alarms`) para a
associação evento-alarme, em vez de reimplementar essa lógica. Como
conferência de consistência, `run_render` exige que o `n_detected_events`
recomputado localmente bata com o valor já publicado em
`event_metrics.json` para o mesmo split; uma divergência levanta
`ValueError`.

## Gatilho do alarme versus início da queda

O quadro renderizado é o `trigger_k` do alarme associado ao evento — o
quadro em que a regra de `trigger_consecutive` janelas positivas
consecutivas dispara —, não o início anotado da queda (primeiro quadro
`fall` no ground truth). Quando vários alarmes caem na janela de associação
de um mesmo evento, usa-se o de `trigger_time_s` mais cedo. A diferença
entre os dois instantes é a latência mostrada na legenda
(`latencia=...s`); um PNG não mostra, portanto, o quadro de início da queda.

## Decodificação de vídeo em uma única passada

Os quadros de cada vídeo são decodificados em uma única chamada a
`decode_frames` por vídeo, agrupando todos os `src_index` distintos
necessários para os alarmes daquele vídeo. Não há busca aleatória
(*seek*) por evento: decodificar quadro a quadro sob demanda seria caro
para vídeo bruto, então todos os alvos de um vídeo são resolvidos numa
única leitura sequencial dos índices de origem necessários.

## Pose imputada

Quando a pose no `trigger_k` do alarme tem `person_found=False`, o esqueleto
e a bbox não são desenhados — a legenda troca para "pose imputada" no lugar
do desenho. A origem do desenho seria imputada em qualquer um dos dois
regimes de ausência: zerada antes da primeira detecção do vídeo, ou copiada
por forward-fill da última observação num gap interior (ver [Contrato
temporal — imputação de pose e causalidade do
prefixo](../data/temporal-contract.md#imputacao-de-pose-e-causalidade-do-prefixo)).

## Saída, nome de arquivo e `--force`

PNGs vão para `{run_dir}/figures/`, um arquivo por
par `(video_id, trigger_k)`: `{video_id com "/" trocado por
"__"}__k{trigger_k:06d}.png`, igual para todos os braços. Com
`--source-panel`, o painel de origem usa o mesmo nome com sufixo
`__dinov3_input` ou `__sam3_mask` antes de `.png`; com `--feature-panel`,
o painel de PCA usa o sufixo `__dinov3_pca`. Sem `--force`, um arquivo já existente é
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

## Painel de origem (`--source-panel`)

`--source-panel` é opcional e só se aplica a B0/B1/C0/C1 (com A, o comando
recusa a flag). Para cada alvo, grava um segundo PNG ao lado do principal,
com o mesmo nome acrescido de um sufixo; o PNG principal não muda.

### B0/B1: quadro de entrada do DINOv3

Sufixo `__dinov3_input`. Mostra o mesmo quadro RGB do alarme transformado
para a geometria 224×224 consumida pelo DINOv3 — o redimensionamento
bicúbico com antialias de `gatefall.dinov3.preprocessing.resize_frames`,
sem preservar a proporção — **antes** da normalização por média/desvio (ver
[Features DINOv3 — Pré-processamento](../data/dinov3-features.md#pre-processamento)).
A legenda fica numa faixa abaixo da imagem, sem alterar nenhum pixel de
entrada, e identifica o painel como entrada do DINOv3.

O painel não é um mapa de atenção nem um heatmap: o HDF5 do DINOv3 guarda
só o descritor de 1536 dimensões (CLS concatenado à média dos patches),
sem tokens espaciais de patch, então não há informação espacial a
projetar sobre o quadro.

### C0/C1: máscara SAM 3 selecionada

Sufixo `__sam3_mask`. Mostra o quadro do alarme com a máscara da instância
selecionada pelo SAM 3 sobreposta em magenta, com a bbox da máscara e uma
legenda com `score` e número de instâncias. Máscaras não são persistidas,
e a seleção de instância (`InstanceSelector`) é causal: a escolha no quadro
`k` depende das escolhas em `0..k-1` (ver
[Fundação SAM 3 — Seleção de instância](../data/sam3-foundation.md#selecao-de-instancia)).
Por isso o comando reexecuta o SAM 3 e a seleção sequencialmente, do
primeiro quadro do vídeo até o último alvo daquele vídeo, numa única
decodificação e numa única passada por vídeo para todos os seus alvos.

Antes de gravar qualquer PNG do vídeo, a observação recomputada em cada
quadro alvo é comparada com o HDF5 persistido: `v_t`, `sam_score` e
`n_instances` precisam ser idênticos. Qualquer divergência levanta
`ValueError` e nenhum PNG do vídeo é publicado. O HDF5 do SAM 3 só é lido,
nunca reescrito.

Requisitos de hardware e ambiente: o runtime isolado do SAM 3 precisa
estar instalado (`uv sync --project sam3_runtime --locked`) e o checkpoint
acessível (`--runtime-dir` e `--sam3-checkpoint`, ou as variáveis de
ambiente de [Fundação SAM 3](../data/sam3-foundation.md#como-executar)).
GPU é recomendada: o custo cresce com o prefixo de cada vídeo até o último
alarme, não com o número de alvos. Sem `--source-panel`, C0/C1 não sobem o
runtime do SAM 3.

## Painel de features (`--feature-panel`)

`--feature-panel` é opcional, independente de `--source-panel` e só se
aplica a B0/B1. Grava mais um PNG por alvo, com sufixo `__dinov3_pca`: uma
visualização **pós-hoc e diagnóstica** dos patch tokens do DINOv3 no
quadro do gatilho do alarme.

- O quadro passa pelo mesmo pré-processamento da extração
  (`preprocess_frames`: 224×224 e normalização) e pelo backbone congelado
  (`forward_features`); o painel usa `x_norm_patchtokens` — os tokens
  espaciais de patch, uma grade 14×14 para patch 16 —, não o descritor de
  1536 dimensões persistido no HDF5.
- PCA de 3 componentes sobre a matriz de patch tokens do quadro (patches
  como amostras, canais do embedding como features), após centralizar cada
  canal. O sinal de cada componente é fixado deterministicamente (maior
  carga absoluta positiva), e cada componente é normalizado por min–max
  para `[0,255]` e vira um canal RGB. A grade 14×14 é ampliada para
  224×224 por vizinho mais próximo, preservando os blocos de patch.
- A legenda (`DINOv3 patch-feature PCA (diagnostico pos-hoc)`) fica numa
  faixa abaixo da imagem.

As cores são relativas ao próprio quadro: a PCA é recalculada por quadro,
então cores iguais em dois PNGs não indicam features iguais. O painel não
mostra o que a TCN usou para decidir — B0/B1 consomem só o descritor
persistido — nem é um mapa de atenção.

Antes de carregar o backbone, o comando confere que o commit do
repositório e o hash dos pesos (`--repo-dir`, `--weights`, ou os padrões e
variáveis de ambiente de [Features DINOv3](../data/dinov3-features.md))
batem com a proveniência gravada nos HDF5 do DINOv3. Inferência roda em
modo determinístico; GPU é recomendada. Nenhum HDF5 é lido para escrita.

Para C0/C1, `--feature-panel` é recusado: o runtime oficial isolado do
SAM 3 devolve só máscaras e scores por quadro, sem embedding espacial
denso, então não há base para uma PCA de features do SAM 3. A evidência
visual de C0/C1 continua sendo a máscara selecionada de `--source-panel`.

## Evidência ancorada versus visualização diagnóstica

| Sufixo | Braços | Conteúdo | Natureza |
| --- | --- | --- | --- |
| (nenhum) | A, B0, B1, C0, C1 | Quadro do gatilho com pose/bbox do YOLO-Pose | Evidência ancorada |
| `__dinov3_input` | B0, B1 | Quadro 224×224 de entrada do DINOv3, antes da normalização | Evidência ancorada |
| `__sam3_mask` | C0, C1 | Máscara SAM 3 selecionada, validada contra o HDF5 | Evidência ancorada |
| `__dinov3_pca` | B0, B1 | PCA de 3 componentes dos patch tokens DINOv3 | Visualização diagnóstica pós-hoc |

Os painéis ancorados mostram exatamente o que entra no pipeline (quadro,
pose, entrada do backbone, máscara que gera `V_t`). O painel de PCA é uma
projeção calculada só para inspeção e não é entrada de nenhum modelo.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.qualitative selftest
```

Roda checagens sintéticas (pose imputada não desenha, `decode_frames`
chamado uma vez por vídeo, desenho altera pixels, escolha do alarme mais
cedo, nome de arquivo estável, despacho A/B0/B1/C0/C1, saída de A
preservada, pré-processamento do painel B, replay/validação da máscara
SAM 3 com segmentador falso, despacho de `--feature-panel` só para B0/B1 e
PCA determinística de patch tokens com backbone falso) sem vídeo real, sem GPU e sem tocar em
nenhum artefato do projeto.

```bash
uv run python -m gatefall.eval.analysis.qualitative render --dataset le2i \
  --run-dir runs/local/le2i/baseline_a --split both
```

Recomputa predições sobre `val` e `test` (`--split val`, `--split test`
ou `--split both`, o padrão) e renderiza os PNGs correspondentes aos
eventos detectados. `--force` sobrescreve figuras já existentes.

```bash
uv run python -m gatefall.eval.analysis.qualitative render --arm B1 \
  --run-dir runs/local/le2i/baseline_b1 --source-panel

uv run python -m gatefall.eval.analysis.qualitative render --arm B1 \
  --run-dir runs/local/le2i/baseline_b1 --source-panel --feature-panel \
  --repo-dir DIR --weights PATH

uv run python -m gatefall.eval.analysis.qualitative render --arm C1 \
  --run-dir runs/local/le2i/baseline_c1 --source-panel \
  --sam3-checkpoint PATH
```

## Ausência intencional do pipeline; `render` fora da CI

Este módulo está deliberadamente fora da lista de estágios de
`gatefall.pipeline` (`build_pipeline()`). Dentro do próprio módulo,
`selftest` é totalmente sintético e roda na suíte de selftests do
`.github/workflows/ci.yml` junto com os demais módulos. Só o subcomando
`render` fica fora da CI: ele depende de vídeo bruto decodificado, que a
CI não tem disponível. Essa exclusão de `render` é intencional, não um
esquecimento — ele requer o dataset real e um run já treinado e
avaliado.
