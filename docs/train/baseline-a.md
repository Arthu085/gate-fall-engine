# Treino — Braço A (TCN sobre pose)

`src/gatefall/train/` implementa o treino do braço A: uma TCN causal dilatada
consumindo o vetor de 134 features de pose descrito em [Contrato
temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose), já
padronizado por [`apply_standardization`](../data/pose-standardization.md)
fora de `PoseWindowDataset` — nem `kinematics.py`, nem `pose_dataset.py`, nem
o JSON de estatísticas fazem parte deste módulo.

## Como executar

```bash
uv run python -m gatefall.train.baseline_a selftest
```

Roda checagens sintéticas da arquitetura da TCN (`tcn_selftest.py`), das
métricas restritas (`metrics_selftest.py`) e das guardas de determinismo de
GPU (`engine_selftest.py`), sem treinar nem tocar no dataset real.

```bash
uv run python -m gatefall.train.baseline_a train --dataset le2i \
  --run-dir runs/local/le2i/baseline_a
```

Treina o braço A sobre o Le2i real. O destino padrão depende de `--dataset`:
`runs/local/le2i/baseline_a/` para `le2i`, `runs/local/le2i_cv/baseline_a/`
para `le2i-cv`; `--run-dir` permite explicitar outro diretório local.
Destinos dentro de `runs/reference/` são rejeitados, inclusive quando o
comando é chamado fora da raiz do repositório, assim como um `--run-dir` que
caia sob o diretório local canônico do *outro* protocolo (por exemplo,
`--dataset le2i` com `--run-dir` sob `runs/local/le2i_cv/`). `--seed N`
sobrepõe a seed
padrão (42) e é gravada em `config.yaml`; é o único campo de configuração que
pode variar entre runs comparáveis (ver "Receita de treino congelada" abaixo
e [Sumário multi-seed](../eval/multiseed-summary.md)).

Um run completo exige `config.yaml`, `metrics.json` e `checkpoint.pt` válidos e
coerentes. O treino escreve os três artefatos em um diretório de staging irmão
do destino, registra hashes de configuração e checkpoint nas métricas, valida
o conjunto e só então o publica com `os.replace`. Sem `--force`, um run completo
é preservado e um run parcial ou inconsistente falha informando o artefato
inválido ou ausente. Com `--force`, o run local anterior é movido para um
backup irmão; se a promoção do staging falhar, o backup é restaurado. Depois de
uma promoção bem-sucedida, o backup é removido. Esse lifecycle do treino não
usa lock nem journal e `--force` não torna a referência gravável.

## Arquitetura

TCN causal com convoluções dilatadas, `kernel_size=3`, dilatações
`[1, 2, 4]` e três blocos de canais `[32, 32, 32]`, `dropout=0.3`. O campo
receptivo resultante é 29 quadros, maior que `WINDOW_FRAMES=24` — a rede
enxerga a janela inteira em pelo menos um caminho de convolução. A cabeça é
many-to-one: classifica sobre `NUM_CLASSES=10`, tomando apenas a saída do
último timestep da janela.

## Receita de treino congelada

A receita abaixo é compartilhada, sem alteração, pelos braços B e C
(`CLAUDE.md`, invariante 1 — só o vetor de feature por passo muda entre A,
B e C):

- Seed 42 por padrão; `--seed N` da CLI de treino permite variá-la
  explicitamente. A seed é o único campo experimental que pode divergir
  entre runs — todo o restante da receita abaixo permanece fixo. Ver
  [Avaliação — Sumário multi-seed](../eval/multiseed-summary.md).
- Otimizador AdamW, `lr=1e-3`, `weight_decay=1e-2`.
- Agendamento de learning rate cosseno.
- `batch_size=64`, 30 épocas, sem early stopping.
- `CrossEntropyLoss` ponderada por frequência inversa das classes.
- Clipping de gradiente por norma, limite 1.0.
- Determinismo de GPU: `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG=:4096:8` —
  retreinos com a mesma seed produzem checkpoint idêntico na mesma
  máquina/GPU/driver/cuDNN. `runs/reference/le2i/baseline_a` já foi
  regenerado sob esse regime determinístico e, depois dele, retreinado do
  zero sob o pipeline de features atual (ver "Migração de referência:
  pipeline de features atual" abaixo). Ver [investigação de determinismo de
  GPU](gpu-determinism.md) para o diagnóstico completo.

As 30 épocas são um orçamento fixo pré-registrado, definido antes de rodar
o treino, não um resultado de monitorar `val_macro_f1_restricted` e parar
quando ela parecesse boa. A ausência de early stopping é intencional: fixar
o orçamento de antemão evita escolher a época com base em uma métrica de
validação pequena e enviesada (ver "Seleção de checkpoint" abaixo), o que
inflaria artificialmente o desempenho reportado.

## Seleção de checkpoint: sempre a última época

O checkpoint salvo é sempre o peso da última época, nunca o de melhor
`val_macro_f1_restricted`. O split `val` do Le2i cobre 19 vídeos,
concentrados em apenas três ambientes (`Coffee_room_01`, `Home_01` e
`Home_02`) — então escolher checkpoint por essa métrica otimizaria para o
ruído desse subconjunto pequeno em vez de generalização real.

## Métrica: macro-F1 restrita

Implementada em NumPy puro em `gatefall/train/metrics.py`, sem adicionar
scikit-learn como dependência. A macro-F1 é calculada apenas sobre as
classes `{0, 1, 2, 3, 4, 7, 8, 9}`, excluindo:

- `5` (`lie_down`): suporte de treino 0, suporte de validação 5, suporte de
  teste 0.
- `6` (`lying`): suporte 0 nos três splits (treino, validação e teste).

Incluir essas duas classes no macro-F1 faria o denominador da média ser
dominado por F1 indefinido ou instável sobre poucas ou nenhuma amostra,
distorcendo a métrica agregada sem refletir desempenho real do modelo.

Essa restrição é uma decisão metodológica do GateFall para o cenário
`le2i-cs` do OmniFall, não uma regra oficial de métrica do OmniFall. O
código de experimento oficial do OmniFall chama
`sklearn.metrics.f1_score(references, predictions, average="macro", zero_division=0)`
sem passar `labels=`. Com `labels=None`, o scikit-learn deriva o conjunto de
classes sobre o qual a média é calculada a partir das classes presentes em
`y_true`/`y_pred`, e não de um denominador fixo de 10 classes. O código
oficial do OmniFall também não usa a seleção fixa de classes do GateFall
baseada em suporte positivo de treino.

O conjunto fixo de classes do GateFall (`RESTRICTED_CLASSES`) é escolhido a
partir do suporte de treino e depois aplicado sem alteração a treino,
validação e teste — o suporte de validação ou de teste não pode, por
construção, alterar silenciosamente quais classes entram no macro-F1.

## Diagnóstico de classificação: `report`

```bash
uv run python -m gatefall.train.baseline_a report --dataset le2i \
  --run-dir runs/local/le2i/baseline_a \
  --output runs/local/le2i/baseline_a/classification_report.json --force
```

Gera um diagnóstico multiclasse completo a partir de um run já treinado, sem
alterar o treino, o protocolo de alarme ou qualquer artefato existente. Antes
de rodar inferência, o comando valida `config.yaml`, `metrics.json` e
`checkpoint.pt` com `gatefall.train.artifacts.validate_training_run` (mesma
configuração esperada usada por `train`) e reconstrói os três splits
(`train`, `val`, `test`) exatamente como `train` faz — mesmo stride de treino
(`TRAIN_STRIDE`) e de avaliação (`EVAL_STRIDE`), mesmo carregador de features
de pose.

Por split, o relatório traz:

- `confusion_matrix`: matriz 10x10 (linha = classe verdadeira, coluna =
  classe predita), cobrindo as 10 classes do Le2i, inclusive `lie_down` e
  `lying` (fora do macro-F1 restrito, mas presentes aqui).
- `per_class`: `tp`/`tn`/`fp`/`fn`/`support`/`precision`/`recall`/`f1` para
  cada uma das 10 classes.
- `binary_fall_fallen`: a mesma família de métricas (mais `specificity` e
  `accuracy`) projetando as classes `{fall, fallen}` como positivo contra
  todas as demais como negativo.

O comando é somente leitura em relação aos artefatos de um run: nunca abre
`config.yaml`, `metrics.json`, `checkpoint.pt`, `alarm_protocol.yaml` ou
`event_metrics.json` para escrita, e recusa explicitamente qualquer
`--output` que resolva para um desses cinco nomes dentro de `--run-dir` ou
para dentro de `runs/reference/`, mesmo fora de `--run-dir`.
`--force` só governa a sobrescrita do próprio `--output`; sem ele, um
`--output` já existente é recusado.

O relatório também inclui `verification_against_metrics_json`: o comando
recalcula, a partir da própria inferência, `macro_f1_restricted`, o `f1` de
cada classe restrita e o `support` de cada split, e compara com o que está
gravado em `metrics.json`. O relatório completo é sempre persistido em
`--output`, mesmo quando há divergência, mas o processo sai com código 1 se
qualquer campo recalculado não bater com o valor histórico — um sinal de que
`metrics.json` foi produzido por código diferente do atual.

O relatório também traz, como chaves de topo (irmãs de `splits`):

- `class_support_table`: uma linha por classe (`id` 0..9, `label`,
  `train_support`, `val_support`, `test_support`,
  `included_in_macro_f1`), construída por `class_support_table()` a partir
  do mesmo `support` reconstruído por split usado por
  `verification_against_metrics_json` — nunca de contagens fixas no código.
- `macro_f1_policy`: `restricted_classes`, `excluded_classes`, `classes_with_positive_train_support`
  e `matches_configured_restriction`, construída por `macro_f1_policy_summary()`
  a partir do suporte de treino reconstruído. Ela apenas relata um eventual
  descasamento entre `RESTRICTED_CLASSES` e o suporte de treino observado —
  nunca redefine a restrição usada pelo macro-F1.

As mesmas informações são impressas no stdout como uma tabela de largura
fixa, logo após a linha "relatório de classificação gravado", seguida de
uma linha em português informando as classes restritas/excluídas
configuradas, as classes com suporte de treino positivo e se os dois
conjuntos coincidem. Em caso de descasamento, essa linha o torna explícito
e visível, mas isso é apenas relato: não altera o código de saída nem a
semântica de `verification_against_metrics_json`.

## Artefatos locais e referência histórica

`runs/reference/le2i/baseline_a/config.yaml` e `metrics.json` são evidência
histórica versionada. Desde o retreino sob o pipeline de features atual (ver
"Migração de referência: pipeline de features atual" abaixo), o conteúdo
desses arquivos é a saída real e regenerada do run determinístico corrente,
não apenas conteúdo movido de um caminho legado. O
checkpoint não é versionado. Uma reprodução grava os três artefatos em
`runs/local/le2i/baseline_a/`, ignorado pelo Git.

`config.yaml` grava a configuração completa da execução: identificação do
run e do braço, `seed`, dimensão de entrada, `window_frames`, stride de
treino e de avaliação, número de classes, hiperparâmetros da TCN
(`kernel_size`, `dilations`, `channels`, `dropout`, `receptive_field`),
hiperparâmetros de otimização (`optimizer_name`, `lr`, `weight_decay`,
`grad_clip_norm`, `lr_schedule_name`, `batch_size`, `epochs`), a loss
(`loss_name`, `class_weighted`) e a proveniência das estatísticas de
padronização usadas (`standardization_stats_path` e
`standardization_stats_sha256`, o hash do JSON no momento do treino).

`metrics.json` grava `epochs_trained`, `device`, `torch_version`, o
histórico por época (`train_loss` e `val_macro_f1_restricted`), o bloco
`final` com `macro_f1_restricted`, `f1_by_class`, `support`, `confusion_matrix`
e `per_class` por split (`train`, `val`, `test`), e as listas
`restricted_classes` / `excluded_classes`. `confusion_matrix` e `per_class`
cobrem as 10 classes (não apenas as restritas ao macro-F1) e só existem em
`metrics.json` gerados após a introdução do diagnóstico de classificação —
`validate_training_metrics` continua aceitando `metrics.json` legado sem
esses dois campos.

## Resultado da execução real

Execução registrada em `runs/reference/le2i/baseline_a/metrics.json`, checkpoint na
última época (30):

Esta referência foi retreinada do zero sob o pipeline de features atual — já
com a [seleção de pessoa por
continuidade](../data/temporal-contract.md#selecao-de-pessoa-na-extracao-de-pose)
e a [causalidade do
prefixo](../data/temporal-contract.md#imputacao-de-pose-e-causalidade-do-prefixo)
valendo nos HDF5 canônicos em disco — e promovida nesta entrega (ver
"Migração de referência: pipeline de features atual" abaixo). Não há mais
defasagem entre a referência versionada e o que a metodologia corrente
produz: um retreino com a mesma seed na mesma máquina/GPU/driver/cuDNN
reproduz exatamente estes números.

| Split | Macro-F1 restrita |
| ----- | ------------------ |
| Treino | 0,8674 |
| Validação | 0,6584 |
| Teste | 0,6231 |

A queda de treino para validação/teste é esperada: o split de validação é
pequeno e enviesado (19 vídeos concentrados em três ambientes, ver acima) e
o split le2i-cs é cross-subject — treino e teste cobrem os mesmos seis
ambientes do Le2i, mudam apenas os subjects — então a queda para teste
reflete subjects não vistos no treino, não ambientes novos. O split é por
vídeo/subject justamente para evitar vazamento entre treino e teste
(`CLAUDE.md`, invariante 2).

Generalização a ambientes não vistos é medida separadamente pelo protocolo
`le2i-cv`, disjunto por ambiente e câmera; ver [Generalização entre
ambientes (Le2i-CV)](../eval/le2i-cv-generalization.md) para o run do braço A
sob esse protocolo e a ressalva de confounding entre os dois.

## Migração de referência: pipeline de features atual

A referência do PR #36 (ver "Migração de referência: determinismo de GPU"
abaixo) foi treinada sobre features de pose anteriores a duas mudanças do
pipeline: a seleção de pessoa por continuidade de track (PR #39) e a
construção causal do prefixo (PR #40). Esta entrega retreina a referência do
zero sob a metodologia corrente de `main` e a promove, eliminando essa
defasagem.

Referência anterior (PR #36, preservada no histórico do Git — não há
diretório legado): checkpoint sha256 `264f4997…`; macro-F1 restrita 0,8565
(treino) / 0,6656 (validação) / 0,6201 (teste); 13/13 eventos detectados em
validação e 20/22 em teste; 12 falsos alarmes em teste.

Referência nova (esta migração): checkpoint sha256
`78278b1a6a6eb929c27ae071ebf364a2032877eb0adbcb4e4efd2a6a26787ab8`;
macro-F1 restrita 0,8674 / 0,6584 / 0,6231; 13/13 eventos detectados em
validação e 22/22 em teste; 0 falsos alarmes em validação e 10 em teste.

Contabilidade do que mudou e do que permaneceu idêntico:

- **Mudou:** o checkpoint, `config_sha256` (`f032ed11…` → `3c487416…`),
  `training_metrics_sha256` (`aa3c7e59…` → `7094b56d…`), as macro-F1 dos três
  splits, as F1 por classe, as métricas binárias de janela, as contagens de
  eventos detectados e de falsos alarmes no teste e as latências. Em
  `config.yaml` mudou exatamente uma linha: `standardization_stats_sha256`
  (`15de9ed0…` → `9604b8a6…`).
- **Permaneceu idêntico:** todo o restante da receita congelada em
  `config.yaml`; `alarm_protocol.yaml`, reescrito byte a byte igual
  (`alarm_protocol_sha256` `6952abea…`); o `support` por classe nos três
  splits, pois os dados não mudaram, só o modelo; e, no teste,
  `false_alarms_per_hour_labeled_time` (51,2821 — ver [Avaliação — Braço
  A](../eval/baseline-a-events.md#resultado-da-execucao-real)). Esta última
  igualdade é coincidência, não invariante: os falsos alarmes em tempo
  rotulado continuaram sendo 8, enquanto o total de falsos alarmes no teste
  caiu de 12 para 10.
- **Proveniência dos artefatos finais:** `features.standardize build --force`
  reproduziu `src/gatefall/features/stats/pose_le2i_cs.json` byte a byte,
  confirmando que estatísticas, HDF5 de pose, parquets e janelas já estavam
  correntes — nada a montante do treino precisou ser regenerado. O checkpoint
  recém-treinado tem o mesmo sha256 do candidato local pré-existente e do run
  independente de seed 42 do [sumário
  multi-seed](../eval/multiseed-summary.md), e `metrics.json`,
  `event_metrics.json`, `alarm_protocol.yaml`,
  `alarm_protocol_sensitivity.csv`, `grouped_bootstrap.csv` e
  `multiseed_summary.csv` saíram byte a byte idênticos aos do candidato local.
  Na promoção, `checkpoint_path` e `alarm_protocol_path` dentro de
  `event_metrics.json` foram reescritos de `runs/local/…` para
  `runs/reference/le2i/baseline_a/…`, como em `74437d3`. Nenhum artefato
  `le2i-cv` foi escrito ou regenerado.

`metrics.json` desta referência passa a trazer `confusion_matrix` e
`per_class` nos três splits, ausentes na referência anterior (ver "Artefatos
locais e referência histórica" acima), e `event_metrics.json` passa a trazer
`n_events_detected_in_fall`, `n_events_detected_in_fall_or_fallen`,
`fall_sensitivity`, `fall_or_fallen_sensitivity` e
`detected_events_alarm_within_fall_rate` por split.

A macro-F1 de teste sobe de 0,6201 para 0,6231. Essa diferença é consequência
da correção do pipeline de features, não de uma seleção: o checkpoint continua
sendo o da última época de um orçamento fixo de 30 épocas (ver "Seleção de
checkpoint" acima), nunca uma época escolhida por desempenho, e a macro-F1 de
validação cai no mesmo movimento (0,6656 → 0,6584).

## Migração de referência: determinismo de GPU

Antes da correção de determinismo de GPU (PR #35), a referência mantinha um
run treinado sem as guardas corrigidas de `CUBLAS_WORKSPACE_CONFIG` e
determinismo.

Referência antiga (mantida apenas como registro histórico, não reprodutível
sob as guardas atuais): macro-F1 de teste 0,6212 (0.6211639593563492); 12/13
eventos de queda detectados em validação; 21/22 em teste; 10 falsos alarmes
em teste.

Referência promovida por aquela migração (PR #36), hoje superada: macro-F1 de
teste 0,6201 (0.6201067209256219); 13/13 em validação; 20/22 em teste; 12
falsos alarmes em teste. Checkpoint sha256
`264f4997f0875881f35e20f64a370c955b3488759d5f3714816c6771f12f0ff7`,
reproduzido de forma idêntica em retreinos independentes sob as guardas de
determinismo corrigidas (ver [determinismo de GPU](gpu-determinism.md)).
Nenhum número desta seção descreve a referência vigente: são os do run
anterior às guardas de determinismo corrigidas e os da referência promovida
pelo PR #36, retreinada depois sob o pipeline de features atual — ver
"Migração de referência: pipeline de features atual" acima.

Esta é uma migração metodológica de referência, para reprodutibilidade sob
o regime determinístico congelado e compartilhado com os braços B e C
(`CLAUDE.md`, invariante 1) — não uma seleção do checkpoint de melhor
desempenho. O macro-F1 de teste promovido então (0,6201) não é maior que o
anterior (0,6212), o que é evidência contra cherry-picking.

## Limitações

Nesta execução de seed único, `val_macro_f1_restricted` não é monotônica
ao longo do treino: pelo histórico em `runs/reference/le2i/baseline_a/metrics.json`, ela
atinge um pico de 0,6662 na época 16 e termina em 0,6584 na época 30 (o
checkpoint salvo, ver "Seleção de checkpoint" acima). Como o orçamento de
30 épocas é fixo e pré-registrado (ver "Receita de treino congelada"), o
checkpoint final não é o de melhor macro-F1 de validação observada — a
diferença entre pico e final (~0,0079) é pequena, mas o pico ocorrer quase na
metade do orçamento é um lembrete de que essa métrica de validação é ruidosa
(19 vídeos, três ambientes) e não deve ser lida como uma curva estável.
