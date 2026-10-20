# Eventos em `val` e decisão do piloto GMDCSA24 (T e S, braço A)

`gatefall.eval.gmdcsa24_weak_label` fecha a parte do piloto que antecede
`test`: a avaliação de eventos só em `val` dos runs de T e S, necessária para
G2 e G3, e a decisão GO/NO-GO pelo
[critério de decisão](../analysis/gmdcsa24-weak-label-protocol.md#criterio-de-decisao).
O módulo não muda nenhuma decisão do protocolo: seeds, receita, controle
pareado, `BASELINE_A_ALARM_PROTOCOL`, condições de validade e limites de
G1–G4 continuam os de lá.

**Estado:** implementação apenas. Neste PR não rodou nenhum treino real de T
ou S, nenhuma avaliação de eventos dos runs do piloto e nenhuma decisão. O
responsável pelo projeto informou que o `prepare` real de T e S terminou, com
os manifestos de treino e os JSON de padronização registrados por SHA-256, e
que os dez `preflight` (T e S nas seeds 42 a 46) passaram nas 11 condições.
Essa evidência não foi reexecutada aqui e entra como pré-treino validado
informado pelo responsável.

## Avaliação só em `val`

O avaliador canônico do braço A processa `val` e `test` juntos, padroniza com
o JSON do controle e publica `event_metrics.json`, que o validador
compartilhado só aceita com os dois splits. O caminho do piloto reaproveita
dele o carregamento do checkpoint, a predição por janela, a associação de
eventos, o lock, o journal e a publicação conjunta, e muda só o necessário:

- a `val` do Le2i-CS é montada como no avaliador canônico (`EVAL_STRIDE`, sem
  descartar `IGNORE_LABEL`), com as features do braço A, e padronizada pelo
  JSON do tratamento já conferido;
- a grade entregue à orquestração tem só as linhas de `val`, e o avaliador
  recusa qualquer outro split. Nenhuma feature, predição, métrica ou artefato
  de `test` é montado, lido ou gravado;
- o relatório vai para `val_event_metrics.json`, com só `val`, ao lado de
  `alarm_protocol.yaml` no run canônico. Como não se chama
  `event_metrics.json` nem tem `test`, nenhum avaliador, sumário ou bootstrap
  canônico o consome.

A orquestração compartilhada ganhou parâmetros para isso (splits, nome do
arquivo, campos de proveniência e uma conferência antes da publicação), com
os valores canônicos como padrão. As armas A, B0, B1, C0 e C1 continuam
publicando `val` e `test` em `event_metrics.json`, com os mesmos campos.

### O que `evaluate` confere antes de avaliar

| Condição | O que é conferido |
| --- | --- |
| Portões de pré-treino | As 11 condições do [`preflight`](../train/gmdcsa24-weak-label.md#portoes-de-pre-treino), inclusive a execução da raiz do repositório |
| run | Run canônico completo: run do braço A válido só com `train` e `val` e `treatment.json` igual ao recalculado (tratamento, seed, manifesto preparado, padronização, pesos e controle pareado); nenhum `event_metrics.json` no run |
| eventos do controle | `event_metrics.json` canônico do controle pareado válido pelo validador do [sumário multi-seed](../analysis/multiseed-summary.md) (protocolo congelado e hashes de checkpoint, métricas e protocolo) |

Depois da predição e antes da publicação, os campos que dependem só da grade
de `val` (`usable_windows`, `total_windows`, `labeled_windows`,
`total_video_time_hours`, `labeled_time_hours` e `n_fall_events`) precisam
ser iguais aos da `val` do controle. Do relatório canônico do controle só
`splits.val` é usado; o `test` dele é o já publicado do controle e não entra
na decisão.

### Relatório

`val_event_metrics.json` tem os campos do relatório canônico, com `splits`
só com `val`, e mais:

- `evaluated_splits`: `["val"]`;
- `treatment`: tratamento, seed, SHA-256 do `treatment.json`, do manifesto
  preparado e do checkpoint do controle.

Um relatório existente só é aceito se o protocolo for o congelado, os hashes
de checkpoint, `metrics.json` e protocolo baterem com o run e esses campos
forem os recalculados. Sem `--force`, um par válido é pulado, e um par parcial
ou inconsistente é recusado; `--force` só reconstrói esse par, nunca o run.

## Decisão GO/NO-GO

`decide` usa só `val` e lê, para T, S e o controle de cada seed:
`macro_f1_restricted` e F1 de `fall` de `final.val` do `metrics.json`, e
`n_missed_events` e `n_false_alarms` da `val` dos relatórios de eventos. A
comparação é pareada por seed, Δ = tratamento − controle, nas seeds da tabela
de `checkpoint_sha256` do controle (42 a 46).

| Condição | Regra (cinco seeds) |
| --- | --- |
| G1 | Média de Δ `macro_f1_restricted` >= +0,015 e Δ > 0 em pelo menos 4 seeds |
| G2 | Soma de `n_missed_events` do tratamento <= soma do controle |
| G3 | Soma de `n_false_alarms` do tratamento <= soma do controle |
| G4 | Média de Δ F1 de `fall` >= 0 |

- Qualquer falha de W1, W2 ou V2–V5 em T ou S, em qualquer seed, dá
  **INVÁLIDO**, sem calcular G1–G4.
- Com a validade cumprida, T decide: G1–G4 aprovadas dão **GO exploratório**;
  qualquer uma reprovada dá **NO-GO**.
- S tem G1–G4 calculadas e relatadas. Com GO exploratório de T, média de Δ
  `macro_f1_restricted` de S <= 0 dá **GO frágil**. S nunca transforma NO-GO
  em GO.

Os limites são comparados sem arredondamento, e a média usa soma exata
(`math.fsum`), então a decisão não depende da ordem das seeds.

### Recusa sem decisão

`decide` não registra nada, e sai com 1, quando falta o que a decisão exige:

- qualquer seed do controle sem run canônico, sem `treatment.json` ou sem
  `val_event_metrics.json`, em T ou em S, ou um diretório de run fora dessas
  seeds;
- `treatment.json` de T e S com controles diferentes na mesma seed;
- falha de checkout, proveniência, tratamento, padronização e pesos ou saídas
  (portões de implementação, não de validade);
- com a validade cumprida, run incompleto ou com `treatment.json` divergente,
  `event_metrics.json` no run do tratamento, relatório canônico do controle
  inválido ou relatório de `val` inválido, com `test`, com outra proveniência,
  outro protocolo de alarme ou outra estrutura de `val`.

### Registro

A decisão é gravada uma vez em `runs/local/gmdcsa24_weak_label/decision.json`:
critério, seeds, `evaluated_splits = ["val"]`, decisão, falhas de validade ou
G1–G4 de T e S com os Δ por seed, as medidas de cada seed e o SHA-256 de cada
arquivo lido (`treatment.json`, `metrics.json` e `val_event_metrics.json` do
run; `metrics.json` e `event_metrics.json` do controle), mais o commit do
código e a data. Um `decide` posterior que recalcula o mesmo conteúdo (fora
commit e data) só confirma; um conteúdo diferente é recusado e nunca
substitui o registro, pelos
[critérios de parada 6 e 7](../analysis/gmdcsa24-weak-label-protocol.md#criterios-de-parada).
Repetir depois de um INVÁLIDO exige corrigir a causa e arquivar o registro
manualmente.

O registro é a decisão. As métricas descritivas de `test`, calculadas uma vez
depois dela, ficam para uma implementação própria, que não existe neste
módulo; nenhum comando aqui avalia `test`.

## Saídas

```text
runs/local/gmdcsa24_weak_label/
├── decision.json
└── {t,s}/seed_<n>/
    ├── config.yaml, metrics.json, checkpoint.pt, treatment.json   # do treino
    ├── alarm_protocol.yaml
    └── val_event_metrics.json                                     # só val
```

Tudo é ignorado pelo Git e não é sincronizado com o Google Drive. Os runs do
controle e os artefatos do Le2i são só lidos.

## Como executar

Na raiz de um checkout limpo, depois dos dez treinos, para cada tratamento e
seed:

```bash
uv run python -m gatefall.eval.gmdcsa24_weak_label evaluate --treatment T --seed 42 \
  --control-run-dir runs/local/le2i/baseline_a_multiseed/seed_42
```

e, com os dez relatórios de `val`:

```bash
uv run python -m gatefall.eval.gmdcsa24_weak_label decide
```

`decide` lê o controle de cada seed do `treatment.json` do run. Como
`preflight` e `train`, os dois comandos exigem executar da raiz do checkout
cujo código está instalado no ambiente usado.

## Selftest

`selftest` (na CI) confere as regras de decisão com cinco seeds sintéticas e,
no mundo sintético do [selftest do treino](../train/gmdcsa24-weak-label.md#selftest),
prepara e treina T e S na seed 42, grava o `event_metrics.json` canônico do
controle e cobre:

- todos os limites de G1–G4 (média exatamente 0,015 e logo abaixo, 4 e 3
  seeds positivas, Δ = 0 não contando, uma perda ou um alarme a mais, Δ de
  F1 de `fall` logo abaixo de 0), GO frágil com S igual a 0 e negativo, S
  sem efeito sobre NO-GO, INVÁLIDO por qualquer condição de validade, seed
  ausente, repetida ou fora das do controle, controle diferente entre T e S e
  determinismo;
- leitura de features que reprova qualquer vídeo de `test` e predição
  espionada: só o vídeo de `val` é lido e predito, a grade só tem `val` e o
  split `test` é recusado;
- relatório com `test`, só com `test`, com `evaluated_splits` a mais, com
  seed, tratamento, manifesto, controle ou checkpoint trocados, com contagens
  inconsistentes ou com outro protocolo de alarme, e `treatment.json`
  adulterado, recusados;
- `val` do controle com outra contagem de quedas e `event_metrics.json`
  plantado no run do tratamento, recusados;
- o contrato canônico intacto: padrões de `val` e `test` em
  `event_metrics.json`, sem campo novo, e o relatório só de `val` recusado
  pelos validadores canônicos;
- `decide` registrando uma vez, pulando a decisão igual, recusando a
  diferente, recusando sem registro cada entrada incompleta ou adulterada e
  registrando INVÁLIDO com V2 ou V4 reprovadas.

Os selftests canônicos das armas (`gatefall.eval.baseline_a` e os demais)
continuam cobrindo a orquestração com `val` e `test`.
