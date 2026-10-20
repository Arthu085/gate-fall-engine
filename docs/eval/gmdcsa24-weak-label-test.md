# `test` descritivo do piloto GMDCSA24 (T e S, braço A)

`describe-test`, em `gatefall.eval.gmdcsa24_weak_label`, calcula as métricas
descritivas de `test` do Le2i-CS dos runs de T e S, uma única vez por
tratamento e seed, estritamente depois da
[decisão registrada](gmdcsa24-weak-label-val.md#registro). Pelo
[critério de parada 5](../analysis/gmdcsa24-weak-label-protocol.md#criterios-de-parada),
`test` não decide nada: o comando não aplica limiar, não aprova nem reprova
condição e não reabre a decisão.

**Estado:** implementação apenas. Neste PR não rodou nenhuma avaliação real
de `test`, nenhum treino, nenhuma preparação e nenhuma decisão nova. O
responsável pelo projeto informou que os dez treinos de T e S (seeds 42 a 46)
e as dez avaliações de eventos em `val` terminaram e que a decisão registrada
é **NO-GO**, em `runs/local/gmdcsa24_weak_label/decision.json` com SHA-256
`9f57ab36d628194950a7edb490a30175a5aba90ae7bcf72c41a7445904ceae5d`. Esse
SHA-256 está fixado no código como `PINNED_DECISION_SHA256`; o registro não é
alterado.

## Antes de qualquer acesso a `test`

Nenhuma feature, predição ou métrica de `test` é montada, lida ou calculada
antes de todas estas conferências:

| Conferência | O que é exigido |
| --- | --- |
| Execução | Raiz do repositório, como em `evaluate` e `decide` |
| Decisão registrada | `decision.json` existe, é arquivo regular (não link simbólico) e tem o SHA-256 fixado |
| Seed | A seed está entre as da decisão registrada |
| Campos | Exatamente os campos de uma decisão válida, critério do protocolo, `split = "val"`, `evaluated_splits = ["val"]` e as seeds do controle (42 a 46) |
| Decisão | GO exploratório, GO frágil ou NO-GO; INVÁLIDO não libera `test` |
| Proveniência | `code.commit` completo, `code.dirty = false`, descendente do commit da extração e ancestral do HEAD atual; `recorded_at` válido |
| Seeds | Entradas de T e S para cada seed do controle, nem a mais nem a menos, cada uma no run canônico e com o mesmo controle em T e S |
| Hashes | Cada arquivo que a decisão leu (`treatment.json`, `metrics.json` e `val_event_metrics.json` do run; `metrics.json` e `event_metrics.json` do controle) com o SHA-256 registrado |
| Coerência | A decisão recalculada desses arquivos (seeds, decisão, G1–G4 e medidas por seed) é a registrada |
| Controle | O controle do `treatment.json` é o que a decisão leu |
| Run | Os portões de [`evaluate`](gmdcsa24-weak-label-val.md#o-que-evaluate-confere-antes-de-avaliar): as 11 condições do `preflight`, run completo e `event_metrics.json` canônico do controle válido; o par `alarm_protocol.yaml`/`val_event_metrics.json` íntegro; e run, preparação e controle sem mudança durante a conferência |
| Saída | Diretório do resultado fora dos artefatos protegidos e ignorado pelo Git |

Qualquer falha recusa o comando sem criar nada em `test_descriptive/`.
Só as recusas do estado da própria seed (tabela abaixo) acontecem depois de o
diretório da seed existir, e nenhuma delas lê `test`.

## Avaliação

Só o tratamento é predito; o controle não é reavaliado. Do braço A são
reaproveitados, sem mudança:

- o carregamento do checkpoint e a predição por janela do avaliador canônico;
- a padronização, com o JSON do tratamento já conferido;
- as métricas de classificação de `final.test` do treino canônico (janelas
  rotuladas em `EVAL_STRIDE`, `macro_f1_restricted`, F1 por classe, suporte,
  matriz de confusão), com o hash do índice de `test` igual ao publicado;
- a grade de eventos (`EVAL_STRIDE`, sem descartar `IGNORE_LABEL`), a
  associação de eventos e `BASELINE_A_ALARM_PROTOCOL`.

O `test` do controle é o já publicado: `final.test` do `metrics.json` e
`splits.test` do `event_metrics.json` canônicos, validados pelo validador do
[sumário multi-seed](../analysis/multiseed-summary.md). Antes da publicação,
janelas, horas, quedas e suporte por classe de `test` precisam ser iguais nos
dois lados, e a decisão, os arquivos que ela leu, o run, a preparação e o
controle (fora os arquivos de lock) precisam ter os mesmos SHA-256 do início.

A comparação é descritiva e pareada por seed: Δ = tratamento − controle para
`macro_f1_restricted`, F1 por classe, cada campo numérico de eventos fora da
estrutura da grade e média e mediana da latência. Não há limiar, campo
`passed` nem nova decisão.

## Uma única vez por tratamento e seed

Antes de acessar `test`, o comando grava uma reserva (`claim.json`) sob lock
exclusivo da seed. Antes de publicar, a reserva passa a registrar o SHA-256 do
resultado; uma reprovação de integridade depois de calcular `test` também
fica registrada nela.

| Estado | Sem `--recovery-reason` | Com `--recovery-reason` |
| --- | --- | --- |
| Nada avaliado | Avalia e publica | Recusa: nada a recuperar |
| Tentativa interrompida antes da publicação (queda ou erro técnico) | Recusa sem ler `test` | Refaz uma única vez e registra o motivo nas tentativas |
| Duas tentativas registradas | Recusa sem ler `test` | Recusa sem ler `test` |
| Reprovação de integridade registrada (estrutura de `test`, decisão, entradas, run, preparação ou controle mudados) | Recusa sem ler `test` | Recusa sem ler `test` |
| Resultado publicado | Recusa sem ler `test` | Recusa sem ler `test` |
| Resultado publicado e apagado | Recusa | Recusa |
| Diretório da seed sem `claim.json` | Recusa | Recusa |

O resultado é publicado por `os.link`, que nunca substitui um arquivo
existente. Não há `--force`. Uma tentativa interrompida só pode ser refeita
uma vez, por falha técnica, como no
[critério de parada 3](../analysis/gmdcsa24-weak-label-protocol.md#criterios-de-parada).
O motivo é registrado, não verificado; como nenhuma métrica é impressa antes
da publicação, a repetição não pode ser escolhida pelo resultado. Nenhum
resultado publicado é refeito. Apagar o diretório inteiro da seed fica fora do
que o comando consegue detectar e é uma intervenção manual; mesmo assim, a
existência de `test_descriptive/` continua bloqueando a decisão e o que a
sustenta.

## Depois de acessar `test`

Com `runs/local/gmdcsa24_weak_label/test_descriptive/` presente, nada que
sustenta a decisão é refeito, mesmo que `decision.json` seja removido:
`decide` não registra decisão nova, e `prepare`, `train` e `evaluate` recusam
gravar ou recuperar, mesmo com `--force`. Com `decision.json` presente, um
`decide` que recalcula a mesma decisão continua só confirmando.

## Resultado

`test_descriptive/{t,s}/seed_<n>/test_metrics.json` traz:

- `descriptive_only = true`, `split = "test"`, `evaluated_splits = ["test"]`,
  tratamento e seed;
- `decision`: caminho, SHA-256, decisão, data e commit da decisão registrada;
- `checkpoint`: caminho e SHA-256 do checkpoint do tratamento;
- `run`, `prepared` e `control`: SHA-256 de cada arquivo do run, da
  preparação e do run do controle;
- `split_provenance`: grade do Le2i-CS, com SHA-256, e hash do índice de
  `test`;
- `alarm_protocol`: o `BASELINE_A_ALARM_PROTOCOL`;
- `metrics.treatment` e `metrics.control`: classificação e eventos de `test`;
- `delta`: tratamento − controle;
- `attempts`, commit do código e data.

```text
runs/local/gmdcsa24_weak_label/
├── decision.json                    # só lido
├── {t,s}/seed_<n>/                  # só lidos
└── test_descriptive/{t,s}/seed_<n>/
    ├── claim.json
    ├── .test.lock
    └── test_metrics.json
```

Tudo é ignorado pelo Git e não é sincronizado com o Google Drive. Os runs de
T e S, os relatórios de `val`, as preparações, os controles e `decision.json`
não são alterados.

## O que fica fora

- O [bootstrap agrupado por sujeito](../analysis/grouped-bootstrap.md) de
  `test`, citado entre as métricas descritivas do
  [protocolo verificado](../analysis/gmdcsa24-pilot-protocol.md#metricas), não
  é calculado aqui.
- Não há resumo entre seeds: cada resultado traz o tratamento, o controle
  pareado e o Δ da própria seed.

## Como executar

Na raiz de um checkout limpo, com a decisão registrada, uma vez para cada
tratamento e seed:

```bash
for treatment in T S; do
  for seed in 42 43 44 45 46; do
    uv run python -m gatefall.eval.gmdcsa24_weak_label describe-test \
      --treatment "$treatment" --seed "$seed"
  done
done
```

Só depois de uma falha técnica antes da publicação:

```bash
uv run python -m gatefall.eval.gmdcsa24_weak_label describe-test --treatment T --seed 42 \
  --recovery-reason "processo interrompido por falta de memória"
```

## Selftest

O `selftest` do módulo (na CI) roda também o selftest do `test` descritivo. No
mundo sintético do [selftest da decisão](gmdcsa24-weak-label-val.md#selftest),
com uma leitura de features que reprova `test` até ser liberada, ele cobre:

- `describe-test` sem `decision.json` recusado sem ler `test` nem criar
  resultado;
- recusa sem ler `test` com SHA-256 diferente do fixado, decisão INVÁLIDO,
  decisão ou medida diferente da recalculada, seed ausente ou seeds diferentes
  das do controle, `test` entre os splits avaliados, run fora do canônico,
  código sujo, link simbólico, seed sem run, seed com run mas fora da decisão,
  checkout sujo e
  relatório de `val`, `event_metrics.json` do controle ou `treatment.json`
  alterados depois da decisão;
- publicação com só o vídeo de `test` lido, proveniência completa, controle
  igual ao `test` canônico publicado, e o caminho do tratamento aplicado ao
  controle reproduzindo `final.test` e `splits.test` canônicos;
- Δ igual a tratamento − controle, sem `passed`;
- runs, relatórios de `val`, preparações, controle e `decision.json` intactos;
- segunda execução da mesma seed, resultado apagado e resultado e reserva
  apagados recusados, com e sem motivo, sem nem tentar ler `test`;
- com `test` acessado e sem `decision.json`, `decide`, `evaluate --force` e
  `train --force` recusados sem gravar;
- tentativa interrompida recusada sem motivo, terceira tentativa recusada,
  reprovação de integridade registrada e nunca refeita, e a repetição única
  com motivo publicando com as duas tentativas registradas, com resultados
  distintos de T e S.
