# Protocolo proposto — piloto de aumento com GMDCSA24 (braço A)

!!! warning "Status: PROPOSTO, não aprovado"
    Este documento propõe um experimento. Ele não autoriza treino, extração de
    features nem implementação de adapter. O protocolo canônico do GateFall, a
    receita congelada, os splits Le2i-CS e Le2i-CV, os checkpoints, a
    calibração, o protocolo de alarme e as referências em `runs/reference/`
    não mudam. A aprovação exige os três passos de
    [Aprovação](#aprovacao) e, até lá, a resposta padrão é não.

É a etapa 1 da [decisão de viabilidade](gmdcsa24-feasibility.md#decisao):
um protocolo separado, fixado antes de qualquer treino, para um piloto
isolado e exploratório só do braço A. O resultado nunca substitui uma
referência.

## O que o piloto desloca

Pela regra de risco dominante, o piloto declara o que ocupa e o que fica
parado:

- **Ocupa:** a revisão manual de 52 fronteiras em 30 vídeos, mais as
  reposições de estrato que as exclusões exigirem
  ([Revisão de proveniência](gmdcsa24-provenance-review.md)), a extração
  offline de pose YOLO-Pose dos 160 vídeos do GMDCSA24, a implementação de um
  adapter isolado e cinco treinos do braço A.
- **Desloca:** qualquer treino de B0, B1, C0 ou C1 com GMDCSA24 e qualquer
  outro dataset externo até a decisão deste piloto. Não cria braço, métrica,
  backbone nem split novo.
- **Não toca:** a extração DINOv3 e SAM 3, que o braço A não usa. Elas só
  entram numa expansão aprovada à parte.

## Hipótese

Somar ao treino Le2i-CS as janelas elegíveis do GMDCSA24 melhora a macro-F1
restrita do braço A na validação Le2i-CS sem piorar a detecção de eventos
nem os alarmes falsos sob o protocolo de alarme congelado.

## Controle e tratamento

| | Controle | Tratamento |
| --- | --- | --- |
| Braço | A (TCN sobre pose, 134 features) | A, sem mudança de arquitetura |
| Dados de treino | Treino Le2i-CS (133 vídeos, 5 219 janelas) | Treino Le2i-CS + janelas elegíveis do GMDCSA24 |
| Validação e teste | `val` e `test` oficiais do Le2i-CS | Os mesmos, byte a byte |
| Receita | [Receita congelada](../train/baseline-a.md#receita-de-treino-congelada) | A mesma |
| Seeds | 42, 43, 44, 45, 46 | 42, 43, 44, 45, 46 |
| Protocolo de alarme | `BASELINE_A_ALARM_PROTOCOL` | O mesmo |

O controle são os cinco runs do [sumário multi-seed](multiseed-summary.md#resultado-da-execucao-real),
aceitos só se o `checkpoint_sha256` de cada seed bater com a tabela daquela
página. Se não bater, o controle é retreinado na mesma máquina do
tratamento, com a mesma receita.

Só três itens podem diferir entre controle e tratamento, e todos são
consequência da regra, não escolha nova: o manifesto de treino, o vetor de
pesos da `CrossEntropyLoss` (frequência inversa, calculada sobre o treino de
cada lado) e o JSON de padronização (calculado sobre o treino de cada lado).
Qualquer outra diferença de configuração invalida o piloto.

## Proveniência dos dados

| Item | Valor fixado |
| --- | --- |
| Vídeos | GMDCSA24 tag `v2.1`, commit `5abac7693229900cf80f722e878fbb119211fc1c`, `verify` OK |
| Identidade | `source_sha256` por vídeo, gravado no manifesto do adapter |
| Rótulos | Somente OmniFall `labels`, revisão `68e5cee`; o CSV do autor fica só como auditoria |
| Proveniência visual | `review-status` com saída `APROVADA` ([revisão](gmdcsa24-provenance-review.md)) |
| Sujeitos | S1–S4, todos no treino; o GMDCSA24 nunca é avaliado |

## Fronteiras de validação e teste

- `val` e `test` do Le2i-CS são os oficiais do OmniFall, sem nenhuma janela
  ou vídeo do GMDCSA24. A única estatística com GMDCSA24 aplicada a eles é a
  padronização do tratamento, ajustada só sobre o treino aumentado (ver
  [Normalização só com treino](#normalizacao-so-com-treino)).
- O índice de janelas de `val` e `test` precisa ter o mesmo hash no controle e
  no tratamento.
- Não há avaliação no GMDCSA24 nem nos splits `gmdcsa24-{cs,cv}`. O teste S4
  teria 17 quedas e não entra em nenhuma decisão.
- O Le2i-CV fica fora do piloto.

## Normalização só com treino

Os dois braços ajustam a padronização só sobre o próprio treino:

- **Controle:** o JSON de padronização existente, ajustado só sobre o treino
  Le2i-CS. Ele não é recalculado e não recebe nenhuma janela do GMDCSA24.
- **Tratamento:** recalcula a padronização z-score com `compute_train_stats`
  sobre todas as janelas do seu treino aumentado (Le2i-CS mais GMDCSA24
  elegível), com o mesmo stride, as mesmas dimensões excluídas e o mesmo
  limite de desvio mínimo. O resultado vai para um JSON próprio, nunca para o
  caminho do Le2i.
- `val` e `test` de cada lado são padronizados com o JSON daquele lado. Para o
  tratamento, isso significa aplicar a `val` e `test` estatísticas que contêm
  GMDCSA24. É permitido, porque elas foram ajustadas só sobre o treino
  aumentado, sem nenhuma amostra de `val` ou `test`. É o terceiro item
  declarado em [Controle e tratamento](#controle-e-tratamento).

Proibido nos dois braços:

- janela, vídeo ou quadro do GMDCSA24 entre as amostras de `val` ou `test`
  do Le2i-CS;
- GMDCSA24 em qualquer estatística do controle (JSON de padronização ou pesos
  da perda);
- qualquer JSON de padronização ajustado com janelas de `val` ou `test`;
- GMDCSA24, `val` ou `test` na calibração do alarme:
  `BASELINE_A_ALARM_PROTOCOL` fica congelado e não é recalibrado em nenhum
  dos lados;
- qualquer decisão tomada com `test`, seja de limiar, época, seed, exclusão
  ou critério.

O pré-requisito 3 da [viabilidade](gmdcsa24-feasibility.md#pre-requisitos-sem-treino)
proíbe GMDCSA24 "nas estatísticas de padronização desses testes". Este
protocolo o lê assim: nenhuma estatística ajustada com amostras de `val` ou
`test` e nenhuma estatística do controle contém GMDCSA24. A padronização do
tratamento, ajustada só no treino dele, pode ser aplicada a `val` e `test`
do tratamento.

O relatório do piloto mostra a diferença de média e desvio por bloco de
features entre os dois JSONs, para separar efeito de dados de efeito de
escala.

## Seleção de janelas e replicação de borda

| Regra | Le2i (controle e tratamento) | GMDCSA24 (só tratamento) |
| --- | --- | --- |
| Stride | `TRAIN_STRIDE = 4` | `TRAIN_STRIDE = 4` |
| Rótulo | Quadro `k_end` (many-to-one) | Quadro `k_end` |
| `IGNORE_LABEL` em `k_end` | Descartada | Descartada |
| `k_end < WINDOW_FRAMES - 1 = 23` | Mantida, como hoje | **Descartada** |
| Classe `lie_down` ou `lying` em `k_end` | Não ocorre no treino | **Descartada** |

- A grade é a de `build_time_grid` (taxa constante), igual à do Le2i. Por
  isso as fronteiras com desvio CFR acima de 0,1 s bloqueiam a revisão e o
  vídeo só entra excluído (ver [Exclusões](#exclusoes)).
- Janelas com replicação de borda saem só do GMDCSA24. Lá, 68% das quedas
  começam antes do quadro 23, contra 8,5% no Le2i. O Le2i fica como está,
  para o controle não mudar.
- `lie_down` e `lying` têm suporte zero no treino Le2i-CS e peso zero na
  perda. Janelas dessas classes no GMDCSA24 ganhariam peso de frequência
  inversa sobre poucas amostras e mudariam a cabeça. `RESTRICTED_CLASSES`
  fica `{0, 1, 2, 3, 4, 7, 8, 9}` nos dois lados.

Estimativa antes da revisão, com a grade do `probe`: o GMDCSA24 tem 3 231
janelas rotuladas. Saem 945 com replicação de borda, 142 de `lie_down` ou
`lying` sem replicação e 25 dos dois vídeos excluídos abaixo. Restam 2 119
janelas em 149 vídeos (41% do treino Le2i-CS), sendo 248 de `fall` e 483 de
`fallen`. O relatório do piloto refaz essa contagem por classe sobre os dados
reais.

## Exclusões

Fixadas antes do treino:

1. `Subject_3/ADL/09`: `stand_up [0; 7,78)` cobre `sitting` e `standing`.
   Fica fora, sem correção.
2. `Subject_1/Fall/12`: na revisão, o início da queda fica a +0,116 s do
   quadro da grade CFR. Fica fora enquanto a grade não usar PTS.
3. `Subject_2/Fall/03`: fica como o OmniFall anotou (`lie_down`/`lying`, sem
   `fall`). Pelas regras de classe acima, essas janelas saem.
4. Todo vídeo com caso `excluded` na revisão de proveniência.

A lista final de `video_id` excluídos, com o motivo, é gravada no manifesto do
adapter antes do treino. Exclusão nova depois de ver qualquer métrica
invalida o piloto.

## Métricas

Todas calculadas pelos avaliadores existentes, sem mudança:

- **Primária:** `macro_f1_restricted` de `val` (classes fixas acima).
- **Eventos (`val`):** `n_missed_events`, `n_false_alarms`,
  `false_alarms_per_hour` e latência mediana, sob
  `BASELINE_A_ALARM_PROTOCOL`.
- **Secundária (`val`):** F1 da classe `fall`.
- **Descritivas (`test`):** as mesmas métricas e o
  [bootstrap agrupado por sujeito](grouped-bootstrap.md), calculados uma vez,
  depois da decisão. Não mudam a decisão e não escolhem nada.

A comparação é pareada por seed: Δ = tratamento − controle na mesma seed.

## Critério de GO/NO-GO

A decisão usa só `val`. Primeiro vêm as condições de validade. Qualquer
falha torna o piloto **INVÁLIDO**: não é GO nem NO-GO, e a causa é
corrigida antes de repetir o piloto.

| # | Condição de validade |
| --- | --- |
| V1 | `review-status` com proveniência `APROVADA` sobre o clone pinado |
| V2 | Nenhuma janela ou `video_id` do GMDCSA24 nas amostras de `val`/`test`, nas estatísticas do controle ou na calibração do alarme; JSON de padronização de cada lado ajustado só sobre o treino daquele lado (o do tratamento pode conter GMDCSA24; ver [Normalização](#normalizacao-so-com-treino)); índices de `val`/`test` com o mesmo hash nos dois lados |
| V3 | Controle aceito pela tabela de `checkpoint_sha256` ou retreinado na mesma máquina |
| V4 | Configuração idêntica fora dos três itens declarados em [Controle e tratamento](#controle-e-tratamento) |
| V5 | Para cada vídeo do GMDCSA24, a decodificação do adapter devolve o mesmo número de quadros que `ffprobe -count_frames` (ver [riscos](#riscos-em-aberto)) |

Com as cinco válidas, o resultado é **GO** só se as quatro condições abaixo
passarem. Caso contrário, é **NO-GO**.

| # | Condição de GO (`val`, cinco seeds) |
| --- | --- |
| G1 | Média de Δ `macro_f1_restricted` >= +0,015 e Δ > 0 em pelo menos 4 das 5 seeds |
| G2 | Soma de `n_missed_events` do tratamento <= soma do controle |
| G3 | Soma de `n_false_alarms` do tratamento <= soma do controle |
| G4 | Média de Δ F1 de `fall` >= 0 |

O limite de G1 é próximo do desvio entre seeds do controle em `val`
(0,0136). A `val` tem 19 vídeos e 13 quedas, então um GO é fraco por
construção. Ele só autoriza propor a etapa 3 (B0, B1, C0 e C1 com a mesma
política de aumento, cada um contra o próprio controle). Não promove nada ao
protocolo canônico: isso exigiria, pela invariante 1, a mesma mudança de
dados em A, B e C e o retreino das três configurações.

## Aprovação

O protocolo passa a valer só quando os três passos estiverem cumpridos, nesta
ordem:

1. A revisão de proveniência termina com `review-status` `APROVADA`, e os
   julgamentos são transcritos para a
   [página de revisão](gmdcsa24-provenance-review.md).
2. Um PR separado muda o status deste documento para aprovado, com a decisão
   explícita do responsável pelo projeto.
3. O adapter isolado (`data/processed/gmdcsa24/`), a montagem do treino
   aumentado e as checagens V2 e V5 são implementados e revisados em PR
   próprio, com selftests sintéticos, sem alterar os selftests e as
   referências do Le2i.

Só depois disso algum treino do piloto pode rodar.

## Riscos em aberto

- **Decodificação de vídeo com taxa variável.** `gatefall.data.video_io`
  lê o pipe do `ffmpeg` sem `-fps_mode passthrough`. Nos MP4 do GMDCSA24, o
  `ffmpeg` então duplica quadros para uma taxa constante: em
  `Subject_1/Fall/12`, o pipe entrega 206 quadros, e o vídeo tem 200. O índice
  de quadro de origem da grade deixaria de corresponder ao quadro decodificado.
  O Le2i (AVI de taxa constante) não é afetado. O adapter do GMDCSA24 precisa
  decodificar sem duplicar quadros, e V5 verifica isso.
- **Proveniência ainda não confirmada por humano.** Os 52 casos estão
  `pending`. A evidência automática não basta.
- **Resolução e taxa.** O GMDCSA24 é 1280×720 ou 640×480, a ~30 fps; o Le2i é
  320×240 ou 320×180, a 24–25 fps. O efeito nas features de pose não foi medido.
- **Poucos sujeitos.** Quatro sujeitos, com S2 em 35% dos quadros e ambiente
  confundido com sujeito.
- **`val` pequena.** A decisão depende de 19 vídeos e 13 quedas.
- **Licença e consentimento.** Anotações CC BY-NC 4.0; consentimento de
  imagem não documentado. Nenhum quadro entra no Git.
