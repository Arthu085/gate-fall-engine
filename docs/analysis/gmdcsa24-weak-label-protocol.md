# Protocolo aprovado — piloto exploratório GMDCSA24 com rótulos fracos (braço A)

!!! warning "Status: APROVADO em 2026-10-19; passo 4 vale só a partir da integração do PR #135"
    O responsável pelo projeto aprovou este protocolo exploratório separado
    (ver [Registro da aprovação](#registro-da-aprovacao)), e o adapter do
    passo 3 foi integrado no PR #134. A autorização do passo 4, proposta no PR
    #135, vale só a partir da revisão e da integração desse PR pelo
    responsável. Ela cobre apenas a execução local e isolada da extração
    offline YOLO-Pose dos 145 vídeos elegíveis e do piloto T/S do braço A, e
    cada etapa depende de uma implementação revisada em PR próprio e dos
    portões de validação dela (ver
    [Autorização do passo 4](#autorizacao-do-passo-4)).
    Sincronização com o Google Drive, mudança do modelo canônico, treinos de
    B0, B1, C0 ou C1 e outras variantes de rótulo fraco continuam
    [proibidos](#o-que-continua-proibido) em qualquer caso. As anotações
    OmniFall do GMDCSA24 continuam não verificadas, e a triagem visual por IA
    nunca vale como confirmação humana. O
    [protocolo verificado](gmdcsa24-pilot-protocol.md) continua PROPOSTO e
    intacto, e a condição V1 dele continua pendente: nada nesta página a dá por
    cumprida. O protocolo canônico do GateFall, a receita congelada, os splits
    Le2i-CS e Le2i-CV, os checkpoints, o protocolo de alarme e as referências
    A, B0, B1, C0 e C1 não mudam.

Este protocolo é uma trilha paralela ao protocolo verificado, não uma versão
dele. Ele troca a revisão manual quadro a quadro por uma decisão explícita de
aceitar rótulos não verificados, e por isso sustenta alegações mais fracas.
Tudo o que não aparece em [O que muda](#o-que-muda-em-relacao-ao-protocolo-verificado)
é idêntico ao protocolo verificado e vale por referência.

## Decisão registrada

- Em 2026-10-19, o responsável pelo projeto decidiu aceitar a proveniência
  não verificada das anotações OmniFall do GMDCSA24 (config `labels`, revisão
  `68e5cee`) sobre os MP4 da tag `v2.1`, **só para esta trilha** exploratória.
- Os rótulos entram como **rótulos fracos**: coerentes no nível do vídeo
  (ver [viabilidade](gmdcsa24-feasibility.md#correspondencia-com-o-omnifall-o-que-foi-verificado)),
  com ruído de fronteira não estimado e possivelmente sistemático.
- A decisão dispensa o pré-requisito 2 da
  [viabilidade](gmdcsa24-feasibility.md#pre-requisitos-sem-treino) só para este
  piloto. Ele continua exigido para o protocolo verificado, para a etapa 3
  (B0, B1, C0 e C1) e para qualquer promoção ao protocolo canônico.
- A [revisão de proveniência](gmdcsa24-provenance-review.md) continua
  PENDENTE. `review.csv` e `review-status` não mudam por causa desta trilha, e
  a revisão humana pode prosseguir de forma independente.

## O que o piloto desloca

- **Ocupa:** a extração offline de pose YOLO-Pose dos vídeos do GMDCSA24, um
  adapter isolado com a política de exclusão desta página, dez treinos do
  braço A (cinco do tratamento principal T e cinco da análise de
  sensibilidade S) e a checagem do controle.
- **Desloca:** a revisão manual como pré-condição deste piloto (ela não é
  cancelada); o treino do protocolo verificado, que não roda ao mesmo tempo
  (um experimento com GMDCSA24 por vez); qualquer treino de B0, B1, C0 ou C1
  com GMDCSA24; qualquer outro dataset externo; e qualquer outra variante de
  rótulo fraco (outra margem, outra lista de exclusão, reetiquetagem ou
  deslocamento de rótulo).
- **Não toca:** a extração DINOv3 e SAM 3, que o braço A não usa.

## O que muda em relação ao protocolo verificado

| Item | Protocolo verificado | Este protocolo |
| --- | --- | --- |
| Proveniência visual | V1: `review-status` `APROVADA` | Aceita sem verificação pela [decisão registrada](#decisao-registrada); V1 não se aplica e não é dada por cumprida |
| Natureza dos rótulos | Conferidos numa amostra estratificada | Fracos; o ruído fica [explícito e não resolvido](#ruido-de-rotulo-nao-resolvido) |
| Exclusão de vídeos | Três exclusões fixas e os casos `excluded` da revisão | Os 15 vídeos com anomalia documentada ([política](#politica-de-inclusao-e-exclusao)) |
| Tratamentos | Um | T e a análise de sensibilidade S, ambos fixados aqui |
| Treinos novos | 5 | 10 (5 de T e 5 de S) |
| Alcance de um GO | Propor a etapa 3 | Nenhuma expansão ([alegações](#o-que-a-evidencia-sustenta)) |

Ficam idênticos, pelas seções do protocolo verificado:

- [hipótese](gmdcsa24-pilot-protocol.md#hipotese), lida com rótulos fracos;
- [controle e tratamento](gmdcsa24-pilot-protocol.md#controle-e-tratamento):
  braço A sem mudança de arquitetura, receita congelada, seeds 42 a 46,
  `BASELINE_A_ALARM_PROTOCOL`, controle aceito pelo `checkpoint_sha256` ou
  retreinado na mesma máquina. T e S são comparados com o mesmo controle,
  cada um com Δ = tratamento − controle na mesma seed;
- os três únicos itens que podem diferir do controle (manifesto de treino,
  pesos da `CrossEntropyLoss` e JSON de padronização), agora para T e para S,
  cada um calculado sobre o próprio treino;
- [fronteiras de validação e teste](gmdcsa24-pilot-protocol.md#fronteiras-de-validacao-e-teste):
  `val` e `test` oficiais do Le2i-CS, byte a byte, sem GMDCSA24; Le2i-CV
  fora do piloto; nenhuma avaliação no GMDCSA24;
- [normalização só com treino](gmdcsa24-pilot-protocol.md#normalizacao-so-com-treino):
  o controle usa o JSON existente; T e S ajustam cada um o próprio JSON só
  sobre o próprio treino aumentado, gravado fora do caminho do Le2i;
- [seleção de janelas](gmdcsa24-pilot-protocol.md#selecao-de-janelas-e-replicacao-de-borda):
  `TRAIN_STRIDE = 4`, rótulo no quadro `k_end`, descarte de `IGNORE_LABEL`,
  de `k_end < 23` e de `lie_down`/`lying` no GMDCSA24;
- [métricas](gmdcsa24-pilot-protocol.md#metricas) e as condições G1–G4 e
  V2–V5 do protocolo verificado.

## Triagem visual assistida por IA (evidência preliminar)

As 52 contact sheets da revisão de proveniência (52 fronteiras em 30 vídeos)
passaram por uma triagem visual feita por um modelo de IA. O resultado:

| Fronteira | Possível divergência | Aparentemente compatível | Inconclusivo | Total |
| --- | ---: | ---: | ---: | ---: |
| Início do `fall` | 7 | 8 | 1 | 16 |
| Fim do `fall` | 9 | 6 | 2 | 17 |
| Transição sem `fall` | 9 | 2 | 5 | 16 |
| Início ou fim de lacuna | 1 | 1 | 1 | 3 |
| **Total** | **26** | **17** | **9** | **52** |

Esses números **não são erros de rótulo verificados** nem uma taxa de erro
representativa do dataset:

- A amostra não é aleatória: 14 dos 30 vídeos entraram por terem anomalia
  documentada, e os outros 16 são dois por estrato sujeito × categoria.
- A triagem olhou quadros estáticos a ±0,5 s da fronteira, com o rótulo
  anotado impresso em cada quadro, sem definição operacional do início de
  `fall` e de `fallen` e sem segundo avaliador.
- "Aparentemente compatível" não demonstra a tolerância de 0,1 s, e
  "possível divergência" é uma hipótese, não uma `discrepancy`.
- Os outros 130 vídeos não passaram por revisão visual equivalente.

O padrão descrito pela triagem, ainda como hipótese, é este: inícios de
`fall` marcados antes de o movimento ficar visível (até cerca de 0,4 s) e
fins de `fall`, `sit_down` e `lie_down` marcados depois de a postura final
aparecer. Se for real, é ruído de fronteira de alguns décimos de segundo,
concentrado perto das transições. Essa hipótese motivou a
[análise de sensibilidade](#analise-de-sensibilidade-predefinida-s) e
informou a margem dela. S não isola esse ruído.

Regras para essa evidência:

- **Julgamento de IA nunca vira registro de revisor humano.** Nenhuma coluna
  humana de `review.csv` (`review_outcome`, `observed_boundary_s`,
  `reviewer`, `reviewed_at`, `notes`) é preenchida ou pré-preenchida com a
  triagem, e nenhum caso conta como `confirmed`, `discrepancy` ou `excluded`
  por causa dela.
- A triagem não entra na exclusão de vídeos, nas regras de janela de T, nas
  condições de validade nem no critério de decisão. A única influência dela
  num parâmetro é a margem de 0,5 s de S, escolhida depois da triagem e
  declarada em [S](#analise-de-sensibilidade-predefinida-s).
- O relatório fica só na máquina local, em
  `data/scratch/gmdcsa24/provenance/gmdcsa24_relatorio_triagem_visual_ia.md`
  (ignorado pelo Git), com SHA-256
  `c3de33da6daaf5e3b3462534cdca0a8a4661ebecaaab2c9e2ac67f86463dbdd9`.

## Política de inclusão e exclusão

Os rótulos OmniFall pinados entram como estão: **sem reetiquetar, deslocar
instantes, cortar ou mesclar segmentos**. Onde há anomalia documentada, sai o
vídeo inteiro; nada é corrigido. Toda regra é recalculada pelo código
existente a partir do clone pinado e do OmniFall pinado. Nenhuma regra de
vídeo ou de janela de T usa a triagem por IA, métrica de modelo ou julgamento
caso a caso. A margem de S é a exceção declarada: a triagem a informou.

### Vídeos

1. **Fonte.** Clone da tag `v2.1` (commit
   `5abac7693229900cf80f722e878fbb119211fc1c`) com `verify` OK e os 160
   identificadores comuns ao clone, ao CSV original, ao `labels` e aos splits
   do OmniFall (`inventory` OK). Sem isso, o piloto não começa.
2. **Anomalia documentada.** Sai todo vídeo com motivo em `anomaly_reasons`
   (`gatefall.data.gmdcsa24.provenance`), o mesmo cálculo da inclusão
   obrigatória da revisão:
    - sobreposição OmniFall >= 0,1 s;
    - lacuna OmniFall >= 0,5 s sobre a duração medida;
    - vídeo `Fall` sem segmento `fall`;
    - `fall` sem intervalo `Falling` no CSV do autor;
    - início do `fall` a 1,5 s ou mais do `Falling` do autor;
    - desvio de PTS acima de 0,1 s.
3. **Não excluem:** os desvios de formato do CSV do autor e o `Length` de
   `Subject_2/ADL/04` (o CSV do autor só serve para auditoria e não mexe nos
   rótulos OmniFall), as sobreposições de arredondamento abaixo de 0,1 s
   (resolvidas por `labels_for_grid`, como no Le2i) e o resultado da triagem.

Os 15 vídeos excluídos, recalculados em 2026-10-19 pela análise local de
[Contagem esperada](#contagem-esperada):

| Vídeo | Motivo |
| --- | --- |
| `Subject_1/ADL/04` | desvio de PTS de 0,108 s |
| `Subject_1/ADL/05` | lacuna de 1,00 s |
| `Subject_1/Fall/03` | início do `fall` −1,70 s do autor |
| `Subject_1/Fall/04` | início do `fall` −1,58 s do autor |
| `Subject_1/Fall/09` | desvio de PTS de 0,101 s |
| `Subject_1/Fall/12` | desvio de PTS de 0,131 s |
| `Subject_1/Fall/15` | início do `fall` −1,54 s do autor |
| `Subject_2/ADL/21` | lacuna de 1,26 s |
| `Subject_2/Fall/03` | vídeo `Fall` sem `fall` no OmniFall |
| `Subject_2/Fall/15` | início do `fall` −1,56 s do autor |
| `Subject_3/ADL/09` | sobreposição de 4,30 s |
| `Subject_3/Fall/08` | desvio de PTS de 0,104 s |
| `Subject_3/Fall/09` | sobreposição de 0,10 s |
| `Subject_4/ADL/02` | lacuna de 0,56 s |
| `Subject_4/Fall/15` | `fall` sem intervalo `Falling` no CSV original |

`Subject_4/ADL/02` não está na revisão de proveniência: a lacuna fica no fim
do vídeo e não forma fronteira revisável. Os outros 14 são as inclusões
obrigatórias da revisão. O protocolo verificado mantém parte deles depois da
confirmação humana. Sem essa confirmação, aqui todos saem.

### Janelas

As regras de janela são as do
[protocolo verificado](gmdcsa24-pilot-protocol.md#selecao-de-janelas-e-replicacao-de-borda),
sem mudança. O tratamento S remove ainda as janelas perto de fronteira
([abaixo](#analise-de-sensibilidade-predefinida-s)).

### Contagem esperada

Calculada em 2026-10-19 por uma análise local não versionada, com a
[receita](#receita-de-contagem-e-congelamento-w2) e as entradas pinadas
abaixo. Esses números ainda não foram reproduzidos de forma independente. A
mesma análise, com as regras do protocolo verificado, dá 2 119 janelas em 149
vídeos, igual à estimativa publicada lá. Essa estimativa também veio de uma
análise local, então a coincidência mostra consistência, não verificação
independente.

| Etapa | Janelas |
| --- | ---: |
| Janelas rotuladas do GMDCSA24 (stride 4) | 3 231 |
| − replicação de borda (`k_end < 23`) | −945 |
| − `lie_down` ou `lying`, sem replicação de borda | −142 |
| − janelas restantes dos 15 vídeos excluídos | −171 |
| **T** | **1 973** |
| − janelas a até 0,5 s de uma fronteira | −411 |
| **S** | **1 562** |

| Classe | Protocolo verificado (estimativa) | T | S |
| --- | ---: | ---: | ---: |
| `walk` | 142 | 113 | 68 |
| `fall` | 248 | 225 | 121 |
| `fallen` | 483 | 440 | 364 |
| `sit_down` | 70 | 66 | 37 |
| `sitting` | 407 | 388 | 353 |
| `stand_up` | 49 | 46 | 25 |
| `standing` | 112 | 112 | 80 |
| `other` | 608 | 583 | 514 |
| **Total** | **2 119** | **1 973** | **1 562** |

- T inclui 145 vídeos, dos quais 137 têm ao menos uma janela elegível; 58
  deles têm janela `fall`. Em S, 45 vídeos têm janela `fall`.
- T equivale a 38% das 5 219 janelas do treino Le2i-CS; S, a 30%.
- Janelas de T por sujeito: S1 208, S2 759, S3 568, S4 438. Janelas `fall`:
  S1 25, S2 101, S3 72, S4 27.

### Receita de contagem e congelamento (W2)

Entradas pinadas:

| Entrada | Identidade |
| --- | --- |
| Vídeos | GMDCSA24, commit `5abac7693229900cf80f722e878fbb119211fc1c` com `verify` OK; o commit fixa o conteúdo dos 160 MP4 |
| Rótulos | `simplexsigil2/omnifall`, config `labels`, revisão `68e5cee56a4bad38cca4aea791cac248f96e79a0`, linhas com `dataset == "GMDCSA24"` |
| Código | GateFall no commit `23b7007`, com `TARGET_FPS = 10`, `TRAIN_STRIDE = 4` e `WINDOW_FRAMES = 24` em `gatefall.config` |

A receita é determinística, sem RNG, e usa só funções existentes da auditoria
e do pipeline, nunca o adapter:

1. `build_inventory` lista os 160 vídeos; `probe_file` (`probe_video` e
   `resolve_frame_rate`) dá o número de quadros decodificados e o fps de cada
   um.
2. `K` é o tamanho da grade de `build_time_grid(quadros, fps, TARGET_FPS)`. O
   rótulo de cada quadro vem de `labels_for_grid` sobre os segmentos OmniFall,
   nos instantes `k / 10`.
3. Os fins de janela são `window_end_indices(K, TRAIN_STRIDE)`, e o rótulo da
   janela é o do quadro `k_end`.
4. Os vídeos excluídos são os que têm motivo em `anomaly_reasons`.
5. T: janelas de vídeo não excluído com rótulo diferente de `IGNORE_LABEL`,
   `k_end >= 23` e rótulo fora de `lie_down` e `lying`.
6. S: janelas de T a mais de 0,5 s, em `k_end / 10`, de toda fronteira de
   `video_boundaries`. Um vídeo sem fronteira mantém todas as janelas.

Congelamento e comparação com o adapter:

- **Agregados publicados (valem agora).** W2 compara o adapter com o que esta
  página publica: a lista dos 15 vídeos excluídos, as janelas por classe de T
  e de S, os 145 vídeos incluídos, os 137 com janela em T e os vídeos com
  janela `fall` (58 em T e 45 em S). Qualquer diferença bloqueia o treino.
- **Oráculo por vídeo (não exigido na validação inicial).** A análise local
  não está versionada, e a tabela por vídeo não é publicada. A
  [aprovação](#registro-da-aprovacao) escolheu a W2 só por agregados para a
  validação inicial do adapter, sem alegar verificação independente por
  vídeo. A alternativa fica documentada, mas não foi adotada: um PR próprio,
  antes da implementação do adapter, acrescentaria à auditoria
  (`gatefall.data.gmdcsa24.audit`) um comando que executaria a receita acima
  e gravaria uma tabela `video_id` × classe com as contagens de T e S,
  publicada com o SHA-256 dela e o commit do código.
- **Comparação por vídeo.** Só com o oráculo congelado antes do adapter: o
  adapter gravaria a mesma tabela no manifesto, e W2 exigiria igualdade exata
  de cada linha. Como o oráculo não foi adotado, W2 fica restrita aos
  agregados publicados. Uma tabela por vídeo criada depois do adapter não
  conta como verificação independente.
- **Alcance.** Se o adapter reutilizar `build_time_grid`, `labels_for_grid` e
  `window_end_indices`, W2 não confere essas funções de forma independente;
  elas já são cobertas pelos selftests do Le2i. W2 confere a montagem do
  adapter: seleção e exclusão de vídeos, contagem de quadros e filtros de
  janela.

O manifesto do adapter também grava a lista de vídeos excluídos, com motivo e
`source_sha256`, antes do treino.

## Ruído de rótulo não resolvido

T treina com ruído de rótulo que este protocolo não mede nem corrige:

- **Fronteiras.** A triagem sugere deslocamentos de alguns décimos de
  segundo, talvez sistemáticos: janelas `fall` perto do início podem mostrar a
  pessoa ainda sentada ou em pé, e janelas `fall` perto do fim podem mostrar a
  postura de `fallen`.
- **Vídeos mantidos com alerta.** Os 16 vídeos de estrato da amostra ficam em
  T, e 9 deles têm ao menos uma fronteira com possível divergência na
  triagem. Excluí-los usaria julgamento de IA, e só eles foram triados.
- **Vídeos sem revisão.** 129 dos 145 vídeos incluídos não passaram por
  nenhuma revisão visual.
- **Proveniência.** O OmniFall não publica a versão nem o checksum dos
  vídeos que anotou. A correspondência continua inferida por nome e duração.
- **Semântica.** Classes como `other`, `standing` e `walk` têm limites
  ambíguos mesmo com proveniência correta.

Nenhuma taxa de erro é estimada ou extrapolada. O ruído fica só no treino:
`val` e `test` do Le2i-CS não mudam, então as métricas continuam medindo o
modelo no Le2i. O que o ruído torna incerto é a atribuição do efeito.

## Análise de sensibilidade predefinida (S)

S é T sem as janelas cujo `k_end` (no instante `k_end / 10` s) fica a 0,5 s
ou menos de alguma fronteira OmniFall do vídeo. A fronteira é a mesma de
`video_boundaries` na revisão: início de segmento após 0 s ou fim seguido de
lacuna. Os rótulos não mudam; só saem janelas.

- **Como 0,5 s foi escolhido.** A margem foi escolhida depois da triagem por
  IA e informada por ela: é a meia largura das contact sheets em que a
  triagem julgou, e cobre os deslocamentos aparentes que ela descreve (até
  cerca de 0,4 s). A escolha veio antes de qualquer treino, extração de
  features ou resultado de avaliação de modelo. A margem fica fixa, e nenhuma
  outra será testada.
- **O que isso não muda.** Usar a triagem para fixar a margem não transforma
  julgamento de IA em rótulo verificado. S continua uma análise exploratória
  sobre rótulos fracos, e nenhuma fronteira passa a ser tratada como
  confirmada ou errada.
- **O que S muda além das fronteiras.** S remove as janelas perto de
  fronteira, mas também muda o tamanho do treino (−21% das janelas), o
  balanço de classes (−46% das janelas `fall`), os pesos da
  `CrossEntropyLoss` e as estatísticas do JSON de padronização, recalculados
  sobre o treino de S. Por isso S não isola o efeito do ruído de fronteira.
- S roda com as mesmas cinco seeds, a mesma receita e o mesmo controle de T.

## Critério de decisão

A decisão usa só `val`. As condições de validade valem para T e S. Qualquer
falha torna o piloto **INVÁLIDO**.

| # | Condição de validade |
| --- | --- |
| W1 | Este protocolo aprovado; lista de vídeos excluídos igual à tabela acima; `review.csv` sem nenhuma alteração feita por esta trilha e sem julgamento de IA nas colunas humanas |
| W2 | Saída do adapter igual aos agregados publicados em [Receita de contagem](#receita-de-contagem-e-congelamento-w2); só por agregados, pela opção escolhida na [aprovação](#registro-da-aprovacao), sem comparação por vídeo |
| V2–V5 | As do protocolo verificado, aplicadas a T e a S |

Com as condições cumpridas:

- **T contra o controle** decide pelas condições G1–G4 do protocolo
  verificado, com os mesmos limites. Se as quatro passam, o resultado é **GO
  exploratório**; caso contrário, **NO-GO**.
- **S contra o controle** tem G1–G4 calculadas e relatadas. Se T dá GO
  exploratório e a média de Δ `macro_f1_restricted` de S for <= 0, o resultado
  passa a **GO frágil**. S nunca transforma NO-GO em GO.

GO frágil é uma regra operacional, não uma explicação causal. Ele significa
que a melhora de T não se reproduziu sob a distribuição de treino modificada
de S. Não prova que o ruído das janelas perto de fronteira causou a melhora
de T: a falha pode vir de qualquer outra mudança de S (tamanho, balanço,
pesos da perda ou padronização). Do mesmo modo, um S positivo não prova que
as janelas de fronteira eram ruído.

O relatório mostra também a diferença de média e desvio por bloco de features
entre o JSON do controle e os de T e S, e as métricas descritivas de `test`,
calculadas uma vez depois da decisão.

## Critérios de parada

1. Nada é extraído nem treinado antes de cumpridos os passos de
   [Aprovação](#aprovacao), inclusive a autorização separada do passo 4 e os
   [portões](#portoes-antes-de-cada-etapa) dela.
2. Antes do treino, falha de `verify`, `inventory`, V5 ou W2 para a trilha. O
   código é corrigido sem mudar a política. Se a correção exigir mudar a
   política, a trilha para e precisa de um novo protocolo.
3. Cada seed roda uma vez em T e uma vez em S. Só uma falha técnica (queda do
   processo, erro de E/S) permite repetir a mesma seed, com a mesma
   configuração e registro do motivo. Métrica nunca justifica repetição.
4. Depois de qualquer métrica, nada muda: vídeos, janelas, margem, limites,
   seeds, épocas e classes.
5. `test` é calculado uma vez, depois da decisão, e não decide nada.
6. Um resultado INVÁLIDO permite uma única repetição depois da correção da
   causa. Um segundo INVÁLIDO encerra a trilha.
7. O primeiro resultado válido encerra a trilha, qualquer que seja. Outra
   execução com rótulo fraco exige um protocolo novo, com o que ele desloca.

## O que a evidência sustenta

Com GO exploratório, o piloto sustenta só isto: sob esta política e com os
rótulos OmniFall pinados e não verificados, somar as janelas do GMDCSA24 ao
treino Le2i-CS mudou as métricas do braço A na `val` do Le2i-CS em Δ,
pareado em cinco seeds, sem piorar perdas de eventos nem alarmes falsos sob o
protocolo de alarme congelado. Isso basta para decidir se a revisão humana e o
protocolo verificado merecem prioridade. Essa decisão continua do
responsável pelo projeto.

O piloto **não** sustenta, com nenhum resultado:

- que as anotações OmniFall do GMDCSA24 estão corretas, nem qualquer taxa de
  erro delas;
- que o ruído de rótulo não importa, nem que o ruído de fronteira explica o
  resultado de T (S muda tamanho, balanço, pesos e padronização junto);
- a condição V1, o pré-requisito 2 da viabilidade, a etapa 3 (B0, B1, C0 e
  C1), a promoção ao protocolo canônico ou a troca de alguma referência;
- qualquer desempenho no GMDCSA24, no Le2i-CV ou em `test` (só descritivo);
- que o efeito vem das quedas, e não da escala da padronização ou do volume;
- com NO-GO ou GO frágil, que o GMDCSA24 não serve: ruído de rótulo, diferença
  de resolução e taxa e a própria exclusão ficam confundidos, e o protocolo
  verificado continua sem resposta.

A `val` tem 19 vídeos e 13 quedas. Mesmo um GO exploratório é fraco por
construção.

## Adapter e validação

O adapter do passo 3 é `gatefall.data.gmdcsa24.adapter` (comandos em
[Referência de comandos](../reference/commands.md#orquestracao-e-preparacao)).
Ele aplica esta página sem mudar nenhuma regra: a política de exclusão vem de
`anomaly_reasons`, a grade de `build_time_grid` e `labels_for_grid`, e as
janelas de `build_window_index` em `TRAIN_STRIDE`. Ele não extrai features,
não ajusta padronização, não calcula pesos da perda e não treina.

### Saída

Tudo fica em `data/processed/gmdcsa24/`, ignorado pelo Git. O adapter recusa
gravar sobre `data/processed/le2i*/`, `data/features/`, `data/labels/`,
`runs/`, as estatísticas de padronização e o diretório da revisão de
proveniência.

| Arquivo | Conteúdo |
| --- | --- |
| `videos.parquet` | Os 160 vídeos: `source_sha256`, fps, quadros por `ffprobe -count_frames` e pelo decodificador do adapter, desvio de PTS, exclusão e motivo |
| `frames.parquet` | Grade de 10 fps dos 145 vídeos incluídos, no esquema de quadros do Le2i, com `split = train` |
| `window_candidates.parquet` | As janelas rotuladas em `TRAIN_STRIDE` dos 160 vídeos, com cada filtro de T e S |
| `windows_t.parquet`, `windows_s.parquet` | Índices de janela de T e de S, no esquema do Le2i |
| `manifest.json` | Proveniência (protocolo, clone, OmniFall, commit do código, Le2i-CS), parâmetros, exclusões com motivo e `source_sha256`, agregados e hash de conteúdo de cada tabela |

O decodificador do adapter usa `ffmpeg -fps_mode passthrough`: o quadro `i`
do pipe é o `i`-ésimo quadro decodificado, sem as duplicações do pipe do Le2i
nos MP4 de taxa variável. O `src_index` da grade aponta para esse quadro.

### Checagens

`validate` recalcula tudo das tabelas gravadas, nunca reescreve o manifesto e
só passa com as seis condições:

- **Integridade:** hash de conteúdo e linhas de cada tabela iguais ao
  manifesto; T e S iguais aos filtros de `window_candidates.parquet`; resumo
  do manifesto igual ao recalculado; grade gravada igual à de cada vídeo
  incluído.
- **Proveniência:** cada seção do manifesto conferida com a identidade
  esperada: caminho, marca de aprovação e SHA-256 do protocolo; repositório,
  tag e commit pinados do clone, com `verify` e `inventory` OK; dataset,
  config e revisão pinados do OmniFall; parâmetros fixos do adapter (fps,
  janela, stride, margem de 0,5 s, classes descartadas, split e
  decodificador); e cada registro de `excluded_videos` (`video_id`, motivos e
  `source_sha256`) igual ao de `videos.parquet`.
- **W1:** marca de aprovação nesta página; os 15 vídeos excluídos e os motivos
  iguais à [tabela](#videos); `review.csv` com o mesmo SHA-256 antes e depois
  do build.
- **W2:** igualdade exata com todos os agregados publicados em
  [Contagem esperada](#contagem-esperada): lista de exclusão, funil, janelas
  por classe de T e S, vídeos incluídos, com janela em T e com janela `fall`
  em T e S, e janelas de T por sujeito.
- **V2:** GMDCSA24 só em `split = train` e só com vídeos incluídos; nenhum
  `video_id` do GMDCSA24 em `val`/`test` do Le2i-CS e nenhum `source_sha256`
  igual a um vídeo do manifesto Le2i; treino Le2i-CS com as 5 219 janelas
  publicadas; hash dos índices de `val` e `test` do Le2i-CS igual ao gravado
  no build.
- **V5:** para os 160 vídeos, o decodificador do adapter entrega o mesmo
  número de quadros que `ffprobe -count_frames`.

O selftest sintético (`adapter selftest`, na CI) cobre alinhamento temporal,
exclusões, a margem de 0,5 s de S, decodificação sem quadros duplicados,
adulteração das tabelas e do manifesto, divergência de um único agregado e
vazamento para `val`/`test`. Ele também adultera, um de cada vez, a margem de
fronteira, o commit do clone, a revisão do OmniFall, o SHA-256 do protocolo, o
motivo e o `source_sha256` de um vídeo excluído. Cada adulteração reprova a
proveniência, e a saída intacta continua válida.

A proveniência não revalida tudo contra a fonte. `validate` relê só o
documento do protocolo, as tabelas gravadas e o Le2i-CS local. O commit do
clone, a revisão e o hash dos segmentos do OmniFall e o `source_sha256` dos
MP4 são os valores registrados no build. Eles são conferidos contra a
identidade pinada no código e contra `videos.parquet`, sem reler o clone, o
OmniFall ou os vídeos. Revalidar essas fontes exige um novo `build`, que
refaz `verify`, `inventory`, a sondagem e os hashes. `validate` imprime as
duas listas separadas.

### Resultado sobre os dados pinados

Executado em 2026-10-19 sobre o clone `v2.1` (`verify` e `inventory` OK) e o
OmniFall `labels` na revisão `68e5cee`, com o Le2i-CS canônico local:

- integridade, proveniência, W1, W2, V2 e V5 passam;
- as contagens coincidem com todos os agregados publicados, sem nenhuma
  diferença: 3 231 janelas rotuladas, −945, −142, −171, **T = 1 973**, −411,
  **S = 1 562**, as contagens por classe e por sujeito, 145 vídeos incluídos,
  137 com janela em T, 58 com janela `fall` em T e 45 em S;
- os 15 vídeos excluídos e os motivos são os da [tabela](#videos);
- V5: nos 160 vídeos, o decodificador do adapter e `ffprobe -count_frames`
  contam os mesmos quadros;
- `review.csv` não mudou durante o build;
- hash de conteúdo das tabelas: `windows_t.parquet`
  `dd904e1d21af41f36738157fb6bf61cc5792ffa8c6477f1e4dc8e86efff926a8`,
  `windows_s.parquet`
  `bf9371715a152d7cbf795155f31a2c28ffef5f98410649aee5ed17c13862efcb`,
  `window_candidates.parquet`
  `4493cafd8772cc0df1435b9d188c5d6e67ffed26c0bcb4177c83ffeeedbd07c6`;
- hash dos índices do Le2i-CS gravado no manifesto: `val`
  `56f40fe5e32ac6558bcef7e157178ad81dd0fd497da9f53c91a6cffcce90422b`,
  `test`
  `36168466d34f6bea11d601b60cbcc3d82d830baeb0b27d209a1cd1b4c617c984`.

W2 confere a montagem do adapter contra os agregados, não as funções
compartilhadas com o Le2i nem uma tabela por vídeo, como decidido na
[aprovação](#registro-da-aprovacao).

### O que fica para depois do passo 4

O adapter entrega só o lado GMDCSA24 do treino de T e S. Continuam fora do
adapter, cada um com implementação própria e só nas condições de
[Autorização do passo 4](#autorizacao-do-passo-4):

- a extração de pose dos 145 vídeos incluídos, com o decodificador do adapter;
- a junção das janelas de T ou S com o treino Le2i-CS no carregador de
  treino, o JSON de padronização e os pesos da perda de cada lado;
- as partes de V2 que dependem do treino: estatísticas do controle sem
  GMDCSA24, alarme não recalibrado e índices de `val`/`test` com o hash acima
  nos dois lados.

## Aprovação

O protocolo passa a valer só quando estes passos estiverem cumpridos, nesta
ordem:

1. **Cumprido em 2026-10-19.** Um PR separado muda o status deste documento
   para aprovado, com a decisão explícita do responsável pelo projeto sobre: a
   aceitação da proveniência não verificada só para esta trilha, a política de
   exclusão e as contagens, a análise S com a margem de 0,5 s escolhida depois
   da triagem, os critérios de parada e os limites das alegações (ver
   [Registro da aprovação](#registro-da-aprovacao)).
2. **Não adotado.** Se a comparação por vídeo de W2 for exigida, o oráculo de
   contagem é implementado e congelado em PR próprio, antes do adapter (ver
   [Receita de contagem](#receita-de-contagem-e-congelamento-w2)). A
   aprovação escolheu a W2 só por agregados, então este passo não se aplica à
   validação inicial do adapter.
3. **Cumprido em 2026-10-19.** O adapter isolado (`data/processed/gmdcsa24/`), o manifesto
   com a lista de exclusão, a montagem dos treinos de T e S e as checagens W1,
   W2, V2 e V5 foram implementados e revisados no PR #134, com selftests
   sintéticos, sem alterar os selftests e as referências do Le2i. O adapter
   decodifica sem duplicar quadros. A implementação e o resultado sobre os
   dados pinados estão em [Adapter e validação](#adapter-e-validacao). O PR
   #134 foi integrado em `experiment/external-fall-data` com o head
   `4ef87d1` (merge `fc7f4de`).
4. **Proposto no PR #135; vale só a partir da integração dele.** A revisão e
   a integração desse PR pelo responsável pelo projeto são a autorização
   explícita da execução local e isolada descrita em
   [Autorização do passo 4](#autorizacao-do-passo-4). Antes dessa integração,
   nada do passo 4 vale. A autorização não cobre sincronização com o Google
   Drive, que continua proibida.

### Registro da aprovação

Em 2026-10-19, o responsável pelo projeto aprovou este protocolo e decidiu
seguir **sem revisão visual manual das anotações** para esta trilha
exploratória separada. A decisão cobre os itens exigidos no passo 1:

- **Proveniência.** A proveniência não verificada das anotações OmniFall do
  GMDCSA24 é aceita só para esta trilha, como rótulos fracos. As anotações
  continuam **não verificadas**: a aprovação não confirma nenhuma fronteira,
  não estima taxa de erro e não cumpre o pré-requisito 2 da viabilidade.
- **Protocolo verificado.** O [protocolo verificado](gmdcsa24-pilot-protocol.md)
  continua PROPOSTO e sem mudança, com a condição V1 pendente. A
  [revisão de proveniência](gmdcsa24-provenance-review.md) continua PENDENTE.
- **Triagem por IA.** A triagem visual assistida por IA continua evidência
  preliminar. Ela **nunca** é tratada como confirmação humana, nunca entra nas
  colunas humanas de `review.csv` e nunca muda o estado de um caso.
- **Escopo congelado, sem mudança em relação ao PR #132:**
    - só o braço A, sem mudança de arquitetura e com a receita congelada;
    - as 15 exclusões de vídeo predeclaradas em
      [Vídeos](#videos);
    - os tratamentos de treino T e S, com 1 973 e 1 562 janelas estimadas do
      GMDCSA24 e as contagens por classe e por sujeito de
      [Contagem esperada](#contagem-esperada);
    - S com a margem de 0,5 s, escolhida depois da triagem por IA e fixa;
    - seeds 42 a 46, uma execução por seed em T e em S, contra o mesmo
      controle;
    - `val` e `test` do Le2i-CS sem mudança e as referências canônicas
      intactas;
    - normalização ajustada só sobre o treino de cada lado;
    - as condições G1–G4, W1, W2 e V2–V5, o critério de decisão e os
      [critérios de parada](#criterios-de-parada) desta página;
    - os limites de [O que a evidência sustenta](#o-que-a-evidencia-sustenta).
- **W2 só por agregados.** A validação inicial do adapter usa a opção já
  documentada de W2 só por agregados: a lista dos 15 vídeos excluídos, as
  janelas por classe de T e de S, os 145 vídeos incluídos, os 137 com janela
  em T e os vídeos com janela `fall` (58 em T e 45 em S). Ela não alega
  verificação independente por vídeo, e o oráculo por vídeo não é criado.
- **Divergência bloqueia.** Qualquer diferença entre o adapter e os agregados
  predeclarados bloqueia o treino e exige investigação. Nem os agregados, nem
  a lista de exclusão, nem as regras de janela são ajustados depois do fato
  para coincidir. Se a causa estiver no código, ele é corrigido sem mudar a
  política. Se a correção exigir mudar a política ou os agregados publicados,
  a trilha para e precisa de um novo protocolo aprovado
  ([critério de parada 2](#criterios-de-parada)).

A aprovação do passo 1 autorizou só o planejamento da implementação e o PR
separado do adapter (passo 3). Ela não autorizou extração de features nem
treino de modelos.

### Autorização do passo 4

O PR #135 propõe esta autorização. Ela vale só a partir da revisão e da
integração desse PR em `experiment/external-fall-data` pelo responsável pelo
projeto; antes disso, nada desta seção está autorizado, e nenhuma autorização
anterior é alegada.

#### Evidência de execução informada pelo responsável

O responsável pelo projeto informou os logs abaixo, executados no checkout
local de `experiment/external-fall-data` depois da integração do PR #134. O
manifesto gravado registra o commit `fc7f4de` e `code.dirty = true`. Os logs
não estão versionados e não foram reexecutados neste PR.

- `adapter validate` sobre a saída existente: integridade, proveniência, W1,
  W2, V2 e V5 passam.
- `adapter build --force`, seguido de `adapter validate`:
    - clone na tag `v2.1` (commit `5abac7693229900cf80f722e878fbb119211fc1c`),
      sem arquivo modificado ou não rastreado; 160 MP4, sem duplicata por
      SHA-256; os mesmos 160 vídeos no clone, no CSV original e nas configs
      `labels`, `cs` e `cv` do OmniFall;
    - os 15 vídeos excluídos são os da [tabela](#videos);
    - funil 3 231, −945, −142, −171, **T = 1 973**, −411, **S = 1 562**;
      contagens por classe de T e S, 145 vídeos incluídos, 137 com janela em
      T, 58 e 45 com janela `fall` em T e S, e janelas de T e janelas `fall`
      por sujeito, todos iguais a [Contagem esperada](#contagem-esperada);
    - integridade, proveniência, W1, W2, V2 e V5 passam no build e no
      `validate` isolado.

O próprio `validate` separa o que conferiu contra a fonte atual (documento do
protocolo, tabelas gravadas e Le2i-CS local) do que só compara com o valor
registrado no build (clone, OmniFall, parâmetros, exclusões e commit do
código). O `build --force` refez `verify`, `inventory`, a sondagem e os hashes
dos MP4. Essa evidência cumpre a pré-condição do passo 3 e não substitui o
[portão E0](#portoes-antes-de-cada-etapa).

#### O que passa a ser autorizado

Com a integração do PR #135, e só depois dos portões de cada etapa, fica
autorizada a execução **local e isolada** de:

1. **Extração offline YOLO-Pose dos 145 vídeos elegíveis.** São os vídeos
   incluídos de `videos.parquet`, decodificados pelo decodificador do adapter
   e alinhados ao `src_index` de `frames.parquet`, com o backbone congelado e
   a mesma configuração de pose do braço A no Le2i. Os 15 vídeos excluídos
   não são extraídos. A saída fica em caminho próprio do GMDCSA24, ignorado
   pelo Git, fora de `data/features/le2i/` e dos outros artefatos canônicos.
2. **Piloto T/S só do braço A.** Cinco treinos de T e cinco de S, nas seeds 42
   a 46, uma execução por seed, com a receita congelada e o mesmo controle
   pareado (aceito pelo `checkpoint_sha256` ou retreinado na mesma máquina,
   condição V3). JSON de padronização e pesos da `CrossEntropyLoss` são
   ajustados só sobre o treino de cada lado. A decisão usa só `val` do
   Le2i-CS, pelo [critério de decisão](#criterio-de-decisao); `test` é
   calculado uma vez, depois da decisão, só como descrição.

Nada mais é autorizado. As decisões fixadas nesta página continuam valendo
sem mudança:

- as fontes pinadas, as 15 exclusões, as definições de T e S e a W2 só por
  agregados;
- a margem fixa de 0,5 s de S;
- as seeds 42 a 46 e a receita congelada;
- a padronização só com treino e o controle pareado;
- o isolamento de `val` e `test` do Le2i-CS;
- as condições G1–G4, W1, W2 e V2–V5;
- os [critérios de parada](#criterios-de-parada).

A [revisão de proveniência](gmdcsa24-provenance-review.md) continua
PENDENTE, e a condição V1 do [protocolo verificado](gmdcsa24-pilot-protocol.md)
continua não cumprida.

#### Portões antes de cada etapa

Cada etapa só começa depois dos portões dela. Uma falha bloqueia a etapa e
segue o [critério de parada 2](#criterios-de-parada): o código é corrigido sem
mudar a política, ou a trilha para.

- **E0 — saída do adapter sob o texto integrado.** O PR #135 muda este
  documento, então o SHA-256 do protocolo registrado no manifesto atual deixa
  de bater, e `validate` reprova a proveniência até um novo build. Antes da
  extração, `adapter build --force` e `adapter validate` rodam a partir do
  commit integrado. As seis condições precisam passar, e os agregados precisam
  ser os mesmos. Os hashes de conteúdo de `windows_t.parquet`,
  `windows_s.parquet` e `window_candidates.parquet` também precisam ser iguais
  aos de [Resultado sobre os dados pinados](#resultado-sobre-os-dados-pinados).
- **Extração.** A implementação é revisada e integrada em PR próprio, sem
  aprovação ou mudança de protocolo no mesmo PR, e com selftest sintético. A
  validação dela confere pelo menos estes pontos:
    - cobertura exata dos 145 vídeos incluídos, sem nenhum excluído;
    - alinhamento com a grade de `frames.parquet`;
    - isolamento dos caminhos do Le2i e backbone congelado;
    - nenhuma métrica de modelo calculada.
- **Treino.** A implementação da junção com o treino Le2i-CS, da padronização
  e dos pesos de cada lado é revisada e integrada em PR próprio, com selftest
  sintético. Antes do primeiro treino passam, para T e para S:
    - W1, W2 e V2–V5, inclusive as partes de V2 que dependem do treino:
      estatísticas do controle sem GMDCSA24, alarme não recalibrado e índices
      de `val`/`test` com o hash acima nos dois lados;
    - configuração idêntica ao controle fora dos três itens permitidos (V4).

### O que continua proibido

Sem a integração do PR #135, continua proibido tudo o que a autorização do
passo 4 cobriria. Depois dela, a extração, a padronização, o treino e as
métricas ficam bloqueados até a validação da implementação correspondente:

- extrair features do GMDCSA24 antes de a implementação da extração ser
  integrada e de o [portão E0](#portoes-antes-de-cada-etapa) passar;
- ajustar os JSON de padronização de T e S, montar os treinos de T e S com
  features ou treinar T, S ou um novo controle antes de a implementação do
  treino ser integrada e de os portões dela passarem;
- calcular qualquer métrica de modelo com GMDCSA24 no treino, em `val` ou em
  `test` antes desses portões.

Continuam proibidos mesmo com a autorização do passo 4:

- extrair features de vídeos fora dos 145 elegíveis ou com qualquer backbone
  além do YOLO-Pose, inclusive DINOv3 e SAM 3;
- treinar ou ajustar o YOLO-Pose;
- avaliar modelos no GMDCSA24 ou no Le2i-CV;
- sincronizar dados, features, manifestos ou checkpoints do GMDCSA24 com o
  Google Drive;
- implementar a extração ou o treino no mesmo PR de uma aprovação ou de uma
  mudança de protocolo;
- alterar o [protocolo verificado](gmdcsa24-pilot-protocol.md), `review.csv`,
  `review-status` ou as colunas humanas da revisão, inclusive com a triagem
  por IA;
- alterar os experimentos ou o modelo canônico do Le2i: receita congelada,
  splits Le2i-CS e Le2i-CV, checkpoints, calibração, protocolo de alarme,
  selftests e referências em `runs/reference/`;
- treinar B0, B1, C0 ou C1 com GMDCSA24, usar outro dataset externo ou testar
  outra variante de rótulo fraco (outra margem, outra lista de exclusão,
  reetiquetagem ou deslocamento de rótulo).

## Riscos em aberto

- **Ruído sistemático nas fronteiras.** Se os inícios de `fall` estiverem
  adiantados, o modelo aprende `fall` em quadros sem queda visível, e a
  latência e os alarmes falsos podem piorar no Le2i.
- **Viés de seleção.** A exclusão retira os vídeos com queda anotada muito
  antes do autor e os de taxa mais irregular. O que fica não representa o
  GMDCSA24 inteiro.
- **Equilíbrio de sujeitos.** S1 perde sete vídeos e fica com 25 janelas
  `fall`; S2 tem 38% das janelas de T.
- **Riscos do protocolo verificado.** Decodificação de taxa variável,
  diferença de resolução e taxa, poucos sujeitos, `val` pequena, licença e
  consentimento valem aqui sem mudança
  ([riscos](gmdcsa24-pilot-protocol.md#riscos-em-aberto)).
