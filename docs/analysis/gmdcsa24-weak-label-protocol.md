# Protocolo proposto — piloto exploratório GMDCSA24 com rótulos fracos (braço A)

!!! warning "Status: PROPOSTO, não aprovado"
    Este documento propõe um experimento exploratório separado. Ele não
    autoriza treino, extração de features nem implementação de adapter. O
    [protocolo verificado](gmdcsa24-pilot-protocol.md) continua PROPOSTO e
    intacto, e a condição V1 dele continua pendente: nada nesta página a dá por
    cumprida. O protocolo canônico do GateFall, a receita congelada, os splits
    Le2i-CS e Le2i-CV, os checkpoints, o protocolo de alarme e as referências
    A, B0, B1, C0 e C1 não mudam. A aprovação exige os passos de
    [Aprovação](#aprovacao) e, até lá, a resposta padrão é não.

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
concentrado perto das transições. É isso que a
[análise de sensibilidade](#analise-de-sensibilidade-predefinida-s) testa.

Regras para essa evidência:

- **Julgamento de IA nunca vira registro de revisor humano.** Nenhuma coluna
  humana de `review.csv` (`review_outcome`, `observed_boundary_s`,
  `reviewer`, `reviewed_at`, `notes`) é preenchida ou pré-preenchida com a
  triagem, e nenhum caso conta como `confirmed`, `discrepancy` ou `excluded`
  por causa dela.
- A triagem não entra na política de exclusão, nas condições de validade, no
  critério de decisão nem na escolha de qualquer parâmetro.
- O relatório fica só na máquina local, em
  `data/scratch/gmdcsa24/provenance/gmdcsa24_relatorio_triagem_visual_ia.md`
  (ignorado pelo Git), com SHA-256
  `c3de33da6daaf5e3b3462534cdca0a8a4661ebecaaab2c9e2ac67f86463dbdd9`.

## Política de inclusão e exclusão

Os rótulos OmniFall pinados entram como estão: **sem reetiquetar, deslocar
instantes, cortar ou mesclar segmentos**. Onde há anomalia documentada, sai o
vídeo inteiro; nada é corrigido. Toda regra é recalculada pelo código
existente a partir do clone pinado e do OmniFall pinado. Nenhuma usa a
triagem por IA, métrica de modelo ou julgamento caso a caso.

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

Os 15 vídeos excluídos, recalculados em 2026-10-19:

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

Calculada em 2026-10-19 sobre o clone pinado, com a grade de taxa constante
do `probe` e as regras acima. A mesma conta reproduz a estimativa do
protocolo verificado (2 119 janelas em 149 vídeos).

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

A lista de vídeos excluídos, com motivo e `source_sha256`, e a contagem por
classe e por vídeo de T e S são gravadas no manifesto do adapter antes do
treino e precisam bater com esta página (condição W2).

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

- **Por que 0,5 s.** É a meia largura das contact sheets em que a triagem
  julgou, e os deslocamentos aparentes que ela descreve (até cerca de 0,4 s)
  cabem nessa margem. A margem é fixada aqui, antes de qualquer métrica, e
  nenhuma outra será testada.
- **O que S não isola.** S remove 46% das janelas `fall` e 21% do total.
  Volume e balanço de classes também mudam, e os pesos da perda e o JSON de
  padronização de S são recalculados sobre o treino de S. S qualifica T; não
  é um teste limpo de ruído.
- S roda com as mesmas cinco seeds, a mesma receita e o mesmo controle de T.

## Critério de decisão

A decisão usa só `val`. As condições de validade valem para T e S. Qualquer
falha torna o piloto **INVÁLIDO**.

| # | Condição de validade |
| --- | --- |
| W1 | Este protocolo aprovado; lista de vídeos excluídos igual à tabela acima; `review.csv` sem nenhuma alteração feita por esta trilha e sem julgamento de IA nas colunas humanas |
| W2 | Contagem de janelas por classe e por vídeo do adapter igual à de [Contagem esperada](#contagem-esperada) |
| V2–V5 | As do protocolo verificado, aplicadas a T e a S |

Com as condições cumpridas:

- **T contra o controle** decide pelas condições G1–G4 do protocolo
  verificado, com os mesmos limites. Se as quatro passam, o resultado é **GO
  exploratório**; caso contrário, **NO-GO**.
- **S contra o controle** tem G1–G4 calculadas e relatadas. Se T dá GO
  exploratório e a média de Δ `macro_f1_restricted` de S for <= 0, o resultado
  passa a **GO frágil**: o ganho depende das janelas perto de fronteira, que
  são as mais expostas ao ruído. S nunca transforma NO-GO em GO.

O relatório mostra também a diferença de média e desvio por bloco de features
entre o JSON do controle e os de T e S, e as métricas descritivas de `test`,
calculadas uma vez depois da decisão.

## Critérios de parada

1. Nada é extraído nem treinado antes da [aprovação](#aprovacao).
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
- que o ruído de rótulo não importa (S muda volume e balanço junto);
- a condição V1, o pré-requisito 2 da viabilidade, a etapa 3 (B0, B1, C0 e
  C1), a promoção ao protocolo canônico ou a troca de alguma referência;
- qualquer desempenho no GMDCSA24, no Le2i-CV ou em `test` (só descritivo);
- que o efeito vem das quedas, e não da escala da padronização ou do volume;
- com NO-GO ou GO frágil, que o GMDCSA24 não serve: ruído de rótulo, diferença
  de resolução e taxa e a própria exclusão ficam confundidos, e o protocolo
  verificado continua sem resposta.

A `val` tem 19 vídeos e 13 quedas. Mesmo um GO exploratório é fraco por
construção.

## Aprovação

O protocolo passa a valer só quando estes passos estiverem cumpridos, nesta
ordem:

1. Um PR separado muda o status deste documento para aprovado, com a decisão
   explícita do responsável pelo projeto sobre: a aceitação da proveniência
   não verificada só para esta trilha, a política de exclusão e as contagens,
   a análise S com a margem de 0,5 s, os critérios de parada e os limites das
   alegações.
2. O adapter isolado (`data/processed/gmdcsa24/`), o manifesto com a lista de
   exclusão, a montagem dos treinos de T e S e as checagens W1, W2, V2 e V5
   são implementados e revisados em PR próprio, com selftests sintéticos, sem
   alterar os selftests e as referências do Le2i. O adapter decodifica sem
   duplicar quadros.
3. A extração, os treinos e qualquer sincronização com o Google Drive são
   autorizados à parte, depois dos passos 1 e 2.

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
