# Rascunho do manuscrito — Resultados e Discussão

!!! warning "Rascunho de escrita, não é fonte de evidência"
    Esta página prepara as seções de Resultados e Discussão do TCC. Ela não
    introduz medição nova: não houve treino, avaliação, seleção, ajuste nem
    promoção de artefato para escrevê-la. Todo número vem de uma página já
    publicada nesta documentação, de `runs/reference/le2i/` ou de um relatório
    local validado, e o identificador `[E..]` de cada afirmação aponta para o
    [mapa evidência → fonte](#mapa-evidencia-fonte). Em caso de divergência, a
    fonte prevalece sobre este rascunho.

## Regras de leitura

**Braços.** Os cinco braços congelados seguem a mesma receita de treino
(janela, splits, otimização, perda, 30 épocas e checkpoint da última época),
a mesma configuração de TCN causal dilatada como backbone temporal e o mesmo
protocolo de alarme. Eles diferem nas fontes de entrada, nas projeções e no
mecanismo de fusão: A entrega a pose padronizada diretamente à TCN; B0, B1,
C0 e C1 projetam cada fonte para 128 canais (`Linear → LayerNorm → ReLU`) e
combinam as projeções por concatenação (B0, C0) ou por gate escalar seguido
de concatenação (B1, C1), entregando 256 canais à TCN. Cada seed é treinada
sob essa mesma receita `[E1]`.

| Braço | Entrada por timestep | Fusão |
| --- | --- | --- |
| A | Pose (134-d) | Sem projeção nem fusão; a pose entra direto na TCN |
| B0 | Pose + DINOv3 (1536-d) | Concatenação das projeções de 128-d |
| B1 | Pose + DINOv3 + `[q_pose, q_visual]` | Gate escalar `g_t = sigmoid(Linear([q_pose, q_visual]))`; `g_t` pondera a pose e `1 − g_t`, o visual |
| C0 | Pose + descritor SAM 3 `V_t` (10-d) | Concatenação das projeções de 128-d |
| C1 | Pose + `V_t` + `[q_pose, q_sam3]` | Gate escalar análogo ao de B1 |

`q_pose`, `q_visual` e `q_sam3` são proxies operacionais de qualidade, não
confianças calibradas `[E1]`.

**Protocolos.** Le2i-CS é a comparação principal: divide por sujeito, e o
treino contém os seis ambientes do Le2i. Le2i-CV é uma análise separada de
generalização entre ambientes dentro do Le2i: divide por ambiente e câmera,
mas os sujeitos se sobrepõem entre splits. Os dois protocolos respondem
perguntas diferentes e nunca são agregados nem comparados como ablação
`[E2]`.

**Níveis de evidência.** O texto final deve manter os quatro níveis
separados:

1. **Agregados de cinco seeds (42–46).** Treinos independentes sob a mesma
   receita, resumidos por média, desvio-padrão amostral, mínimo e máximo. O
   desvio entre seeds descreve inicialização e otimização; não é intervalo de
   confiança nem incerteza entre sujeitos `[E3]`.
2. **Exemplos canônicos da seed 42.** Somente Le2i-CS tem referência
   versionada, em `runs/reference/le2i/`. A seed 42 foi fixada antes dos
   resultados e não foi escolhida pelo teste. Não existe referência canônica
   de Le2i-CV `[E4]`.
3. **Diagnósticos e investigações exploratórios.** Consolidação de erros,
   dobras internas, auditoria, neutralização visual e Etapas 3A–3C e 4A–4D
   são não canônicos e não alteram nenhuma decisão congelada `[E5]` `[E27]`
   `[E28]` `[E29]` `[E34]`.
4. **Experimentos externos.** Os experimentos de física e movimento de outro
   repositório ficam fora dos resultados do GateFall
   ([seção própria](#experimentos-externos-de-fisica-e-movimento)).

O split `test` é descritivo em todo o trabalho. Nenhum braço, seed, época,
limiar ou protocolo foi escolhido por ele `[E3]`.

## Resultados

### Le2i-CS: classificação

Macro-F1 restrita às oito classes de `RESTRICTED_CLASSES`, média ±
desvio-padrão amostral entre cinco seeds `[E6]`:

| Braço | Treino | Validação | Teste | Teste (mín–máx) |
| --- | --- | --- | --- | --- |
| A | 0,8669 ± 0,0056 | 0,6484 ± 0,0136 | 0,6109 ± 0,0137 | 0,5919–0,6251 |
| B0 | 0,9972 ± 0,0002 | 0,6802 ± 0,0203 | 0,6558 ± 0,0143 | 0,6348–0,6751 |
| B1 | 0,9940 ± 0,0007 | 0,6885 ± 0,0182 | 0,6722 ± 0,0089 | 0,6607–0,6844 |
| C0 | 0,9856 ± 0,0017 | 0,6617 ± 0,0140 | 0,6391 ± 0,0071 | 0,6314–0,6461 |
| C1 | 0,9779 ± 0,0027 | 0,6813 ± 0,0094 | 0,6254 ± 0,0179 | 0,6031–0,6448 |

- Os quatro braços de fusão têm média de teste maior que A. As faixas de
  B0, B1 e C0 ficam inteiramente acima da faixa de A; a de C1 se sobrepõe a
  ela `[E6]`.
- B1 tem média maior que B0 na validação e no teste, com faixas de teste
  sobrepostas. C1 tem média maior que C0 na validação e menor no teste: a
  comparação C0/C1 é mista entre splits `[E6]`.
- Os braços de fusão passam de 0,97 no treino, contra 0,87 em A. A lacuna
  treino − teste é 0,2561 em A e de 0,3217 a 0,3524 nos braços de fusão
  `[E7]`.
- `standing` e `other` são as duas maiores parcelas do déficit da Macro-F1 em
  teste nos cinco braços (49% a 50% juntas). `standing` tem 43 janelas de
  teste e F1 médio entre 0,02 (C1) e 0,14 (B1). A confusão dominante em teste
  é `other→walk`, com 36% a 41% do suporte de `other` `[E7]`.

### Le2i-CS: alarmes por evento

Protocolo congelado `N=3`, `R=5,0 s` `[E8]`. A validação tem 13 eventos de um
único sujeito; o teste tem 22 eventos de dois sujeitos `[E9]`. Média ±
desvio-padrão entre cinco seeds, teste `[E6]`:

| Braço | Sensibilidade | Sensibilidade `fall` | FA/h (mín–máx) | Latência média (s) |
| --- | --- | --- | --- | --- |
| A | 0,9182 ± 0,0593 | 0,8909 ± 0,0610 | 87,55 ± 21,44 (58,37–110,89) | 0,500 ± 0,000 |
| B0 | 0,9455 ± 0,0498 | 0,9455 ± 0,0498 | 46,69 ± 8,25 (40,86–58,37) | 0,320 ± 0,045 |
| B1 | 0,9636 ± 0,0203 | 0,9636 ± 0,0203 | 43,19 ± 8,85 (35,02–52,53) | 0,300 ± 0,000 |
| C0 | 0,9727 ± 0,0407 | 0,9636 ± 0,0380 | 35,02 ± 11,67 (23,35–52,53) | 0,380 ± 0,045 |
| C1 | 0,9545 ± 0,0455 | 0,9545 ± 0,0455 | 39,69 ± 10,44 (29,18–52,53) | 0,420 ± 0,045 |

- **Validação saturada.** B0, B1, C0 e C1 têm sensibilidade 1,0 e 0 FA/h nas
  cinco seeds; A tem sensibilidade 0,9846 ± 0,0344 e 20,77 ± 22,57 FA/h. Esse
  split não discrimina os braços de fusão `[E6]` `[E10]`.
- **Falsos alarmes.** No teste, a média de FA/h de todos os braços de fusão
  fica abaixo da de A. O máximo de B1, C0 e C1 fica abaixo do mínimo de A; o
  máximo de B0 coincide com o mínimo de A `[E6]`.
- **Sensibilidade e latência.** As médias de sensibilidade dos braços de
  fusão ficam acima da de A, com faixas sobrepostas. Com 22 eventos, um
  evento perdido muda a sensibilidade em cerca de 0,045 `[E6]` `[E9]`. As
  latências médias ficam entre 0,30 s e 0,50 s.
- **Escala de FA/h.** O teste tem 0,171 h de vídeo; na referência de A, 10
  falsos alarmes correspondem a 58,4 FA/h `[E11]`. Os valores de FA/h são
  taxas extrapoladas de poucos minutos de vídeo.
- **Dependência do protocolo.** Na varredura pós-hoc de A (seed 42), os
  falsos alarmes de teste vão de 6 a 75 e a sensibilidade de teste, de 54,5% a
  100%, entre as 25 combinações de `N` e `R`. A varredura não selecionou
  protocolo; ela mostra que as métricas por evento dependem do protocolo
  congelado `[E12]`.

### Le2i-CS: exemplos canônicos da seed 42

A referência de cada braço é a seed 42 de `runs/reference/le2i/`, que
coincide com a seed 42 do agregado `[E4]` `[E13]`:

| Braço | Macro-F1 val | Macro-F1 teste | Detectados (teste) | Falsos alarmes (teste) | FA/h (teste) | Latência média / mediana (teste, s) |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0,6584 | 0,6231 | 22/22 | 10 | 58,37 | 0,5 / 0,3 |
| B0 | 0,7034 | 0,6552 | 21/22 | 9 | 52,53 | 0,4 / 0,3 |
| B1 | 0,7047 | 0,6714 | 21/22 | 9 | 52,53 | 0,3 / 0,3 |
| C0 | 0,6505 | 0,6422 | 22/22 | 4 | 23,35 | 0,4 / 0,3 |
| C1 | 0,6755 | 0,6115 | 20/22 | 8 | 46,69 | 0,5 / 0,3 |

Use esta tabela para exemplos e figuras qualitativas, nunca como substituta
do agregado. Com uma seed, a ordem entre braços muda: C0 tem o menor FA/h da
seed 42, mas no agregado sua faixa se sobrepõe às de B1 e C1 `[E6]` `[E13]`.

O bootstrap agrupado por sujeito sobre a seed 42 tem resolução muito
limitada: a validação tem um sujeito, e o intervalo colapsa no ponto; o teste
tem dois sujeitos, e cada réplica é só o sujeito 2, só o sujeito 7 ou os dois
`[E14]`. Os intervalos são descritivos, não testes de hipótese.

### Le2i-CV: generalização entre ambientes

O treino tem `Coffee_room_01`, `Coffee_room_02` e `Lecture_room`; a
validação, `Office`; o teste, `Home_01` e `Home_02`. Ambiente, câmera,
resolução, fps e qualidade de pose variam juntos entre os splits: só o teste
tem resolução mista (320×180 em 25 de 60 vídeos) e 24 fps. `lie_down` e
`lying` não aparecem no treino `[E2]`.

Macro-F1 restrita, cinco seeds `[E7]` `[E15]`:

| Braço | Validação | Teste | Teste (mín–máx) | Lacuna treino − teste |
| --- | --- | --- | --- | --- |
| A | 0,5485 ± 0,0144 | 0,4317 ± 0,0235 | 0,4042–0,4518 | 0,4513 |
| B0 | 0,5763 ± 0,0143 | 0,2295 ± 0,0720 | 0,1313–0,3298 | 0,7690 |
| B1 | 0,5808 ± 0,0177 | 0,2742 ± 0,0554 | 0,1943–0,3230 | 0,7217 |
| C0 | 0,5270 ± 0,0113 | 0,4042 ± 0,0199 | 0,3839–0,4314 | 0,5819 |
| C1 | 0,5161 ± 0,0115 | 0,4131 ± 0,0176 | 0,3856–0,4291 | 0,5672 |

Alarmes, cinco seeds; validação com 17 eventos e teste com 37 `[E10]` `[E15]`:

| Braço | Sensibilidade val | FA/h val | Sensibilidade teste | FA/h teste (mín–máx) | Latência média teste (s) |
| --- | --- | --- | --- | --- | --- |
| A | 0,835 ± 0,049 | 179,8 ± 32,7 | 0,870 ± 0,082 | 120,4 ± 29,2 (90,3–156,5) | 0,52 ± 0,04 |
| B0 | 0,906 ± 0,089 | 73,0 ± 30,0 | 0,660 ± 0,071 | 137,2 ± 92,2 (18,1–258,8) | 1,10 ± 0,23 |
| B1 | 0,894 ± 0,064 | 81,7 ± 27,5 | 0,681 ± 0,073 | 116,8 ± 59,1 (24,1–186,6) | 0,96 ± 0,36 |
| C0 | 0,918 ± 0,032 | 88,3 ± 16,1 | 0,627 ± 0,136 | 151,7 ± 27,0 (114,3–174,5) | 0,40 ± 0,00 |
| C1 | 0,882 ± 0,042 | 102,4 ± 13,0 | 0,816 ± 0,052 | 87,9 ± 28,4 (54,2–114,3) | 0,44 ± 0,05 |

- Todos os braços perdem desempenho da validação para o teste, mais do que
  no Le2i-CS `[E7]`.
- B0 e B1 têm a maior Macro-F1 de validação e a menor de teste. As faixas de
  teste dos dois ficam inteiramente abaixo da faixa de A `[E7]` `[E15]`.
- Em todas as seeds, B0 prediz `other` para 22% a 80% das janelas de teste
  fora de `other`, e B1 para 26% a 70%, contra 6% a 11% em A e 9% a 13% em
  C0/C1. A seed 42 é a mais extrema de B0 (80%), e a Macro-F1 de teste de B0
  vai de 0,13 a 0,33 entre seeds `[E16]`.
- C0 e C1 ficam próximos de A em Macro-F1 de teste, com faixas sobrepostas,
  mas não em alarmes: C0 tem a menor sensibilidade média de teste, com o
  maior desvio, e C1 tem a maior sensibilidade entre os braços de fusão e o
  menor FA/h médio `[E15]`.
- Em validação (`Office`, também fora do treino), os braços de fusão têm FA/h
  menor que A, e B0/B1 não superpredizem `other`. O split de desenvolvimento
  não antecipou a falha de B0/B1 no teste `[E10]`.

Exemplos da seed 42 do Le2i-CV ficam em runs locais e não são canônicos. O
braço A da seed 42 tem Macro-F1 de teste 0,4082, sensibilidade 0,9459 e 90,27
FA/h `[E17]`.

### Comportamento dos gates de B1 e C1

Análise pós-hoc dos checkpoints canônicos da seed 42, em Le2i-CS. Média de
`g_pose` na condição limpa e na maior severidade de cada degradação; valor
maior dá mais peso à pose `[E18]`:

| Braço | Split | Limpo | Pose: 16 pontos removidos | Visual: blur 12 |
| --- | --- | --- | --- | --- |
| B1 | val | 0,4998 | 0,5073 | 0,5104 |
| B1 | teste | 0,5007 | 0,5074 | 0,5114 |
| C1 | val | 0,7551 | 0,6269 | 0,7461 |
| C1 | teste | 0,7389 | 0,6258 | 0,7316 |

O gate de B1 fica perto de 0,50 em todas as condições e sobe com a
degradação da pose, na direção contrária à esperada. O gate de C1 reage à
pose na direção esperada, mas também reduz `g_pose` quando o SAM 3 é
degradado, na direção contrária à esperada `[E18]`.

## Discussão

### Fusão visual no Le2i-CS

Os quatro braços de fusão têm Macro-F1 média de teste e FA/h médio melhores
que A no Le2i-CS `[E6]`. É uma observação descritiva de cinco seeds sobre um
teste de dois sujeitos e 22 eventos, sem teste de hipótese `[E3]` `[E14]`.
O texto não deve afirmar superioridade estatística.

Os braços de fusão quase memorizam o treino `[E6]` `[E7]`, e os erros
dominantes (`other→walk`, `standing→walk`, `sitting→stand_up`) são
compartilhados por todos os braços. Isso sugere uma origem no conteúdo ou nos
rótulos, não no backbone, mas a hipótese não foi testada `[E7]` `[E19]`. As
Etapas 3A–3C descrevem `other→walk` sem atribuir causa
([seção própria](#investigacao-other-walk)).

### Limitações do gate adaptativo (B1 e C1)

- **B1 versus B0.** A vantagem média de B1 é pequena, e as faixas de teste se
  sobrepõem `[E6]`. Na seed 42, todos os intervalos pareados B1 − B0 de teste
  contêm zero, e o FA/h é igual nos dois braços `[E14]`.
- **C1 versus C0.** O resultado é misto: C1 é melhor na validação e pior no
  teste, em média `[E6]`. Na seed 42, C1 − C0 tem Macro-F1 de teste −0,0307
  e FA/h +23,3. Com dois sujeitos, isso significa só que os dois sujeitos e o
  conjunto têm delta do mesmo sinal `[E14]`.
- **Gate pouco responsivo e direções inconsistentes.** O gate de B1 fica
  perto de 0,50, compatível com B1 se comportar como B0 com ramos
  reescalados. Nenhum dos dois gates responde nas duas direções esperadas.
  A evidência não sustenta uma leitura de confiabilidade bidirecional dos
  gates escalares e, sozinha, não mostra que um gate vetorial seja
  necessário `[E18]` `[E19]`.
- **Escopo.** Os proxies de qualidade não são confianças calibradas `[E1]`,
  e o gate não foi medido no Le2i-CV `[E19]`.

O manuscrito deve descrever B1 e C1 como fusão adaptativa por proxies de
qualidade que não produziu ganho consistente sobre a concatenação simples.

### Limitações de generalização entre domínios

A queda de Le2i-CV é evidência de generalização limitada a ambientes não
vistos, mas o deslocamento de domínio é confundido: ambiente, câmera,
resolução, fps e qualidade de pose variam juntos, e os sujeitos se sobrepõem
entre splits `[E2]`. Nenhum desses fatores pode ser apontado como causa.

B0 e B1 mostram a maior queda e um deslocamento reprodutível para `other`,
mas a intensidade varia muito entre seeds `[E16]`. As explicações
concorrentes registradas, como atalho de cena nas features DINOv3, mudança de
entrada exclusiva do teste (resolução mista e 24 fps), composição de classes
e sujeitos, instabilidade de otimização e gate de B1 quase constante, não
foram testadas nem separadas `[E19]`. O texto não deve afirmar que o
deslocamento visual de domínio ou atalhos de cena foram estabelecidos como
causa.

O Le2i-CS não deve ser descrito como evidência entre domínios: seu treino já
contém os seis ambientes `[E2]`. O trabalho também usa um único conjunto de
dados, e Le2i-CV é uma análise intra-Le2i.

### Investigação de B0/B1 (Etapas 2A–2G)

Investigação exploratória e não canônica. Nenhum artefato dela foi promovido,
e nenhuma decisão congelada mudou `[E19]`.

1. **Etapa 2A, decisão.** A falha de B0/B1 no Le2i-CV não aparece em
   `train`/`val`. Sem sinal de desenvolvimento, uma correção seria escolhida
   pelo teste, e por isso nenhuma mudança de modelo foi feita. Foi proposto um
   instrumento: dobras internas por ambiente no `train` do Le2i-CV `[E19]`.
2. **Etapas 2B e 2C, viabilidade e implementação.** As três dobras são
   estruturalmente viáveis como instrumento, mas não como medida por classe
   no ambiente retido `[E20]`.
3. **Etapa 2D, piloto (seed 42, nove treinos).** Resultado negativo. Em
   nenhuma dobra B0/B1 deslocam para `other`; nas três, predizem `other`
   menos que A. O critério de cinco seeds se tornou impossível de satisfazer,
   e os 36 treinos restantes não foram executados. Apareceu outro modo de
   falha, não pré-registrado: em `Lecture_room`, B0 e B1 predizem `walk` em
   97,6% e 99,9% das janelas, com Macro-F1 de 0,096 e 0,086, contra 0,447 em A
   `[E21]`.
4. **Etapa 2E, auditoria.** As checagens não encontram defeito nos arrays de
   pose, DINOv3 e qualidade nem no alinhamento das janelas, dentro do escopo
   testado. Com as estatísticas da dobra, o DINOv3 de `Lecture_room` fica
   fora da faixa do desenvolvimento (RMS do z médio 4,208 contra 0,557 na
   mediana), e as duas salas de café são quase indistinguíveis no DINOv3. A
   coincidência entre esse deslocamento e o colapso não estabelece causa
   `[E22]`.
5. **Etapa 2F, contrafactual.** Trocar o DINOv3 padronizado pela média da
   dobra desfaz o colapso (fração `walk` de 0,976 para 0,017 em B0 e de 0,999
   para 0,022 em B1), mas a Macro-F1 em `Lecture_room` sobe só para 0,167 e
   0,147, ainda abaixo de A. Nas salas de café, a mesma troca derruba a
   Macro-F1 de 0,492–0,712 para 0,124–0,265 `[E23]`.
6. **Etapa 2G, encerramento da decisão.** A Etapa 2G encerra a decisão
   sobre intervenções de modelo com a evidência disponível: ela não justifica
   dropout visual, neutralização na inferência, mudança de arquitetura nem
   mais seeds. Ela não resolve a causa da falha de B0/B1 no teste do
   Le2i-CV; essa pergunta continua em aberto, e a falha segue registrada como
   limitação de generalização sem causa atribuída `[E19]`.

A neutralização mostra que as predições são sensíveis à entrada visual. Ela
não é uma melhoria validada: a entrada neutralizada é constante e fora da
distribuição de treino, piora as salas de café, não supera A, não neutraliza
o gate de B1 e nunca foi avaliada em `val` ou `test` oficiais `[E19]` `[E23]`.
Nada nas Etapas 2D–2F explica a falha no teste oficial, porque as dobras são
todas 320×240 a 25 fps e não testam resolução nem fps `[E19]`.

### Investigação de `other→walk` (Etapas 3A–3C) { #investigacao-other-walk }

Investigação exploratória e não canônica do erro dominante de Le2i-CS,
recomendada pela síntese diagnóstica entre braços `[E26]`. Nenhuma etapa
treinou, reextraiu features, ajustou limiar, selecionou ou promoveu
artefato. A evidência tem três naturezas, que o texto final deve manter
separadas:

| Etapa | Natureza da evidência | Split | Unidade |
| --- | --- | --- | --- |
| 3A, contexto de janela | Observação descritiva no `test` já exposto, com as predições da reexecução validada | `val` e `test` | Janela, média entre as seeds 42–46 |
| 3B, revisão de trechos | Anotação humana não cega, feita após o desmascaramento, sobre uma amostra selecionada do `test` exposto | `test` | 15 trechos |
| 3C, composição do treino | Medição só em `train`, com regras de decisão fixadas antes da execução | `train` | Janela e sujeito |

1. **Etapa 3A, contexto de janela.** Em `test`, `other→walk` persiste em
   janelas puras de `other`. Nos cinco braços, a taxa média de erro é de 35%
   a 44% nas 274 janelas puras, de 35% a 43% nas 586 misturadas e de 37% a
   51% nas 229 janelas a 23 ou mais quadros do início do trecho, inteiramente
   dentro dele. Pela leitura definida na síntese, a associação com o contexto
   da janela é fraca. Em `val`, as 34 janelas `other` são todas misturadas, e
   a pergunta não pode ser respondida ali `[E26]` `[E27]`.
2. **Etapa 3B, revisão humana.** Foram selecionados de forma determinística
   9 trechos com erro persistente (todos os braços preveem `walk` em pelo
   menos 4 de 5 seeds) e 6 controles corretos, em `Lecture_room` e
   `Coffee_room_02` do `test`. Após o desmascaramento, o revisor não relatou
   caminhada em quatro erros persistentes, todos em janelas puras de `other`:
   pegar algo na geladeira, pegar uma jaqueta de uma cadeira, abaixar-se para
   pegar algo do chão e fechar uma porta. Nos outros cinco erros, relatou
   passos curtos, não caminhada contínua. A leitura preliminar das folhas,
   feita por um modelo de IA antes do desmascaramento, não entra como
   evidência `[E28]`.
3. **Etapa 3C, composição do treino.** Só com `train` (seis sujeitos),
   nenhuma das três regras pré-declaradas foi sustentada: `other` não tem
   menos janelas puras que `walk` (64% contra 50%), a fração de janelas puras
   com deslocamento baixo da bbox é maior em `other` (81%) que em `walk`
   (32%), e a pose no quadro final está disponível em
   quase todas as janelas dos dois rótulos (100% e 98%). O veredito
   pré-declarado é `no_model_modification` `[E29]`.

A investigação termina com `no_model_modification`: nenhuma mudança de
modelo, treino, janelamento, rótulos ou alarmes foi feita `[E29]`. Ela não
identifica qual sinal de entrada ou regra de decisão produz a predição
`walk`, não estabelece erro de anotação e não mede a prevalência desses
casos em `test` nem em outros ambientes `[E28]` `[E29]`.

- Os quatro casos sem caminhada relatada vêm de uma amostra escolhida a
  partir de erros persistentes em dois ambientes, com pureza desbalanceada
  entre erros e controles (6 de 9 puros contra 1 de 6) e itens do mesmo
  vídeo. São relatos não cegos, sem estimativa de taxa nem teste de
  significância `[E28]`.
- As regras da Etapa 3C são convenções, a consistência entre seis sujeitos é
  descritiva, e nenhum valor do treino foi comparado com as janelas de
  `test` `[E29]`.
- Os casos da Etapa 3B não são alvo de otimização. Uma intervenção futura só
  caberia com nova pergunta pré-declarada, aplicada igualmente a A, B e C e
  avaliada sem o `test` já exposto `[E28]` `[E29]`.

O manuscrito deve descrever `other→walk` como erro compartilhado e
persistente em janelas puras, sem causa atribuída.

### Dobras por sujeito e perda simulada de detecção (Etapas 4A–4D) { #dobras-por-sujeito }

Investigação exploratória e não canônica de uma única intervenção
predeclarada de robustez de pose no braço A do Le2i-CS. Ela usa só o `train`
oficial, dividido em três dobras por sujeito bruto; `val` e `test` oficiais
não foram lidos, e nenhum artefato foi promovido `[E30]` `[E34]`.

Macro-F1 restrita no grupo retido, braço A, seed 42 e checkpoint da última
época `[E31]` `[E32]` `[E33]`:

| Dobra | Sujeitos retidos | Retido: vídeos / janelas | Controle limpo (4B) | Δ limpo, 4C − 4B | Perda 4B − perda 4C (4D) |
| --- | --- | ---: | ---: | ---: | ---: |
| `retain_s0` | 0 | 67 / 10243 | 0,5584 | +0,0026 | +0,0003 |
| `retain_s1` | 1 | 31 / 3961 | 0,6576 | +0,0001 | +0,0031 |
| `retain_s3-4-6-8` | 3, 4, 6, 8 | 35 / 6536 | 0,6157 | +0,0018 | −0,0006 |

Cada linha é uma dobra, de tamanho muito diferente das outras. As linhas não
são agregadas nem comparadas entre si, e nenhum valor é comparável com as
tabelas de teste acima `[E30]` `[E31]`.

1. **Etapas 4A e 4B, dobras e controles.** O `train` sustenta três dobras com
   as oito classes restritas nos dois lados e nenhum vídeo compartilhado, com
   veredito `feasible_with_limitations` `[E30]`. Os controles usam a receita
   congelada e padronização ajustada só no desenvolvimento de cada dobra;
   `standing` tem a menor F1 nas três `[E31]`.
2. **Etapa 4C, intervenção.** No treino de desenvolvimento, cada janela
   recebe com probabilidade 0,20 três quadros consecutivos de detecção
   ausente, com as features de pose recalculadas pelo pipeline inalterado. O
   critério congelado (delta positivo em pelo menos duas dobras e nenhum
   abaixo de −0,02) foi atendido formalmente, mas os deltas ficam abaixo de
   0,003. A leitura sustentada é que a intervenção não piorou o retido limpo,
   não que o melhorou `[E32]`.
3. **Etapa 4D, robustez na inferência.** Com a mesma família de perda no
   retido, os dois modelos perdem no máximo 0,0061 de Macro-F1 restrita, e a
   diferença pareada das perdas não tem direção consistente entre as dobras
   `[E33]`.

A investigação termina sem promover `detection_dropout` a nenhuma
configuração canônica A/B/C `[E34]`. O manuscrito deve descrever a
intervenção como resultado exploratório sem melhoria relevante demonstrada,
não como ganho de desempenho ou de robustez.

- Há uma seed de treino por dobra e uma única realização da máscara de
  avaliação, sem teste de significância `[E32]` `[E33]`.
- As dobras são desbalanceadas: o sujeito 0 tem 49% das janelas rotuladas do
  `train`, e duas dobras retêm um único sujeito `[E30]`.
- As janelas são sobrepostas, e a corrupção é uma única condição sintética,
  que não reproduz a distribuição real das perdas de detecção `[E33]`.
- O `test` oficial, já exposto, não entrou na seleção de modelo nem da
  intervenção `[E34]`.

### Experimentos externos de física e movimento

Os experimentos do repositório derivado `rsramos30/gate-fall-motion-physics`
não são resultados do GateFall e não devem entrar nas tabelas acima. A
receita difere (pose de 138 canais, gate `physics` em B1/C1, checkpoint
`best_val`), os números não são comparáveis, a rodada local de 150 treinos
não foi revisada e seus resultados de teste não foram usados. Em validação,
`motion_physics − motion` fica entre −0,022 e +0,006, sem melhora consistente
`[E24]`. Se o manuscrito os mencionar, deve ser como trabalho externo
relacionado, em seção própria.

### Limitações gerais

- Le2i-CS tem um sujeito e 13 eventos em validação e dois sujeitos e 22
  eventos em teste `[E9]` `[E14]`.
- As métricas por janela usam `eval_stride` 1, com janelas sobrepostas que
  não são observações independentes `[E7]`.
- Cinco seeds dão desvio-padrão descritivo, não intervalo de confiança
  `[E3]`.
- O teste já foi exposto pelos diagnósticos. Qualquer intervenção derivada
  deles é exploratória e não pode ser confirmada pelo mesmo teste `[E19]`.
- O determinismo foi verificado na mesma máquina e stack de GPU, não entre
  máquinas `[E25]`.

## Mapa evidência → fonte

Caminhos de documentação são relativos a `docs/`. Os arquivos em
`runs/local/` não são versionados; o SHA-256 registrado permite conferir a
cópia usada.

| ID | Afirmação | Fonte |
| --- | --- | --- |
| E1 | Arquitetura dos braços, receita idêntica, gate escalar, qualidade como proxy | `train/baseline-a.md`, `train/baseline-b0.md`, `train/baseline-b1.md` (Gate adaptativo), `train/baseline-c0.md`, `train/baseline-c1.md` |
| E2 | Composição de Le2i-CV, fatores confundidos, sobreposição de sujeitos, classes ausentes do treino, Le2i-CS não é entre domínios | `analysis/le2i-cv-generalization.md` |
| E3 | Semântica do desvio entre seeds, teste descritivo, nenhuma seleção | `analysis/multiseed-summary.md` (Resultado da execução real — B0, B1, C0 e C1) |
| E4 | Referências canônicas da seed 42, apenas Le2i-CS | `reference/le2i-runs.md`; `analysis/classification-diagnosis.md` (Entradas) |
| E5 | Caráter exploratório dos diagnósticos e das dobras | `analysis/multiseed-error-consolidation.md`, `analysis/intervention-decision.md` |
| E6 | Agregados de cinco seeds do Le2i-CS: Macro-F1 e eventos | `analysis/multiseed-summary.md` (A: Resultado da execução real; B0, B1, C0, C1: seções próprias) |
| E7 | Lacunas treino − teste, parcelas do déficit, confusões dominantes, Macro-F1 de Le2i-CV | `analysis/classification-diagnosis.md` (Resultado da execução real) |
| E8 | Protocolo de alarme `N=3`, `R=5,0 s` | `eval/baseline-a-events.md` (Protocolo de alarme) |
| E9 | 13 eventos em validação e 22 em teste no Le2i-CS; um sujeito em validação | `analysis/alarm-protocol-sensitivity.md`; `analysis/stratified-error-diagnosis.md` (Histórico de execuções) |
| E10 | Alarmes em validação de Le2i-CS e Le2i-CV, cinco seeds; B0/B1 não superpredizem `other` em `val` | `analysis/intervention-decision.md` (Evidência confirmada em `train` e `val`) |
| E11 | 0,171 h de vídeo de teste; 10 falsos alarmes = 58,4 FA/h | `eval/baseline-a-events.md` (Resultado da execução real) |
| E12 | Varredura pós-hoc de `N` e `R` para A | `analysis/alarm-protocol-sensitivity.md` (Resultado da execução real) |
| E13 | Métricas da seed 42 por braço | `runs/reference/le2i/baseline_{a,b0,b1,c0,c1}/metrics.json` (`final.*.macro_f1_restricted`) e `event_metrics.json` (`splits.*`) |
| E14 | Bootstrap agrupado por sujeito, resolução inferencial, comparações pareadas | `analysis/grouped-bootstrap.md` (Resultado da execução real) |
| E15 | Agregados de cinco seeds do Le2i-CV: eventos de teste | `runs/local/le2i_cv/multiseed_summaries/baseline_{a,b0,b1,c0,c1}/multiseed_summary.json`, campo `aggregate.events.test` (SHA-256 abaixo) |
| E16 | Deslocamento de B0/B1 para `other` no teste do Le2i-CV, variação entre seeds | `analysis/multiseed-error-consolidation.md` (Histórico de execuções) |
| E17 | Braço A, seed 42, Le2i-CV | `analysis/le2i-cv-generalization.md` (Resultados — braço A, seed 42) |
| E18 | Média de `g_pose` sob degradação | `analysis/gate-degradation.md` (Resultados reais no Le2i); `runs/reference/le2i/baseline_{b1,c1}/gate_degradation.json` |
| E19 | Decisão da Etapa 2A, explicações concorrentes, evidência ausente, encerramento da Etapa 2G | `analysis/intervention-decision.md` |
| E20 | Viabilidade das dobras (Etapa 2B) e implementação (Etapa 2C) | `analysis/internal-fold-feasibility.md`; `train/internal-folds.md` |
| E21 | Piloto das dobras internas e colapso em `Lecture_room` (Etapa 2D) | `analysis/internal-fold-pilot.md` |
| E22 | Auditoria dos ambientes retidos (Etapa 2E) | `analysis/internal-fold-retained-audit.md` |
| E23 | Contrafactual de neutralização visual (Etapa 2F) | `analysis/visual-neutralization-counterfactual.md` |
| E24 | Experimentos externos de física e movimento | `analysis/intervention-decision.md` (Experimentos externos de física e movimento) |
| E25 | Determinismo na mesma máquina, sem garantia entre máquinas | `analysis/multiseed-summary.md` (Resultado da execução real); `reference/le2i-runs.md` (Conferência e reprodução) |
| E26 | Investigação recomendada para `other→walk` e leitura do contexto da janela | `analysis/cross-arm-diagnostic-synthesis.md` (Investigação recomendada) |
| E27 | Contexto de janela dos erros `other→walk` em `val` e no `test` exposto (Etapa 3A) | `analysis/window-context-diagnosis.md` (Conclusão; Resultado da execução real) |
| E28 | Seleção dos trechos e revisão humana não cega no `test` exposto (Etapa 3B) | `analysis/other-walk-review-set.md` (Cobertura da execução real; Revisão concluída) |
| E29 | Composição de `other` e `walk` só em `train`, regras e veredito `no_model_modification` (Etapa 3C) | `analysis/other-walk-train-composition.md` (Regras de decisão; Resultados observados; Interpretação) |
| E30 | Viabilidade das dobras por sujeito do Le2i-CS, esquema `pooled_minority_subjects`, desbalanço entre dobras e veredito `feasible_with_limitations` (Etapa 4A) | `analysis/subject-fold-feasibility.md` (Resultados observados; Veredito) |
| E31 | Dobras congeladas, padronização por dobra e controles limpos do braço A, seed 42 (Etapa 4B) | `train/subject-folds.md` (Dobras congeladas; Execução registrada (não canônica)) |
| E32 | Intervenção `detection_dropout`, critério congelado e deltas pareados limpos (Etapa 4C) | `train/subject-fold-detection-dropout.md` (A intervenção; Critério exploratório congelado; Execução registrada (não canônica)) |
| E33 | Robustez de inferência à perda simulada de detecção e diferença pareada das perdas (Etapa 4D) | `analysis/subject-fold-dropout-robustness.md` (Execução registrada (não canônica); Limitações) |
| E34 | Encerramento das Etapas 4A–4D sem promoção de `detection_dropout`, `test` oficial fora da seleção | `analysis/intervention-decision.md` (Encerramento da intervenção de perda de detecção) |

Os sumários locais de E15 foram produzidos pelo
[sumário multi-seed](multiseed-summary.md), que valida configuração,
checkpoint e `event_metrics.json` de cada seed. Seus agregados de Macro-F1
coincidem com E7, e os de validação, com E10. Os agregados de eventos de
teste do Le2i-CV não estão em nenhuma outra página versionada.

| Braço | `multiseed_summary.json` (SHA-256) |
| --- | --- |
| A | `f6932b05ff1ee803767c0dc07ee65f1a07709125e334ef8efe694474ae3581fe` |
| B0 | `3201aba159583603f2aaad08c8b33c681ab4092ff28149b8d40d4ec62081208a` |
| B1 | `2b01a50078c4d23f7e0ef987b5f3ff655cc1ed21fbe20c5a81376077c3c23e76` |
| C0 | `070ea22e4f8e0ef062cb712ebf98dd0c668348171f00901e7bffc8aa41c2fb10` |
| C1 | `8f6b44bf79eb29ada470c1ae5dd9a59b27bc95faa6a38816163662661cdda924` |

## Pendências para a versão final

- Este rascunho não cita literatura. Comparações com trabalhos anteriores e
  referências bibliográficas precisam ser adicionadas pelo autor, a partir de
  fontes conferidas.
- O texto final deve preservar a separação entre agregados, exemplos da seed
  42, investigações exploratórias e experimentos externos.
