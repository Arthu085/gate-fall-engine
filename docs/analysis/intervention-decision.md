# Análise — Decisão sobre intervenções no modelo (Etapas 2A, 2G, 4A–4D, 5A e 6A–6B)

Este documento registra uma decisão **exploratória e não canônica**: se a
evidência de desenvolvimento (`train` e `val`) justifica alguma melhoria
controlada de modelo em A, B0, B1, C0 ou C1. Ele não implementa modelos,
receitas de treino, aumentos de dados, gates nem buscas de hiperparâmetros, e
não altera nenhum artefato canônico.

O split `test` já foi exposto pelas etapas anteriores. Aqui ele aparece só
como contexto, para dizer qual falha motivou a pergunta. Nenhum número de
`test` é usado como critério de escolha ou de aprovação de uma intervenção.

## Decisão

**Não implementar agora nenhuma mudança de modelo.** A falha mais grave
observada, o deslocamento de B0/B1 para `other` em Le2i-CV, não aparece nos
dados de desenvolvimento. Sem um sinal em `train`/`val`, qualquer correção
seria escolhida, direta ou indiretamente, pelo `test`. Os erros sistemáticos
que aparecem em `val` são compartilhados por todas as armas ou têm suporte
pequeno demais para orientar uma intervenção específica de B0/B1.

Na Etapa 2A, o único próximo passo justificável era o candidato 1 abaixo, um
instrumento de validação que não muda nenhum modelo. O candidato 2 só seria
considerado se o candidato 1 reproduzisse o deslocamento para `other` em
dados de desenvolvimento.

O candidato 1 foi concluído depois, com resultado negativo na
[Etapa 2D](internal-fold-pilot.md). A
[Etapa 2G](#encerramento-da-investigacao-de-b0b1-etapa-2g) consolida esse
resultado com as Etapas 2E e 2F, encerra a investigação e mantém a decisão:
nenhuma mudança de modelo é justificada.

As Etapas 4A–4D testaram depois uma única intervenção de robustez de pose no
braço A, avaliada só em dobras por sujeito do `train` oficial do Le2i-CS. O
[encerramento das Etapas 4A–4D](#encerramento-etapas-4a-4d) registra o
resultado: `detection_dropout` não é promovido, e nenhum modelo canônico
muda.

A Etapa 5A testou, nas mesmas dobras e contra os mesmos controles, uma
intervenção distinta de amostragem balanceada por sujeito no treino do braço
A. O [encerramento da Etapa 5A](#encerramento-etapa-5a) registra que o
critério congelado não foi atendido: a amostragem não é promovida, e nenhum
modelo canônico muda.

As [Etapas 6A–6B](#encerramento-etapas-6a-6b) examinaram os erros
`standing→walk` e `sitting→stand_up` em desenvolvimento. O diagnóstico e a
revisão visual preliminar por IA não identificaram causa única nem
justificaram intervenção ou correção de anotações.

## Fontes

- [Diagnóstico de erros de classificação](classification-diagnosis.md),
  [diagnóstico estratificado](stratified-error-diagnosis.md) e
  [consolidação multi-seed](multiseed-error-consolidation.md): relatórios
  locais das seeds 42–46, nos dois protocolos.
- `final.train` e `final.val` de `metrics.json` dos 50 runs listados na
  consolidação multi-seed, incluindo o histórico por época de
  `val_macro_f1_restricted`. A fração de janelas preditas como `other` em
  `train` foi calculada da `confusion_matrix`: soma da coluna `other` fora da
  linha `other`, dividida pelas janelas fora de `other`.
- [Sumário multi-seed](multiseed-summary.md), para as métricas de alarme de
  `val`, e [degradação do gate B1/C1](gate-degradation.md),
  [bootstrap agrupado](grouped-bootstrap.md) e
  [generalização Le2i-CV](le2i-cv-generalization.md), para a composição dos
  splits e as análises anteriores de modalidade.

Os relatórios locais não são versionados. Os comandos das páginas acima os
reproduzem.

## Evidência confirmada em `train` e `val`

Valores calculados, média ± desvio-padrão amostral entre cinco seeds. O desvio
entre seeds não é um intervalo de confiança.

| Protocolo | Arma | Macro-F1 train | Macro-F1 val | Fração predita `other` em train | Fração predita `other` em val | Revocação de `other` em val | Precisão de `other` em val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Le2i-CS | A | 0,867 ± 0,006 | 0,648 ± 0,014 | 0,036 | 0,042 ± 0,010 | 0,48 ± 0,17 | 0,16 ± 0,04 |
| Le2i-CS | B0 | 0,997 ± 0,000 | 0,680 ± 0,020 | 0,000 | 0,010 ± 0,002 | 0,44 ± 0,01 | 0,42 ± 0,05 |
| Le2i-CS | B1 | 0,994 ± 0,001 | 0,689 ± 0,018 | 0,001 | 0,013 ± 0,001 | 0,44 ± 0,00 | 0,37 ± 0,03 |
| Le2i-CS | C0 | 0,986 ± 0,002 | 0,662 ± 0,014 | 0,005 | 0,052 ± 0,006 | 0,51 ± 0,05 | 0,14 ± 0,02 |
| Le2i-CS | C1 | 0,978 ± 0,003 | 0,681 ± 0,009 | 0,009 | 0,055 ± 0,002 | 0,62 ± 0,11 | 0,16 ± 0,03 |
| Le2i-CV | A | 0,883 ± 0,009 | 0,549 ± 0,014 | 0,053 | 0,095 ± 0,018 | 0,40 ± 0,07 | 0,63 ± 0,02 |
| Le2i-CV | B0 | 0,998 ± 0,000 | 0,576 ± 0,014 | 0,000 | 0,070 ± 0,058 | 0,30 ± 0,20 | 0,66 ± 0,05 |
| Le2i-CV | B1 | 0,996 ± 0,001 | 0,581 ± 0,018 | 0,000 | 0,041 ± 0,020 | 0,18 ± 0,06 | 0,65 ± 0,05 |
| Le2i-CV | C0 | 0,986 ± 0,001 | 0,527 ± 0,011 | 0,007 | 0,146 ± 0,020 | 0,51 ± 0,08 | 0,58 ± 0,02 |
| Le2i-CV | C1 | 0,980 ± 0,002 | 0,516 ± 0,012 | 0,013 | 0,118 ± 0,008 | 0,43 ± 0,03 | 0,60 ± 0,01 |

Alarmes em `val`, com o protocolo congelado (`N=3`, `R=5,0 s`):

| Protocolo | Arma | Eventos | Sensibilidade | FA/h |
| --- | --- | --- | --- | --- |
| Le2i-CS | A | 13 | 0,985 ± 0,034 | 20,8 ± 22,6 |
| Le2i-CS | B0, B1, C0, C1 | 13 | 1,000 nas cinco seeds | 0,0 nas cinco seeds |
| Le2i-CV | A | 17 | 0,835 ± 0,049 | 179,8 ± 32,7 |
| Le2i-CV | B0 | 17 | 0,906 ± 0,089 | 73,0 ± 30,0 |
| Le2i-CV | B1 | 17 | 0,894 ± 0,064 | 81,7 ± 27,5 |
| Le2i-CV | C0 | 17 | 0,918 ± 0,032 | 88,3 ± 16,1 |
| Le2i-CV | C1 | 17 | 0,882 ± 0,042 | 102,4 ± 13,0 |

O que esses números sustentam:

1. **B0/B1 não superpredizem `other` em desenvolvimento.** Nos dois
   protocolos, a fração média de janelas preditas como `other` em `val` é
   menor em B0/B1 que em A, C0 e C1. Em Le2i-CS, as predições de `other` de B0/B1 são
   raras e mais precisas que as de A. Em Le2i-CV, `Office` também é um
   ambiente fora do treino e, mesmo assim, B0/B1 têm Macro-F1 e FA/h melhores
   que A.
2. **B0/B1 quase memorizam o treino.** A Macro-F1 de `train` passa de 0,99, e
   a perda final de treino média fica entre 0,008 e 0,029, contra 0,27 a 0,34
   em A.
   C0/C1 também passam de 0,97 sem o mesmo comportamento, então memorizar o
   treino, sozinho, não explica a falha.
3. **O erro estável de B0/B1 em Le2i-CS `val` é `standing→walk`.** A confusão
   é a dominante nas cinco seeds, com 90% (B0) e 84% (B1) do suporte de
   `standing`. A revocação de `standing` é 0,06 ± 0,06 em B0 e 0,11 ± 0,07 em
   B1, contra 0,34 a 0,38 em A, C0 e C1. Em troca, a revocação de `walk` sobe
   para 0,97 contra 0,81 em A. O suporte é pequeno: 96 janelas de 4 vídeos de
   um único sujeito (5), com janelas sobrepostas.
4. **Em Le2i-CV `val`, B1 confunde `other→standing`** em cinco de cinco
   seeds (30% do suporte de `other`), e a revocação de `other` cai para
   0,18. B0 é instável: a fração predita como `other` vai de 0,023 a 0,166, e
   a revocação de `standing`, de 0,02 a 0,85. Essa instabilidade é o único
   sinal de desenvolvimento ligado à falha de `test`, e ele é fraco.
5. **Os erros dominantes de Le2i-CS `val` são compartilhados.**
   `standing→walk` e `sitting→stand_up` dominam em todas as armas. Essa
   observação não identifica uma causa nem estabelece defeito de anotação.
6. **O `val` de Le2i-CS satura os alarmes.** B0, B1, C0 e C1 têm
   sensibilidade 1,0 e FA/h 0 nas cinco seeds com 13 eventos. Esse split não
   discrimina efeito de intervenção em alarmes.

Para contexto, já exposto e não usado como critério: em Le2i-CV `test`
(`Home_01`, `Home_02`), B0 prediz `other` para 22% a 80% das janelas fora de
`other` e B1 para 26% a 70%, contra 6% a 11% em A. Em Le2i-CS `test`, o erro
dominante de todas as armas é `other→walk` em `Lecture_room`.

## Explicações concorrentes para o deslocamento de B0/B1 em Le2i-CV

Nenhuma delas foi testada. `train`/`val` não as separam.

- **Atalho de cena nas features DINOv3.** O treino de Le2i-CV tem só três
  ambientes (`Coffee_room_01`, `Coffee_room_02`, `Lecture_room`), e B0/B1
  quase memorizam esse treino. Contra essa leitura, `Office` também é um
  ambiente não visto e não provoca o deslocamento.
- **Mudança de entrada específica do `test`.** Só o `test` de Le2i-CV tem
  resolução mista (320×180 em 25 de 60 vídeos) e fps 24. O DINOv3 recebe o
  quadro redimensionado para 224×224, então uma mudança de proporção afeta a
  entrada visual de forma diferente da pose e do descritor SAM 3; o
  redimensionamento direto, sem recorte, distorce a proporção 16:9. `val` e
  `train` são todos 320×240 a 25 fps, então não podem testar isso.
- **Composição de classes e sujeitos.** O ambiente retido também muda
  sujeitos, ações e suporte por classe (ver
  [generalização Le2i-CV](le2i-cv-generalization.md)).
- **Instabilidade de otimização.** A magnitude do deslocamento em `test`
  varia muito entre seeds, e B0 já é a arma mais instável em `val`.
- **Gate de B1 praticamente constante.** Na
  [degradação pós-hoc](gate-degradation.md), em Le2i-CS, `g_pose` médio de B1
  fica perto de 0,50 em todas as condições e reage pouco, e na direção
  contrária à esperada para a degradação de pose. Isso é compatível com B1 se
  comportar como B0 com ramos reescalados. Não há medida do gate em Le2i-CV.

## Evidência ausente

- Um split de desenvolvimento em que o deslocamento para `other` apareça.
  Sem ele, nenhum critério de `train`/`val` mede se uma intervenção corrige a
  falha.
- Medidas sem rótulo da distância entre as features DINOv3 de `val`/`test` e
  as de `train`, separadas por resolução.
- Comportamento do gate de B1/C1 em Le2i-CV.
- Mais de um sujeito em Le2i-CS `val` e mais de 13 eventos. O
  [bootstrap agrupado](grouped-bootstrap.md) não estima variação entre
  sujeitos com um único sujeito, e as métricas por janela usam janelas
  sobrepostas.
- Uma revisão humana independente dos casos `standing` e `sitting` examinados
  nas Etapas 6A–6B. A leitura preliminar por IA não valida os rótulos.

## Experimentos externos de física e movimento

Os experimentos do repositório derivado
[rsramos30/gate-fall-motion-physics](https://github.com/rsramos30/gate-fall-motion-physics)
já foram registrados: o gate `physics` com provas Lean (commit `301b32e`) e a
supervisão auxiliar de movimento com coerência cinemática (commits `9fb8a8a`
e `5f6e2ae`). O registro de validação daquele repositório
(`specs/002-motion-physics-learning/validation.md`) cobre só checagens de
engenharia e afirma que os experimentos reais ainda não tinham sido
executados quando foi escrito.

Existe, fora deste repositório, uma rodada local de 150 treinos (5 armas × 3
receitas × 5 seeds × 2 protocolos, revisão `5f6e2ae` sem alterações locais),
não revisada. Ela não serve como evidência para esta decisão:

- a receita difere da congelada aqui: esquema de pose de 138 canais, gate
  `physics` em B1/C1 e checkpoint `best_val` em vez da última época. Por isso
  os números não são comparáveis aos das armas deste repositório;
- com `best_val`, a Macro-F1 de `val` é o próprio critério de seleção do
  checkpoint e é otimista;
- mesmo assim, em `val` nenhuma receita auxiliar melhora de forma consistente:
  `motion_physics − motion` fica entre −0,022 e +0,006, dentro de um a dois
  desvios entre seeds, e é negativo em B0/B1 nos dois protocolos;
- os resultados de `test` dessa rodada não foram usados.

## Candidatos

No máximo dois, ambos fora de execução até revisão explícita.

Os dois candidatos são motivados por uma falha já examinada no `test`. Por
isso, qualquer intervenção que venha deles permanece exploratória, mesmo que
seja avaliada só em dobras de desenvolvimento: o `test` já exposto não pode
confirmar de forma independente a generalização.

### Candidato 1 — validação interna por ambiente em Le2i-CV (instrumento)

- **Mecanismo:** criar um sinal de desenvolvimento para a falha. Cada dobra
  retém um dos três ambientes de treino de Le2i-CV e treina nos outros dois.
  Se B0/B1 deslocarem para `other` no ambiente retido, a falha passa a ser
  observável sem `test`.
- **Mudança mínima:** treinar A, B0 e B1 com a receita congelada em três
  dobras internas, com estatísticas de padronização ajustadas só nos dois
  ambientes de treino da dobra. Não muda arquitetura, features nem
  protocolo de alarme, e não lê `val` nem `test` oficiais. São 45 treinos com
  as cinco seeds, ou 9 com uma seed como piloto, sem nova extração.
- **Critério em train/val:** o deslocamento é considerado reproduzido se, em
  pelo menos dois dos três ambientes retidos, a menor fração predita como
  `other` entre as seeds de B0 (ou B1) superar a maior entre as seeds de A,
  com o mesmo cálculo da consolidação multi-seed.
- **Leitura de um resultado negativo:** não reproduzir o deslocamento nas
  dobras não identifica a causa da falha no `test` oficial e não prova que
  propriedades de entrada específicas do `test`, como resolução e fps, sejam
  responsáveis. Cada dobra treina com dois ambientes em vez de três, com menos
  janelas e outra composição de classes, sujeitos e cenas. A ausência do
  deslocamento pode vir dessa mudança de treino, e não de uma diferença entre
  os ambientes retidos e o `test`.
- **Impacto em classificação e alarmes:** nenhum nos modelos existentes. O
  resultado só decide se o candidato 2 é testável.
- **Custo:** um novo modo de split e de padronização por dobra no pipeline de
  treino, mais os treinos. Os ambientes retidos têm suporte desigual por
  classe, e as dobras treinam com menos dados, então as magnitudes não são
  comparáveis às de Le2i-CV.
- **Reabertura de decisão congelada:** não muda a receita nem os splits
  oficiais. Acrescenta um protocolo de desenvolvimento, que precisa de
  aprovação explícita. Ele desloca a próxima etapa exploratória de treino,
  não se soma a ela.
- **Viabilidade:** avaliada sem treino na
  [Etapa 2B](internal-fold-feasibility.md).
- **Implementação:** [Etapa 2C](../train/internal-folds.md).
- **Resultado:** [Etapa 2D](internal-fold-pilot.md), encerrada como resultado
  exploratório negativo. No piloto da seed 42, B0 e B1 predizem `other` com
  fração menor que A nas três dobras, o que torna o critério impossível de
  satisfazer com as cinco seeds. Os 36 treinos restantes não são
  recomendados.

### Candidato 2 — dropout de modalidade visual em B0/B1 (condicional)

- **Mecanismo:** reduzir uma possível dependência de B0/B1 de um atalho
  visual. É uma hipótese não verificada, uma das explicações concorrentes
  acima. A memorização de `train` e a troca `standing→walk` em Le2i-CS `val`
  são compatíveis com ela, mas não estabelecem que a TCN separe estados
  próximos pela cena DINOv3 em vez do movimento.
- **Mudança mínima:** só no treino, zerar a projeção `E_V(visual)` de cada
  janela com uma probabilidade fixa definida antes de treinar, sem busca. A
  inferência não muda.
- **Critério em train/val:** nas dobras do candidato 1, a fração predita como
  `other` no ambiente retido cai para a faixa das seeds de A. Em `val`
  oficial, a Macro-F1 não cai mais que um desvio entre seeds em relação a
  B0/B1, o FA/h de Le2i-CV não sobe mais que um desvio, e a revocação de
  `standing` em Le2i-CS não piora.
- **Impacto potencial:** pode reduzir o deslocamento para `other` e a
  lacuna `train − val`. Também pode apagar a vantagem de B0/B1 sobre A em
  `val`. Em alarmes, Le2i-CS `val` está saturado e não mostraria efeito; só
  Le2i-CV `val` (17 eventos) mede mudança de sensibilidade ou FA/h.
- **Custo:** uma opção de treino nas armas de fusão e 20 treinos (B0 e B1 ×
  dois protocolos × cinco seeds), mais os das dobras.
- **Reabertura de decisão congelada:** sim. A invariante 1 exige receitas
  idênticas entre A, B e C. Uma regularização exclusiva das armas visuais
  muda a receita e exige reabrir formalmente essa decisão. Ela desloca, não
  acrescenta: a variante ocuparia o lugar de B0/B1 nessa ablação e não
  vira uma sexta arma.
- **Resultado:** não executado. A condição não ocorreu na
  [Etapa 2D](internal-fold-pilot.md), e a
  [Etapa 2G](#encerramento-da-investigacao-de-b0b1-etapa-2g) não encontra
  outro sinal de desenvolvimento que o justifique.

## Resultado de não prosseguir

Se o candidato 1 não for aprovado, ou não reproduzir o deslocamento, a
decisão permanece **não**: nenhuma intervenção de modelo é justificada pela
evidência atual. A [Etapa 2D](internal-fold-pilot.md) confirmou o segundo
caso: o candidato 1 não reproduz o deslocamento, e o candidato 2 não é
executado. O resultado de B0/B1 em Le2i-CV fica descrito como
limitação de generalização entre ambientes, sem causa atribuída, ao lado dos
erros compartilhados `standing→walk`, `sitting→stand_up` e `other→walk`.

## Encerramento da investigação de B0/B1 (Etapa 2G)

Esta seção encerra a investigação exploratória de melhoria de B0/B1 aberta
nesta página, consolidando as Etapas 2D, 2E e 2F. É só documentação: não há
novo experimento, script, mudança de modelo, promoção de referência nem
avaliação em `val` ou `test` oficiais. Os números completos ficam nas páginas
de cada etapa.

### Decisão final

**A evidência de desenvolvimento atual não justifica:**

- dropout de modalidade visual no treino (candidato 2);
- adotar a neutralização visual na inferência;
- mudar as arquiteturas de B0 ou B1;
- executar mais seeds, seja os 36 treinos restantes das dobras internas,
  seja repetições do contrafactual.

As decisões congeladas continuam valendo: receita idêntica entre A, B e C,
checkpoint da última época, protocolo de alarme `N=3`, `R=5,0 s`, splits
oficiais, runs canônicos e `runs/reference/`. Nenhum artefato das Etapas
2D–2F é promovido. A falha de B0/B1 no `test` oficial do Le2i-CV continua
registrada como limitação de generalização entre ambientes, sem causa
atribuída.

### Observações

1. **O critério original é negativo.** O
   [piloto da Etapa 2D](internal-fold-pilot.md#decisao) não reproduz o
   deslocamento para `other`: nas três dobras, B0 e B1 predizem `other` com
   fração menor que A, e o critério do candidato 1 não pode ser satisfeito
   com as cinco seeds. A pergunta pré-registrada fica respondida.
2. **Aparece outro modo de falha, não pré-registrado.** No `Lecture_room`
   retido, B0 e B1 colapsam para `walk` em quase todas as janelas e vídeos,
   e A não colapsa ([diagnóstico](internal-fold-pilot.md#diagnostico-de-lecture_room)).
   O colapso não é o deslocamento para `other` do `test` oficial e não
   satisfaz o critério.
3. **As checagens de integridade não encontram defeito no escopo testado.**
   A [auditoria da Etapa 2E](internal-fold-retained-audit.md#conclusao) não
   encontra defeito de forma, valores finitos, quadros zerados, variação
   temporal ou faixa nos arrays de pose, DINOv3 e qualidade dos 97 vídeos do
   `train` oficial. As janelas batem com `folds.json`, e a reinferência dos
   nove runs reproduz a matriz de `metrics.json`. As checagens não
   inspecionam os quadros de vídeo de origem e não excluem causas não
   examinadas.
4. **O deslocamento extremo das features padronizadas aparece no DINOv3.**
   Com as estatísticas da dobra, todos os vídeos de `Lecture_room` ficam
   acima de todos os vídeos de desenvolvimento na RMS do z médio DINOv3,
   enquanto a pose padronizada fica na faixa do desenvolvimento. O ambiente
   também tem confiança de keypoints e `q_pose` menores. As duas salas de
   café são quase indistinguíveis no DINOv3, e o desvio-padrão DINOv3
   ajustado na dobra `Lecture_room`, cujo desenvolvimento tem só essas duas
   salas, é menor que nas outras dobras. O deslocamento coincide com o
   colapso. São descrições, sem relação causal estabelecida.
5. **As predições são sensíveis à entrada visual.** No
   [contrafactual da Etapa 2F](visual-neutralization-counterfactual.md#conclusao),
   substituir o DINOv3 padronizado pela média da dobra desfaz o colapso para
   `walk` em `Lecture_room`, mas a Macro-F1 restrita continua muito abaixo
   da de A. Nas salas de café, a mesma intervenção derruba o desempenho de
   B0 e B1. As duas armas dependem da entrada visual também onde não há
   colapso.

### Por que a neutralização não é uma melhoria

O contrafactual estabelece sensibilidade das predições, não a causa do
colapso nem uma melhoria validada. A entrada neutralizada é constante e fora
da distribuição do treino, então a resposta pode refletir essa entrada
inédita e não a remoção de informação de cena. Ela piora as duas salas de
café, não supera A em `Lecture_room`, não neutraliza o gate de B1 e não foi
selecionada nem avaliada em `val` ou `test` oficiais. Na prática, ela
descarta a informação visual, e A já é a referência sem entrada visual.

### Hipóteses não resolvidas

Atalho de cena, mudança de entrada visual com recaída na classe mais
frequente do treino, variância visual estreita no desenvolvimento,
composição da dobra e variação de seed seguem compatíveis com as
observações. Nenhuma foi separada das demais, e nenhuma é um resultado (ver
as hipóteses da [Etapa 2E](internal-fold-retained-audit.md#hipoteses-nao-resolvidas)).

### Limitações do encerramento

- Uma seed por arma, três ambientes retidos e uma única intervenção fixa.
- Todo o `train` oficial é 320×240 a 25 fps. As dobras não testam a
  resolução mista nem o fps do `test` oficial do Le2i-CV.
- Nada das Etapas 2D–2F explica a falha do `test` oficial. A distância das
  features DINOv3 de `val`/`test` às de `train`, listada em
  [evidência ausente](#evidencia-ausente), continua não medida.
- O `test` oficial já exposto não pode confirmar de forma independente
  nenhuma intervenção derivada desta investigação.

## Encerramento da intervenção de perda de detecção (Etapas 4A–4D) { #encerramento-etapas-4a-4d }

Esta seção encerra a investigação exploratória das Etapas 4A–4D: uma
tentativa controlada de melhorar a Macro-F1 restrita do braço A no Le2i-CS
com uma única intervenção predeclarada de robustez de pose, avaliada só em
dobras por sujeito do `train` oficial. É só documentação: não há novo
experimento, treino, avaliação, seleção nem mudança de protocolo. As tabelas
por classe, os hashes e os comandos ficam nas páginas de cada etapa:

- [Etapa 4A, viabilidade das dobras por sujeito](subject-fold-feasibility.md);
- [Etapa 4B, dobras congeladas e controles do braço A](../train/subject-folds.md);
- [Etapa 4C, perda simulada de detecção no treino](../train/subject-fold-detection-dropout.md);
- [Etapa 4D, robustez de inferência à perda de detecção](subject-fold-dropout-robustness.md).

### Decisão final

**`detection_dropout` não é promovido** a nenhuma configuração canônica A/B/C,
e a intervenção exploratória fica encerrada. A evidência atual não justifica:

- acrescentar a perda simulada de detecção à receita de treino. Pela
  invariante 1, isso exigiria a mesma mudança em A, B e C e o retreino dos
  três;
- buscar um efeito com mais seeds, outras realizações da máscara, outras
  probabilidades ou outras durações de lacuna;
- avaliar a intervenção no `val` ou no `test` oficiais.

As decisões congeladas continuam valendo: receita idêntica entre A, B e C,
checkpoint da última época, splits oficiais, runs canônicos e
`runs/reference/`. Os runs, estatísticas e relatórios das Etapas 4B–4D ficam
em `runs/local/le2i/subject_folds/`, não versionado, e nenhum deles é
promovido. O `test` oficial, já exposto, ficou fora de toda seleção de modelo
e de intervenção e não foi lido por nenhuma das quatro etapas.

### Observações

1. **Três dobras por sujeito congeladas.** A
   [Etapa 4A](subject-fold-feasibility.md#veredito) chegou a
   `feasible_with_limitations` com o esquema `pooled_minority_subjects`. A
   [Etapa 4B](../train/subject-folds.md#dobras-congeladas) congelou os grupos
   retidos `retain_s0` (sujeito 0; 67 vídeos e 10243 janelas em stride 1),
   `retain_s1` (sujeito 1; 31 e 3961) e `retain_s3-4-6-8` (sujeitos 3, 4, 6 e
   8; 35 e 6536). Cada dobra ajusta a padronização de pose só no próprio
   treino de desenvolvimento.
2. **Controles limpos do braço A, seed 42.** Na
   [execução da Etapa 4B](../train/subject-folds.md#execucao-registrada-nao-canonica),
   a Macro-F1 restrita no retido é 0,5584, 0,6576 e 0,6157, nessa ordem de
   dobras. `standing` tem a menor F1 nas três.
3. **O critério da Etapa 4C foi atendido formalmente.** A
   [regra congelada](../train/subject-fold-detection-dropout.md#criterio-exploratorio-congelado)
   exige delta pareado de Macro-F1 restrita (aumentado menos controle)
   positivo em pelo menos duas das três dobras e nenhum delta abaixo de
   −0,02. Os deltas foram +0,0026, +0,0001 e +0,0018.
4. **Atender o critério não mostra melhoria relevante.** Os três deltas
   ficam abaixo de 0,003, e o de `retain_s1` só aparece na quarta casa
   decimal. Com uma seed por dobra, deltas desta ordem podem vir só da
   variação entre seeds, que não foi estimada. `standing` e `fall` não mudam
   de forma consistente entre as dobras. A leitura sustentada é que a
   intervenção não piorou o retido limpo, não que o melhorou
   ([Etapa 4C](../train/subject-fold-detection-dropout.md#execucao-registrada-nao-canonica)).
5. **A Etapa 4D não estabelece ganho consistente de robustez.** Na condição
   degradada predeclarada, os dois modelos perdem no máximo 0,0061 de
   Macro-F1 restrita. A diferença pareada das perdas (controle menos
   aumentado) é +0,0003, +0,0031 e −0,0006: não passa de 0,0031 em módulo e
   não tem direção consistente entre as dobras. O modelo da Etapa 4C foi
   treinado com a mesma família de perda avaliada, então uma perda menor nele
   indicaria robustez à condição treinada, não robustez geral
   ([Etapa 4D](subject-fold-dropout-robustness.md#execucao-registrada-nao-canonica)).

### Limitações do encerramento

- Uma seed de treino (42) por dobra e modelo, e uma única realização da
  máscara de avaliação na Etapa 4D. Nenhuma das duas variações é estimada.
- As três dobras têm tamanhos muito diferentes: o sujeito 0 tem 49% das
  janelas rotuladas do `train`, duas dobras retêm um único sujeito, e na
  dobra agregada `standing` vem de dois vídeos de um único sujeito. As dobras
  não são agregadas, comparadas entre si nem comparadas com o `test`
  oficial ou com os agregados de cinco seeds, e não verificam o alvo de
  0,800.
- As janelas são sobrepostas e não são observações independentes.
- Há uma única condição sintética de corrupção, com lacunas de três quadros
  em 20% das janelas. Ela não reproduz a distribuição real das perdas de
  detecção, lacunas mais longas, detecções erradas nem ruído de keypoints.
- O critério da Etapa 4C é uma regra de decisão predeclarada, não um teste
  de significância. Nenhuma afirmação de significância estatística é feita.

## Encerramento da amostragem balanceada por sujeito (Etapa 5A) { #encerramento-etapa-5a }

Esta seção encerra a [Etapa 5A](../train/subject-fold-subject-balanced-sampling.md):
uma única intervenção predeclarada de amostragem no treino do braço A do
Le2i-CS, em que cada época sorteia as janelas do treino de desenvolvimento
com probabilidade igual por sujeito. A pergunta era se reduzir o domínio de
poucos sujeitos sobre as janelas de treino melhora a generalização para
sujeitos retidos. É só documentação: não há novo experimento, treino,
avaliação, seleção nem mudança de protocolo. A Etapa 5A usa as dobras e os
controles limpos congelados da [Etapa 4B](../train/subject-folds.md), mas é
uma intervenção distinta da perda simulada de detecção: o
[encerramento das Etapas 4A–4D](#encerramento-etapas-4a-4d) e suas
conclusões ficam inalterados.

### Decisão final

**A amostragem balanceada por sujeito não é promovida** a nenhuma
configuração canônica A/B/C, e a intervenção exploratória fica encerrada. A
evidência atual não justifica:

- acrescentar a amostragem balanceada por sujeito à receita de treino. Pela
  invariante 1, isso exigiria a mesma mudança em A, B e C e o retreino dos
  três;
- buscar um efeito com mais seeds, outras probabilidades de sorteio,
  combinação com balanceamento por classe ou outros braços;
- avaliar a intervenção no `val` ou no `test` oficiais.

As decisões congeladas continuam valendo: receita idêntica entre A, B e C,
checkpoint da última época, splits oficiais, runs canônicos e
`runs/reference/`. Os runs e o resumo da Etapa 5A ficam em
`runs/local/le2i/subject_folds/`, não versionado, e nenhum deles é
promovido. O `test` oficial, já exposto, não foi lido nem entrou em nenhuma
seleção.

### Critério congelado e resultado

O [critério](../train/subject-fold-subject-balanced-sampling.md#criterio-exploratorio-congelado),
fixado em `DECISION_RULE` e commitado antes dos treinos, exige delta pareado
de Macro-F1 restrita no retido (balanceado menos controle) de pelo menos
+0,03 em pelo menos duas das três dobras e nenhum delta abaixo de −0,02 na
dobra restante. Os deltas não são agregados entre dobras.

| Dobra | Controle (4B) | Balanceado (5A) | Δ |
| --- | ---: | ---: | ---: |
| `retain_s0` | 0,5584 | 0,5416 | −0,0168 |
| `retain_s1` | 0,6576 | 0,6797 | +0,0221 |
| `retain_s3-4-6-8` | 0,6157 | 0,6163 | +0,0006 |

**O critério não foi atendido.** Nenhuma dobra chega a +0,03. O menor delta,
−0,0168, fica dentro do limite de −0,02, então a intervenção falha pela
ausência de ganho, não por uma piora que ultrapasse a tolerância
([execução registrada](../train/subject-fold-subject-balanced-sampling.md#execucao-registrada-nao-canonica)).

### Observações

1. **Os deltas não têm direção consistente.** Um é negativo, um é positivo e
   um é praticamente nulo. O de `retain_s0`, a dobra cujo treino perde o
   sujeito dominante, é negativo. O único delta positivo relevante,
   +0,0221 em `retain_s1`, vem de uma dobra que retém um único sujeito e
   ainda fica abaixo do limiar de +0,03.
2. **Os efeitos por classe são inconsistentes.** Δ F1 de `standing` é
   −0,048, +0,138 e −0,009; o de `other`, +0,012, +0,031 e −0,016; e o de
   `fall`, −0,000, −0,010 e +0,015, na ordem `retain_s0`, `retain_s1`,
   `retain_s3-4-6-8`. Nenhuma das três classes muda na mesma direção nas
   três dobras. O maior delta, de `standing` em `retain_s1`, vem de 70
   janelas de cinco vídeos de um único sujeito.
3. **A amostragem mudou também a distribuição de classes do treino.** Por
   exemplo, em `retain_s1`, `other` passou de 1052 janelas limpas para cerca
   de 1332 sorteadas por época, e `sit_down`, de 121 para cerca de 62. O
   efeito observado não isola o balanceamento por sujeito dessa mudança.

### Limitações do encerramento

- O critério é uma regra de decisão predeclarada, não um teste de
  significância. Nenhuma afirmação de significância estatística é feita, e o
  alvo de 0,800 não é verificado.
- Uma seed de treino (42) por dobra e modelo. A variação entre seeds não é
  estimada, e deltas desta ordem podem vir só dela.
- As três dobras têm tamanhos muito diferentes, duas retêm um único sujeito,
  e em `retain_s3-4-6-8` só há dois sujeitos de desenvolvimento. Nas outras
  dobras, o sujeito com menos janelas é repetido várias vezes por época, e o
  efeito mistura menos domínio de poucos sujeitos com mais sobreajuste aos
  vídeos dos sujeitos pequenos.
- Os pesos de classe continuam os do treino limpo completo e não compensam a
  mudança na distribuição de classes sorteada.
- As janelas são sobrepostas e não são observações independentes.
- O resultado vale para esta receita, este braço, esta seed e esta regra de
  sorteio. Ele não mostra que o balanceamento por sujeito não possa
  funcionar em outras condições, como outro esquema de dobras, mais sujeitos,
  outros braços ou outra intensidade de reamostragem.

## Encerramento dos erros `standing→walk` e `sitting→stand_up` (Etapas 6A–6B) { #encerramento-etapas-6a-6b }

O [diagnóstico da Etapa 6A](standing-sitting-diagnosis.md) examinou as
predições de desenvolvimento, com o `val` oficial (A, B0, B1, C0 e C1,
seeds 42–46) separado dos grupos retidos das dobras por sujeito do `train`
oficial (braço A, seed 42). As regras predeclaradas não sustentaram uma causa
comum para as confusões, e o veredito dos dois pares foi
`unresolved_limitation`. As taxas do `val` não foram agregadas às das dobras;
nenhuma predição do `test` foi usada nesta decisão.

A [revisão da Etapa 6B](standing-sitting-review-set.md#revisao-preliminar-por-ia-concluida)
leu 54 janelas selecionadas: 11 do `val` oficial (7 erros, 4 controles) e
43 das dobras (23 erros, 20 controles). O ChatGPT registrou os julgamentos
antes de receber a chave, mas já conhecia a cobertura geral da PR #125. Após
o desmascaramento, os 54 IDs coincidiram com a chave; havia 24 pares
válidos e seis erros sem controle. Nos sete pares `standing→walk`, todos das
dobras, caminhada plausível apareceu em dois erros e dois controles, e
ambiguidade de fronteira em três erros e um controle. Nos 17 pares
`sitting→stand_up` (quatro do `val`, 13 das dobras), nenhum item foi julgado
visualmente plausível como `stand_up`; alguns controles não tinham
informação temporal suficiente para julgamento. A leitura é preliminar por
IA, parcialmente mascarada, não representativa e sem validação humana
independente. Essas diferenças não são testes de significância nem provas de
causalidade ou de erro nos rótulos.

**Decisão final:** encerrar a Etapa 6B sem intervenção. Os dois erros seguem
como limitação de generalização não resolvida. A evidência não justifica
retreino, ajuste de janela ou limiar, alteração de anotações ou promoção de
artefatos. O alvo de Macro-F1 restrita ≥0,800 não foi verificado. Receita,
splits, checkpoints, protocolo de alarme e resultados canônicos permanecem
congelados; o `test` já exposto não pode selecionar uma correção.

## Limitações

- As frações e revocações são por janela com `eval_stride` 1, sobre janelas
  sobrepostas.
- Le2i-CS `val` tem um sujeito, e Le2i-CV `val` tem um ambiente. Os dois
  blocos nunca são agregados.
- As predições estratificadas vêm de reexecuções validadas. A igualdade
  agregada com `metrics.json` não prova identidade janela a janela com as
  predições históricas.
- O histórico por época mostra picos de `val` antes da época 30, mas o
  máximo de 30 valores ruidosos é enviesado para cima. Isso não justifica
  mudar a seleção pela última época.
