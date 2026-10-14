# Análise — Decisão sobre intervenções no modelo (Etapa 2A)

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

O único próximo passo justificável é o candidato 1 abaixo, um instrumento de
validação que não muda nenhum modelo. O candidato 2 só deve ser considerado se
o candidato 1 reproduzir o deslocamento para `other` em dados de
desenvolvimento.

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
   `standing→walk` e `sitting→stand_up` dominam em todas as armas. Isso aponta
   para o conteúdo ou os rótulos, não para o backbone, sem teste causal.
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
- Uma leitura de rótulo para `standing` e `other`, cujas confusões dominam o
  déficit de todas as armas.

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

## Resultado de não prosseguir

Se o candidato 1 não for aprovado, ou não reproduzir o deslocamento, a
decisão permanece **não**: nenhuma intervenção de modelo é justificada pela
evidência atual. O resultado de B0/B1 em Le2i-CV fica descrito como
limitação de generalização entre ambientes, sem causa atribuída, ao lado dos
erros compartilhados `standing→walk`, `sitting→stand_up` e `other→walk`.

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
