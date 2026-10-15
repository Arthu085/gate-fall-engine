# Análise — Síntese diagnóstica entre braços

!!! warning "Síntese exploratória e não canônica"
    Esta página reúne, para A, B0, B1, C0 e C1, a evidência já publicada sobre
    por que os resultados do GateFall continuam limitados. Não houve treino,
    reextração, ajuste, seleção nem promoção de artefato. Os números marcados
    com `[D..]` foram derivados nesta síntese, por leitura, de relatórios e
    dados locais já existentes ([como foram obtidos](#numeros-derivados));
    os marcados com `[E..]` vêm de páginas publicadas. O split `test` já
    estava exposto e é usado só de forma descritiva.

Salvo indicação, os valores são de Le2i-CS, média entre as seeds 42–46, por
janela com `eval_stride` 1. Le2i-CS e Le2i-CV nunca são agregados.

## Resumo

- **O erro dominante é compartilhado.** Nos cinco braços, `other` participa
  de 49% a 57% dos erros de `test`, e `other→walk` leva 37% a 41% do suporte
  de `other`. `walk` é o destino de 31% a 39% de todos os erros `[D1]`.
- **Mais da metade do `standing` de `test` não tem pose detectada.** Das 43
  janelas, 23 terminam no prefixo de `home_01/video_15` antes da primeira
  detecção, sem pose em nenhum dos 24 quadros, e todas são preditas como
  `walk` nos 25 runs (cinco braços × cinco seeds) `[D1]` `[D3]`. Isso limita a revocação de `standing` em `test`
  a no máximo 20/43 em qualquer braço com o comportamento atual.
- **O suporte de avaliação cobre pouco as duas classes mais custosas.**
  `standing` de `test` vem de 3 vídeos, todos do mesmo sujeito; `other` de
  `val` tem 34 janelas de 3 vídeos, contra 860 em `test` `[D2]`.
- **Os padrões específicos de fusão são menores que os compartilhados.**
  B0/B1 quase não predizem `standing` em `val`. Nos diagnósticos agregados
  por classe, B1 tem padrões de erro próximos aos de B0 e C1, aos de C0; essas
  comparações não isolam um padrão de erro atribuível ao gate, mas também
  não medem a relação entre gate e erro janela a janela `[D1]` `[E14]`
  `[E18]`.
- **A explicação de `other→walk` continua em aberto.** A investigação
  recomendada é uma atribuição, por janela e sem treino, do contexto desses
  erros ([seção própria](#investigacao-recomendada)).

## Padrões compartilhados por todos os braços

`test` de Le2i-CS `[D1]`:

| Padrão | A | B0 | B1 | C0 | C1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Erros com `other` verdadeiro ou predito | 49% | 56% | 57% | 56% | 56% |
| `other→walk` (fração do suporte de `other`) | 38% | 41% | 41% | 37% | 38% |
| Revocação de `other` | 0,36 | 0,48 | 0,49 | 0,50 | 0,47 |
| Erros com predição `walk` | 31% | 36% | 38% | 39% | 38% |
| Revocação de `standing` (43 janelas) | 0,10 | 0,10 | 0,13 | 0,07 | 0,03 |
| `standing→walk` (fração do suporte) | 62% | 70% | 63% | 73% | 70% |

Observações verificadas:

1. **`standing` e `other` respondem por 49% a 50% do déficit da Macro-F1** em
   todos os braços `[E7]`. Só `standing`, com F1 entre 0,02 e 0,14, custa de
   0,11 a 0,12 ponto de Macro-F1 a cada braço, porque cada classe restrita
   pesa 1/8.
2. **`other→walk` é sistemático.** É a confusão dominante nas cinco seeds de
   todos os braços e se concentra em `Lecture_room`, onde leva de 43% a 55%
   do suporte de `other` `[E16]`.
3. **O `standing` sem pose é errado da mesma forma por todos.** As 23
   janelas de `standing` de `home_01/video_15` terminam nos quadros 0–22, e
   a primeira detecção de pose do vídeo é no quadro 27. Nos 25 runs, as 23
   vão para `walk`, inclusive em B0, B1, C0 e C1, que também recebem entrada
   visual `[D1]` `[D3]`. No `train` de Le2i-CS, os quadros rotulados de
   prefixo sem pose são 28 `walk` e 1 `sitting`; os outros 985 são
   `IGNORE_LABEL` `[D3]`.
4. **Os outros dois vídeos de `standing` divergem entre braços.** Em
   `office/video_31` (14 janelas), o destino principal é `other` em todos os
   braços. Em `coffee_room_02/video_68` (6 janelas, início do vídeo), B0 e
   B1 acertam a maior parte; A, C0 e C1 erram mais `[D1]`.
5. **Em `val`, `standing→walk` e `sitting→stand_up` dominam em todos os
   braços** `[E7]` `[E19]`.

## Padrões específicos de fusão visual e do gate

| Padrão | Braços | Evidência | Leitura |
| --- | --- | --- | --- |
| Quase não predizem `standing` em `val` de Le2i-CS: 6 e 16 janelas preditas, contra 60 a 90 em A, C0 e C1; revocação 0,06 e 0,11 contra 0,34 a 0,38 | B0, B1 | `[D1]` `[E19]` | Verificado; suporte de um sujeito |
| Treino quase memorizado (Macro-F1 > 0,97 contra 0,87 em A) | B0, B1, C0, C1 | `[E6]` `[E19]` | Verificado; C0/C1 memorizam sem a falha de B0/B1 em Le2i-CV |
| Mais predições de `other` em `val` de Le2i-CS: 106 e 113 janelas fora de `other` preditas como `other`, contra 85 em A e 21 a 26 em B0/B1; precisão de `other` 0,14 e 0,16, como A (0,16) | C0, C1 | `[D1]` `[E19]` | Verificado; diferença pequena em relação a A, e `val` tem só 34 janelas `other` |
| Superpredição de `standing` em `test` de Le2i-CS: 223 janelas preditas para 43 verdadeiras; das 219 incorretas, 64% são `other` | A | `[D1]` | Verificado |
| Deslocamento para `other` em `test` de Le2i-CV e colapso para `walk` em `Lecture_room` nas dobras internas | B0, B1 | `[E16]` `[E21]` | Verificado; causa não atribuída (Etapa 2G) |
| Revocação de `standing` 0 em `test` de Le2i-CV nas cinco seeds | B0, B1 | `[D1]` | Verificado |
| `g_pose` médio perto de 0,50 em todas as condições de degradação; intervalos pareados B1 − B0 contêm zero na seed 42; frações por classe em `test` próximas às de B0 | B1 | `[D1]` `[E14]` `[E18]` | Nenhum padrão próprio detectado nas métricas agregadas; não mostra que B1 calcule a mesma função que B0 |
| `g_pose` médio reage à pose, mas também cai com a degradação do SAM 3; C1 − C0 é misto entre `val` e `test`; frações por classe próximas às de C0 | C1 | `[D1]` `[E6]` `[E18]` | Nenhum padrão próprio detectado nas métricas agregadas |

Nas métricas agregadas examinadas, nenhum padrão específico de fusão ou de
gate tem peso comparável ao de `other→walk` e de `standing` no déficit de
Le2i-CS, e o ganho médio dos braços de fusão sobre A não altera os erros
dominantes `[E6]` `[E7]`. A ausência de padrão próprio do gate vale só para
esses agregados: o gate foi medido como média sob degradação, em Le2i-CS e
na seed 42, e não há medida da concordância janela a janela entre B1 e B0
nem da relação entre o valor do gate e os erros `[E18]` `[E19]`.

## Limitações de suporte e composição

Contagens de rótulo, sem modelo. `train` em stride 4 e `val`/`test` em
stride 1; trechos são sequências contíguas do mesmo rótulo na grade `[D2]`:

| Classe | Split | Janelas | Fração das rotuladas | Vídeos | Trechos | Duração mediana do trecho |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `standing` | `train` | 173 | 3,3% | 22 | 28 | 1,8 s |
| `standing` | `val` | 96 | 4,6% | 4 | 6 | 1,2 s |
| `standing` | `test` | 43 | 0,8% | 3 | 3 | 1,4 s |
| `other` | `train` | 1170 | 22,4% | 57 | 95 | 2,4 s |
| `other` | `val` | 34 | 1,6% | 3 | 3 | 0,9 s |
| `other` | `test` | 860 | 15,3% | 17 | 30 | 2,6 s |

- **`standing` de `test` é um caso, não uma classe.** As 43 janelas vêm de
  três trechos menores que a janela de 2,4 s, todos do sujeito 7, e 29 delas
  ficam no início do vídeo, com replicação de borda `[D2]`. A pose é
  detectada no quadro final de 20 das 43 janelas (46,5%). Nos quadros
  `standing` de `train` e `val`, a detecção é de 673/673 e 96/96 `[D3]`.
- **`val` cobre pouco `other`.** São 34 janelas de três trechos com mediana
  de 0,9 s, em três vídeos de `Home_01` e `Home_02`, sem `Coffee_room_01`.
  Com esse suporte, o split de desenvolvimento dá uma estimativa limitada e
  pouco representativa do erro que domina `test`, que tem 860 janelas em 17
  vídeos `[D2]`.
- **`other` de `test` concentra-se em um par (ambiente, sujeito).**
  `Lecture_room` do sujeito 7 tem 491 das 860 janelas (57%), e o maior vídeo
  isolado, `office/video_25`, tem 139 (16%) `[D2]`.
- **A unidade de avaliação é pequena.** `val` tem um sujeito e 13 eventos;
  `test` tem dois sujeitos e 22 eventos; os alarmes de `val` saturam nos
  quatro braços de fusão `[E9]` `[E10]` `[E14]`.
- **Le2i-CV confunde fatores.** Ambiente, câmera, resolução, fps e
  qualidade de pose mudam juntos, e os sujeitos se sobrepõem `[E2]`. O mesmo
  `home_01/video_15` também está no `test` de Le2i-CV.
- **Contexto de pose.** No `test` de Le2i-CS, a detecção também é menor nos
  quadros `fall` (386/455, 84,8%) e `fallen` (335/442, 75,8%) que nos de
  `train` (1815/1993, 91,1%, e 2250/2483, 90,6%) `[D3]`. Nada aqui mede o
  efeito disso nos alarmes.

## Explicações plausíveis

Nenhuma delas é uma causa estabelecida.

| Explicação | A favor | Contra ou ausente | Situação |
| --- | --- | --- | --- |
| Janelas de `standing` sem pose são lidas como `walk` porque o único prefixo rotulado do treino é `walk` | 23/43 janelas, 25/25 runs, prefixo de treino com 28 quadros `walk` `[D1]` `[D3]` | Os braços de fusão também erram, apesar da entrada visual; os quadros de vídeo não foram inspecionados | Observação verificada; mecanismo plausível |
| Trechos curtos e rótulo pelo último quadro misturam o estado com o movimento anterior | Trechos de `standing` curtos em todos os splits; `standing→walk` em todos os braços `[D2]` | Em `test`, `fall` e `sit_down` têm mais da metade das janelas com o rótulo final em minoria e, mesmo assim, F1 de 0,70 e 0,66 no braço A `[D2]` `[E6]` | Plausível, enfraquecida |
| `other` é heterogêneo e parte dele se parece com caminhada | `other→walk` em todos os braços e seeds, concentrado em `Lecture_room` `[E16]` | Nenhuma leitura do conteúdo dos segmentos `other`; a definição da classe não está documentada no repositório | Não testada |
| `other→walk` se associa a janelas de contexto misturado | 48% dos trechos `other` de `train` são menores que 2,4 s `[D2]` | Nenhuma atribuição de erros por contexto | Não testada; [investigação recomendada](#investigacao-recomendada) |
| A fusão visual suprime classes de estado minoritárias | B0/B1 quase não predizem `standing` em `val` `[D1]` | B0/B1 acertam mais `coffee_room_02/video_68`; um sujeito em `val` | Não testada |
| O deslocamento de B0/B1 em Le2i-CV vem da entrada visual | Ver Etapas 2A–2G `[E19]` | Encerrada sem causa atribuída | Fora do escopo desta síntese |

## Evidência ausente e valor diagnóstico

| Checagem | Pergunta respondida | Valor esperado | Custo |
| --- | --- | --- | --- |
| Atribuição de contexto dos erros `other→walk` | O erro dominante se concentra em janelas de fronteira ou persiste no meio dos trechos? | Alto: indica se a próxima leitura deve priorizar o contrato de janela ou o conteúdo e os rótulos, nos cinco braços | Reexecução validada existente e uma saída por janela |
| Leitura manual de segmentos `other` em `Lecture_room` e `office/video_25` | O que a classe residual contém onde mais erra? | Alto, mas depende do resultado anterior para escolher os trechos | Manual; inspeção de vídeo |
| Inspeção dos quadros 0–22 de `home_01/video_15` | A pessoa está visível e de pé sem detecção? | Baixo: um vídeo, 23 janelas; não muda nenhuma decisão | Baixo |
| Falsos alarmes por rótulo verdadeiro no gatilho | Quais classes geram os alarmes falsos? | Médio para eventos; não explica a Macro-F1 | Baixo |
| Gate em Le2i-CV, distância DINOv3 por resolução, mais seeds | Causa da falha de B0/B1 em Le2i-CV | Baixo para esta pergunta; a Etapa 2G encerrou essa linha `[E19]` | Médio a alto |

## Investigação recomendada

**Uma única:** atribuir, por janela, o contexto dos erros `other→walk` de
Le2i-CS.

- **Escopo:** `val` e `test` de Le2i-CS, cinco braços e seeds 42–46, com
  as predições da reexecução validada do
  [diagnóstico estratificado](stratified-error-diagnosis.md), que hoje só
  guarda matrizes agregadas. Para cada janela, registrar rótulo final,
  predição, fração dos 24 quadros com o rótulo final, rótulo anterior,
  posição no trecho, replicação de borda e detecção de pose no último
  quadro. Resumir `other→walk` por essas variáveis; `standing` e
  `sitting→stand_up` entram só como comparação.
- **Leitura:** se `other→walk` se concentra em janelas misturadas ou de
  fronteira em todos os braços, isso sustenta uma associação entre o erro e
  o contexto da janela, que deve ser descrita como possível limitação da
  métrica. Não estabelece que o janelamento cause o erro: fronteiras também
  coincidem com transições de ação, posição no vídeo e decisões de
  anotação. Se o erro persiste em janelas puras, no meio do trecho, a
  associação com o contexto fica fraca, e a leitura manual de segmentos é o
  passo seguinte.
- **Limites:** não treina, não muda o contrato de janela nem a receita, não
  seleciona nada e não pode ser confirmada pelo `test` já exposto. Uma
  associação com a fronteira não autoriza mudar o janelamento congelado.
- **O que desloca:** ocupa o lugar de qualquer nova investigação de B0/B1,
  já encerrada na Etapa 2G, e não acrescenta braço, dataset, backbone nem
  métrica.

## Números derivados { #numeros-derivados }

Os relatórios e dados abaixo são locais e não versionados; o SHA-256 permite
conferir a cópia usada. Os cálculos não foram publicados como ferramenta.

| ID | O que foi calculado | Fonte |
| --- | --- | --- |
| D1 | Somas e frações da `confusion_matrix` de cada seed por split e por vídeo, depois média entre seeds: revocação, janelas preditas por classe, destino das janelas verdadeiras e parcela dos erros com `other` ou `walk` | `runs/local/multiseed_error_consolidation/stratified_seeds_42_46/stratified_error_diagnosis.json` (SHA-256 `aa5034a42e531d68ac10e5a8996ba8aa4ef4184cad05dcd104faafe157d669e2`), relatório de 50 runs da [consolidação multi-seed](multiseed-error-consolidation.md) |
| D2 | Janelas por `build_window_index` (stride 4 em `train`, 1 em `val`/`test`), vídeos, trechos contíguos por rótulo e fração da janela com o rótulo final por `window_frame_indices` | `data/processed/le2i/frames.parquet` (SHA-256 `2a563bb04d507f19732baf3bcda7aba7e51ba54b826aa3b0054f74cb67631a61`); o suporte de `train` coincide com `final.train.support` de `runs/reference/le2i/baseline_a/metrics.json` |
| D3 | `person_found` por quadro e primeiro quadro detectado (`load_person_found`, `first_observed_index`), cruzados com o rótulo de cada quadro da grade de D2 | HDF5 de pose por vídeo em `data/features/le2i/pose/<ambiente>/<vídeo>.h5` ([cobertura de pose](#cobertura-de-pose)) |

### Cobertura de pose { #cobertura-de-pose }

O achado de `home_01/video_15` vem de
`data/features/le2i/pose/home_01/video_15.h5` (SHA-256
`a2333fb95f8d00b2d49843f95095240d58d55feb5ae40101089eb0826959d9c8`). O
arquivo tem 99 linhas, igual ao `K` do vídeo na grade; `person_found` é falso
nos quadros 0–26 e verdadeiro no 27, e os quadros `standing` do vídeo são
0–22. Toda janela que termina nesses quadros cobre só quadros 0–22, por
isso nenhuma tem pose em qualquer posição.

As demais taxas de detecção usam os 190 HDF5 de pose. O SHA-256 do conjunto
é `3dc21a8fabce213917d3df649590dfd27fa2398d9c5738e90a2f2c4f1e22052e`,
reproduzível na raiz do repositório com:

```bash
find data/features/le2i/pose -name '*.h5' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

Agregação: cada quadro da grade de `frames.parquet` recebe o `person_found`
do mesmo índice no HDF5 do vídeo (o comprimento de cada `person_found` é
conferido contra `K`). A taxa de uma classe em um split é o número de
quadros dessa classe com `person_found` verdadeiro dividido pelo número de
quadros dessa classe, somados entre vídeos, sem média por vídeo:

| Classe | Split | Quadros com pose / quadros da classe |
| --- | --- | ---: |
| `standing` | `train` | 673 / 673 |
| `standing` | `val` | 96 / 96 |
| `standing` | `test` | 20 / 43 |
| `fall` | `train` | 1815 / 1993 |
| `fall` | `test` | 386 / 455 |
| `fallen` | `train` | 2250 / 2483 |
| `fallen` | `test` | 335 / 442 |

Em `val` e `test`, todo quadro rotulado é o fim de exatamente uma janela com
`eval_stride` 1, então a taxa por quadro é também a taxa no quadro final das
janelas avaliadas. Em `train`, as taxas contam todos os quadros rotulados da
grade, não só os fins de janela em stride 4 usados no treino.

O prefixo sem pose de um vídeo são os quadros com índice menor que
`first_observed_index`. As contagens de rótulo no prefixo de `train` (28
`walk`, 1 `sitting`, 985 `IGNORE_LABEL`) somam esses quadros em todos os
vídeos do split.

As predições de D1 vêm de reexecuções validadas, que reconciliam com
`metrics.json` mas não provam identidade janela a janela com as predições
históricas `[E5]`. Os IDs de sujeito não são globais; "sujeito 7" se refere
ao ID bruto nos ambientes de `test`.

## Mapa evidência → fonte

Caminhos relativos a `docs/`. Os IDs seguem os do
[rascunho de Resultados e Discussão](manuscript-results-discussion.md#mapa-evidencia-fonte).

| ID | Fonte |
| --- | --- |
| E2 | `analysis/le2i-cv-generalization.md` |
| E5 | `analysis/multiseed-error-consolidation.md`, `analysis/intervention-decision.md` |
| E6 | `analysis/multiseed-summary.md` |
| E7 | `analysis/classification-diagnosis.md` (Resultado da execução real) |
| E9 | `analysis/alarm-protocol-sensitivity.md`; `analysis/stratified-error-diagnosis.md` |
| E10 | `analysis/intervention-decision.md` (Evidência confirmada em `train` e `val`) |
| E14 | `analysis/grouped-bootstrap.md` |
| E16 | `analysis/multiseed-error-consolidation.md` (Histórico de execuções) |
| E18 | `analysis/gate-degradation.md` (Resultados reais no Le2i) |
| E19 | `analysis/intervention-decision.md` |
| E21 | `analysis/internal-fold-pilot.md` |

## Limitações

- Métricas por janela com `eval_stride` 1; janelas vizinhas se sobrepõem e
  não são observações independentes.
- O desvio entre seeds descreve inicialização e otimização, não incerteza
  entre sujeitos.
- `test` foi exposto pelos diagnósticos anteriores. Nada aqui seleciona
  modelo, seed, limiar ou protocolo.
