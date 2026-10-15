# Análise — Auditoria dos ambientes retidos das dobras internas (Etapa 2E)

`src/gatefall/eval/analysis/internal_fold_retained_audit.py` investiga o
colapso de B0 e B1 para `walk` no ambiente retido `Lecture_room`, registrado
no [piloto da Etapa 2D](internal-fold-pilot.md#diagnostico-de-lecture_room).
A pergunta é se a evidência existente aponta para um problema de integridade
da entrada ou para uma mudança de distribuição, sem tratar nenhuma das duas
como causa.

A auditoria é **exploratória, não canônica e somente leitura**. Ela não
retreina, não altera checkpoints, arquiteturas, receitas nem limiares, não lê
`val` nem `test` oficiais e não grava nada nos runs, nas estatísticas por
dobra ou nas referências.

## Conclusão

- **Defeitos técnicos verificados: nenhum.** Pose, DINOv3 e qualidade dos 97
  vídeos do `train` oficial passam nas checagens de forma, valores finitos,
  quadros zerados, variação temporal e faixa. As janelas rotuladas do
  ambiente retido batem com `folds.json` nas três dobras, e a reinferência
  dos nove runs reproduz exatamente a matriz de `final.val` de cada
  `metrics.json`.
- **Deslocamento descritivo: só DINOv3, só em `Lecture_room`.** Com as
  estatísticas da própria dobra, todos os 27 vídeos de `Lecture_room` ficam
  acima de todos os vídeos de desenvolvimento na RMS do z médio DINOv3, e 23%
  dos valores padronizados, na mediana por vídeo, passam de |z| = 5. Nos
  outros dois ambientes retidos, o z DINOv3 fica na faixa do
  desenvolvimento. A pose padronizada de `Lecture_room` fica na faixa do
  desenvolvimento, embora a confiança dos keypoints e `q_pose` sejam menores
  nesse ambiente.
- **O colapso coincide com esse deslocamento.** `Lecture_room` é a única
  dobra em que B0/B1 predizem `walk` em quase todos os vídeos, e a única em
  que a entrada DINOv3 sai da faixa do desenvolvimento. Com três dobras e uma
  seed, essa coincidência não estabelece causa.

## O que é reaproveitado

- **Composição e proveniência.** As dobras vêm de `build_fold_partitions` e
  são reconciliadas com `folds.json`, inclusive `input_sha256` e o hash dos
  sidecars de qualidade. As estatísticas de cada dobra passam por
  `load_fold_stats`, que confere a proveniência contra a composição e os
  hashes.
- **Runs.** Cada run passa pela mesma validação do resumo das dobras
  (`run_entry`): configuração determinística, `fold_run.json` ligado aos
  artefatos, `final.val` igual a `final.test` e contagens de janelas.
- **Features.** Os HDF5 de DINOv3 e de qualidade são conferidos com
  `validate_feature_files` só para os vídeos do `train` oficial, e as
  features vêm de `feature_loaders`, os mesmos loaders do treino das dobras.
- **Inferência e alinhamento.** As predições usam os preditores dos
  avaliadores de eventos de A, B0 e B1 sobre o checkpoint congelado, com as
  estatísticas da dobra e loaders restritos aos vídeos retidos. Cada predição
  é ligada à janela exata por `window_metadata` e `align_predictions` do
  [diagnóstico estratificado](stratified-error-diagnosis.md).

## O que é medido

**Integridade, por vídeo.** Forma `(K, dimensão)` de pose (134), DINOv3
(1536) e qualidade (2) contra o `K` da tabela de quadros; valores não
finitos; quadros DINOv3 inteiramente zerados; DINOv3 constante no tempo;
qualidade fora de [0, 1]. Também registra, só como descrição, a confiança
média dos keypoints, a norma L2 e o desvio temporal médio do DINOv3 e a média
de `q_pose` e `q_visual`.

**Alinhamento de janelas e rótulos.** Janelas rotuladas do retido em stride
1 contra `retained_windows_eval_stride` de `folds.json`. A reinferência cobre
todas as janelas, inclusive as `IGNORE_LABEL`, e a matriz das janelas
rotuladas precisa ser igual à `confusion_matrix` de `final.val`.

**Padronização por dobra.** Para cada vídeo, com as estatísticas da dobra:

- RMS, sobre as dimensões padronizadas, do z médio do vídeo;
- fração dos valores padronizados com |z| > 5.

A pose exclui o bloco `kp_conf`, que não é padronizado, e as dimensões
protegidas por desvio quase nulo; nenhuma dimensão foi protegida nas três
dobras. O relatório também traz os quantis do desvio-padrão ajustado.

**Geometria DINOv3 bruta.** Sem estatística de dobra: distância de cosseno
entre os centróides dos ambientes (média das médias por vídeo) e, por vídeo,
a distância ao próprio ambiente (deixando o vídeo fora) e ao ambiente mais
próximo.

**Predições por vídeo.** Para cada run, janelas rotuladas, classes preditas,
fração predita como `walk` e acurácia de cada vídeo retido.

Todos os resumos são por vídeo: mediana, mínimo e máximo. Frações por janela
aparecem só como contexto, porque janelas sobrepostas não são observações
independentes. Os limiares |z| > 5 e fração `walk` ≥ 0,9 só contam valores e
vídeos; não definem defeito nem selecionam nada.

## Resultado da execução real

Executado em 2026-10-14 sobre os nove runs do piloto (seed 42), com
inferência em CUDA. `official_splits_read: ["train"]`. O SHA-256 de frames,
manifesto, `folds.json`, estatísticas e artefatos dos runs foi calculado
antes e conferido depois.

### Integridade e alinhamento

| Ambiente | Vídeos | Com defeito | Conf. keypoints | `q_pose` | Norma DINOv3 | Desvio temporal DINOv3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Coffee_room_01` | 48 | 0 | 0,796 | 0,637 | 16,65 | 0,069 |
| `Coffee_room_02` | 22 | 0 | 0,794 | 0,657 | 16,63 | 0,073 |
| `Lecture_room` | 27 | 0 | 0,641 | 0,487 | 17,54 | 0,075 |

Medianas por vídeo. Os HDF5 de DINOv3 dos três ambientes têm a mesma
proveniência: modelo (`dinov3_vitb16`), pesos, commit do repositório e
pré-processamento (`resize_height` e `resize_width` iguais a 224, mais a
normalização do ImageNet; ver [features DINOv3](../data/dinov3-features.md#pre-processamento)).
Os atributos `width`, `height` e `fps` dos mesmos HDF5 registram 320×240 a
25 fps nos 97 vídeos. Essa é a resolução do quadro decodificado do vídeo de
origem, não a de entrada do DINOv3, que recebe o quadro redimensionado para
224×224.

As janelas rotuladas do retido são 5731, 5176 e 5688 nas dobras
`Coffee_room_01`, `Coffee_room_02` e `Lecture_room`, iguais a `folds.json`.
As nove reinferências reproduzem a matriz de `metrics.json` sem nenhuma
célula divergente. Em `Lecture_room`, B0 prediz `walk` em 5549 janelas,
`other` em 100, `stand_up` em 37 e `fall` em 2; B1 prediz `walk` em 5680,
`other` em 5 e `fall` em 3, como no piloto.

### Padronização por dobra

RMS do z médio por vídeo: mediana (mínimo–máximo).

| Dobra | Lado | Vídeos | Pose | DINOv3 | Fração DINOv3 com \|z\| > 5 (mediana) |
| --- | --- | ---: | --- | --- | ---: |
| `retain_Coffee_room_01` | desenvolvimento | 49 | 0,287 (0,116–0,718) | 0,850 (0,780–0,994) | 0,0000 |
| `retain_Coffee_room_01` | retido | 48 | 0,457 (0,234–1,041) | 0,892 (0,829–0,973) | 0,0000 |
| `retain_Coffee_room_02` | desenvolvimento | 75 | 0,316 (0,128–0,861) | 0,852 (0,774–0,975) | 0,0000 |
| `retain_Coffee_room_02` | retido | 22 | 0,284 (0,106–0,582) | 0,836 (0,793–1,111) | 0,0000 |
| `retain_Lecture_room` | desenvolvimento | 70 | 0,340 (0,081–0,850) | 0,557 (0,373–1,097) | 0,0000 |
| `retain_Lecture_room` | retido | 27 | 0,260 (0,164–0,546) | 4,208 (4,072–4,418) | 0,2323 |

A mediana do desvio-padrão DINOv3 ajustado é 0,138 e 0,133 nas dobras que
retêm uma sala de café e 0,058 na dobra `Lecture_room`, cujo desenvolvimento
tem só as duas salas de café.

### Geometria DINOv3 bruta

| Par de ambientes | Distância de cosseno entre centróides |
| --- | ---: |
| `Coffee_room_01` – `Coffee_room_02` | 0,0031 |
| `Coffee_room_01` – `Lecture_room` | 0,5534 |
| `Coffee_room_02` – `Lecture_room` | 0,5487 |

Por vídeo, `Lecture_room` fica a 0,019 (0,013–0,029) do próprio ambiente e a
0,559 (0,522–0,586) do mais próximo. Os vídeos das salas de café ficam tão
perto da outra sala quanto da própria: 0,010 contra 0,013 em
`Coffee_room_01` e 0,014 contra 0,013 em `Coffee_room_02`.

### Predições por vídeo no ambiente retido

Fração predita como `walk` por vídeo: mediana (mínimo–máximo).

| Dobra | Arma | Fração `walk` por vídeo | Vídeos com fração ≥ 0,9 |
| --- | --- | --- | ---: |
| `retain_Coffee_room_01` | A | 0,242 (0,000–0,667) | 0/48 |
| `retain_Coffee_room_01` | B0 | 0,325 (0,000–0,773) | 0/48 |
| `retain_Coffee_room_01` | B1 | 0,322 (0,000–0,755) | 0/48 |
| `retain_Coffee_room_02` | A | 0,449 (0,058–0,595) | 0/22 |
| `retain_Coffee_room_02` | B0 | 0,480 (0,081–0,773) | 0/22 |
| `retain_Coffee_room_02` | B1 | 0,473 (0,081–0,761) | 0/22 |
| `retain_Lecture_room` | A | 0,203 (0,126–0,440) | 0/27 |
| `retain_Lecture_room` | B0 | 1,000 (0,884–1,000) | 26/27 |
| `retain_Lecture_room` | B1 | 1,000 (0,966–1,000) | 27/27 |

O colapso não vem de poucos vídeos longos: atinge todos os vídeos de
`Lecture_room` em B1 e todos menos um em B0. Nenhum vídeo das outras dobras
passa de 0,9 em nenhuma arma. A fração de janelas rotuladas `walk` é 0,379,
0,287 e 0,492 nos três ambientes, e `walk` aparece nos 27 vídeos de
`Lecture_room`.

## Hipóteses não resolvidas

Nada nesta etapa separa as leituras abaixo. Elas não são resultados.

- **Mudança de entrada visual.** B0/B1 podem recair em `walk`, a classe mais
  frequente do treino de desenvolvimento dessa dobra, quando a entrada DINOv3
  padronizada sai da faixa vista no treino. A auditoria só mostra que o
  deslocamento e o colapso coexistem.
- **Variância visual estreita no desenvolvimento.** As duas salas de café
  são quase indistinguíveis no DINOv3. Por isso o desvio-padrão ajustado na
  dobra `Lecture_room` é menor, e o z de um ambiente novo cresce. É uma
  propriedade da padronização por dobra, não um defeito.
- **Atalho de cena.** O DINOv3 pode codificar a cena junto da postura, e
  B0/B1 quase memorizam o treino de desenvolvimento. Nada aqui mede quanto da
  decisão vem da cena.
- **Pose de menor qualidade.** `Lecture_room` tem confiança de keypoints e
  `q_pose` menores, mas a pose padronizada fica na faixa do desenvolvimento e
  A não colapsa no mesmo ambiente.
- **Composição e seed.** O retido muda sujeitos, câmera e proporção de
  classes ao mesmo tempo, e o piloto tem uma seed por arma.

Separar a primeira leitura das demais exigiria uma intervenção na inferência,
como substituir a entrada visual de `Lecture_room` no checkpoint congelado.
Ela não foi executada. Seria uma etapa nova, que exige aprovação explícita e
precisa declarar o que desloca.

## Saídas

`--output-dir` recebe:

- `internal_fold_retained_audit.json`: integridade por vídeo e por ambiente,
  composição de rótulos, geometria DINOv3, padronização por dobra e por
  vídeo, reconciliação e predições por vídeo de cada run, achados separados
  em `verified_defects`, `descriptive_shifts`, `prediction_observations` e
  `unresolved_hypotheses`, limitações e `input_sha256`;
- `internal_fold_retained_audit.md`: o mesmo conteúdo em resumo legível.

Os textos de `descriptive_shifts` e `prediction_observations` são gerados a
partir dos números. As hipóteses e limitações são texto fixo.

## Garantias de somente leitura

- Só linhas `split == train` de frames e manifesto são carregadas, e os
  loaders da inferência recusam qualquer vídeo fora do ambiente retido da
  dobra.
- `--output-dir` não pode coincidir com, conter ou ficar dentro do diretório
  das dobras, de `runs/reference/`, dos runs locais oficiais, de `src/`, de
  `data/` ou dos diretórios das entradas.
- Sem `--force`, uma saída existente é preservada antes de qualquer cálculo.
- O SHA-256 de frames, manifesto, `folds.json`, estatísticas por dobra e
  `config.yaml`, `metrics.json`, `checkpoint.pt` e `fold_run.json` de cada run
  é conferido depois da gravação.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.internal_fold_retained_audit selftest
uv run python -m gatefall.eval.analysis.internal_fold_retained_audit audit \
  --output-dir runs/local/internal_fold_retained_audit \
  [--fold-root runs/local/le2i_cv_internal_folds] [--frames PATH] [--manifest PATH] [--force]
```

O `selftest` usa dobras e runs sintéticos treinados com o protocolo das
dobras. Ele cobre a detecção de cada defeito de integridade, a reconciliação
só das janelas rotuladas, a diferença entre fração por janela e por vídeo, a
leitura restrita ao `train`, a marcação de um deslocamento exclusivo do
retido, a preservação dos runs e da saída existente e a recusa de saída
protegida, `fold_run.json` adulterado e `folds.json` inconsistente. `audit`
requer o piloto já treinado; GPU é recomendada.

## Limitações

- Exploratório e não canônico. Nada aqui explica a falha observada no `test`
  oficial.
- As estatísticas são ajustadas no próprio desenvolvimento da dobra, então os
  vídeos de desenvolvimento são uma referência dentro da amostra. A
  comparação justa entre ambientes retidos é entre dobras.
- A reconciliação com `metrics.json` é necessária, mas não prova que cada
  janela recebeu o mesmo rótulo da avaliação histórica, que nunca foi
  persistida por janela.
- As checagens de integridade cobrem os arrays de feature. Elas não
  inspecionam os quadros de vídeo de origem.
- Três ambientes e uma seed por arma.
