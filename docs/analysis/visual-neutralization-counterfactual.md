# Análise — Contrafactual de neutralização visual nas dobras internas (Etapa 2F)

`src/gatefall/eval/analysis/visual_neutralization_counterfactual.py` mede
quão sensíveis são as predições congeladas de B0 e B1 à entrada DINOv3 nos
ambientes retidos do [piloto da Etapa 2D](internal-fold-pilot.md), em
particular o colapso para `walk` em `Lecture_room` descrito na
[auditoria da Etapa 2E](internal-fold-retained-audit.md).

A análise é **exploratória, não canônica, somente leitura e só de
inferência**. Ela não retreina, não ajusta parâmetros, não altera
checkpoints, estatísticas, HDF5 de features, splits ou referências, não muda
o comportamento da inferência produtiva e não lê `val` nem `test` oficiais.

## Intervenção

Uma única intervenção fixa, definida antes da execução: para B0 e B1, o
tensor DINOv3 **já padronizado** com as estatísticas da dobra é substituído
por zeros. Em coordenadas padronizadas, zero é a média visual do treino de
desenvolvimento da dobra. Pose padronizada, proxies de qualidade crus
(`q_pose`, `q_visual`), pesos, checkpoints, rótulos, janelas, batch e
dispositivo ficam iguais. A, só pose, entra sem intervenção como referência.

A troca é feita por um invólucro do modelo congelado, aplicado só nesta
análise: os preditores dos avaliadores de eventos de B0 e B1 padronizam as
entradas como sempre, e o invólucro entrega ao modelo a mesma pose, a mesma
qualidade e `zeros_like` do tensor visual.

## Conclusão

- **As predições originais se reproduzem.** As nove reinferências da seed 42
  reproduzem exatamente a matriz de confusão e a Macro-F1 restrita de
  `final.val` de cada `metrics.json`. Sem essa reconciliação, o contrafactual
  não é calculado.
- **O colapso de `Lecture_room` responde à intervenção.** Com a entrada
  visual neutralizada, a fração de janelas preditas como `walk` cai de 0,976
  para 0,017 em B0 e de 0,999 para 0,022 em B1, e nenhum dos 27 vídeos fica
  com fração ≥ 0,9. As predições passam sobretudo para `other` e `fall`, e a
  Macro-F1 restrita vai de 0,096 para 0,167 (B0) e de 0,086 para 0,147 (B1),
  ainda muito abaixo de A (0,447).
- **Nas salas de café, a mesma intervenção destrói o desempenho.** A
  Macro-F1 restrita cai de 0,524–0,712 para 0,124–0,265, e as predições
  migram para `walk`. B0 e B1 dependem da entrada visual também onde não há
  colapso.
- **Isto é sensibilidade observada, não causa.** O resultado não estabelece
  atalho de cena nem mudança de domínio como causa do colapso, e a
  neutralização não é uma melhoria validada (ver
  [interpretação](#o-que-isto-nao-estabelece)).

## O que é reaproveitado

- **Dobras, estatísticas e runs.** Composição de `build_fold_partitions`
  reconciliada com `folds.json`, estatísticas por dobra de `load_fold_stats` e
  validação de cada run por `run_entry`, como na
  [auditoria da Etapa 2E](internal-fold-retained-audit.md#o-que-e-reaproveitado).
- **Features.** `validate_feature_files` e `feature_loaders`, restritos aos
  vídeos retidos da dobra.
- **Inferência.** A, sem intervenção, usa a mesma reinferência da Etapa 2E.
  B0 e B1 usam os preditores dos avaliadores de eventos nas duas condições,
  com o mesmo checkpoint e as mesmas estatísticas; só o invólucro muda.
- **Alinhamento e métricas.** `window_metadata` e `align_predictions` ligam
  cada predição à sua janela; a Macro-F1 restrita vem de
  `restricted_macro_f1`, a mesma do treino.

## O que é medido

Para cada dobra, arma e condição, sobre as janelas rotuladas do ambiente
retido em stride 1:

- classes preditas e fração predita como `walk`, por janela e por vídeo
  (mediana, mínimo e máximo), e vídeos com fração ≥ 0,9;
- matriz de confusão, revocação por classe e Macro-F1 restrita.

Para B0 e B1, o pareamento janela a janela entre original e contrafactual
registra as janelas que mudam de predição, as que passam a acertar ou a
errar, as transições entre classes e a fração alterada por vídeo. O limiar
0,9 só conta vídeos; não seleciona nada.

## Resultado da execução real

Executado em 2026-10-15 sobre os nove runs do piloto (seed 42), com
inferência em CUDA. `official_splits_read: ["train"]`. Duas execuções
seguidas gravaram saídas byte a byte idênticas.

| Dobra | Arma | Fração `walk` (janelas) | Fração `walk` por vídeo: mediana (mín–máx) | Vídeos ≥ 0,9 | Macro-F1 restrita | Janelas alteradas |
| --- | --- | --- | --- | --- | --- | ---: |
| `retain_Coffee_room_01` | A (referência) | 0,224 | 0,242 (0,000–0,667) | 0/48 | 0,612 | — |
| `retain_Coffee_room_01` | B0 | 0,398 → 0,607 | 0,325 → 0,606 (0,286–0,838) | 0/48 → 0/48 | 0,712 → 0,265 | 0,434 |
| `retain_Coffee_room_01` | B1 | 0,383 → 0,740 | 0,322 → 0,818 (0,400–1,000) | 0/48 → 9/48 | 0,685 → 0,150 | 0,574 |
| `retain_Coffee_room_02` | A (referência) | 0,378 | 0,449 (0,058–0,595) | 0/22 | 0,609 | — |
| `retain_Coffee_room_02` | B0 | 0,463 → 0,599 | 0,480 → 0,565 (0,191–0,791) | 0/22 → 0/22 | 0,524 → 0,193 | 0,471 |
| `retain_Coffee_room_02` | B1 | 0,495 → 0,810 | 0,473 → 0,803 (0,610–0,900) | 0/22 → 1/22 | 0,492 → 0,124 | 0,428 |
| `retain_Lecture_room` | A (referência) | 0,205 | 0,203 (0,126–0,440) | 0/27 | 0,447 | — |
| `retain_Lecture_room` | B0 | 0,976 → 0,017 | 1,000 → 0,000 (0,000–0,177) | 26/27 → 0/27 | 0,096 → 0,167 | 0,965 |
| `retain_Lecture_room` | B1 | 0,999 → 0,022 | 1,000 → 0,010 (0,000–0,156) | 27/27 → 0/27 | 0,086 → 0,147 | 0,978 |

Original → contrafactual. O intervalo por vídeo é o do contrafactual; o
original está na [auditoria](internal-fold-retained-audit.md#predicoes-por-video-no-ambiente-retido).

Em `Lecture_room`, B0 passa de `walk` 5549, `other` 100, `stand_up` 37 e
`fall` 2 para `other` 4506, `fall` 697, `fallen` 388 e `walk` 97. B1 passa
de `walk` 5680 para `other` 3040, `fall` 2361, `fallen` 163 e `walk` 124. A
revocação de `walk` cai de 0,996 para 0,034 (B0) e de 1,000 para 0,030 (B1),
enquanto `other`, `fall` e `fallen` passam a ser reconhecidos em parte.
`sit_down`, `sitting`, `stand_up` e `standing` continuam com revocação zero.

Nas salas de café, as transições dominantes vão para `walk` (por exemplo,
`fallen` → `walk` em 1066 janelas de B1 em `Coffee_room_01`), e a revocação
de `sitting`, `stand_up` e `sit_down` cai para zero ou quase zero nas duas
armas.

## O que isto não estabelece

- **Sensibilidade observada não é causa.** A intervenção mostra como as
  predições respondem quando a entrada visual vira a média do treino da
  dobra. Ela não estabelece que o colapso vem de um atalho de cena nem de
  mudança de domínio; essas leituras, e outras, preveem alguma sensibilidade
  à entrada visual.
- **A entrada neutralizada é fora da distribuição.** Um vetor visual
  constante, igual à média do desenvolvimento, em todos os passos da janela
  nunca ocorreu no treino. A resposta pode refletir essa entrada inédita, e
  não a remoção de informação de cena. A queda nas salas de café é
  compatível com isso.
- **Zero não é uma cena neutra.** É a média do treino de desenvolvimento da
  dobra; em `retain_Lecture_room`, essa média vem só das duas salas de café.
- **O gate de B1 não foi neutralizado.** Ele continua recebendo o
  `q_visual` original, que pode não combinar com a entrada visual zerada.
- **Não é uma melhoria validada.** Não há seleção, ajuste nem avaliação em
  `val` ou `test` oficiais. A Macro-F1 maior de B0/B1 em `Lecture_room` sob
  neutralização continua abaixo de A, que já é a referência sem entrada
  visual.

## Saídas

`--output-dir` recebe:

- `visual_neutralization_counterfactual.json`: intervenção, reconciliação,
  métricas das duas condições, pareamento e predições por vídeo de cada run,
  achados separados em `reconciliation`, `observed_sensitivity` e
  `interpretation_limits`, limitações, `input_sha256` e `feature_sha256`;
- `visual_neutralization_counterfactual.md`: resumo legível;
- `visual_neutralization_counterfactual_videos.csv`: uma linha por dobra,
  arma e vídeo retido, com as frações `walk`, acurácias e fração alterada; as
  colunas contrafactuais ficam vazias para A.

Os textos de `observed_sensitivity` são gerados a partir dos números; os
limites de interpretação e as limitações são texto fixo. O relatório não tem
carimbo de tempo, e a mesma entrada gera as mesmas saídas.

## Garantias de somente leitura e proveniência

- Só linhas `split == train` de frames e manifesto são carregadas; cada dobra
  só carrega e infere os vídeos do próprio ambiente retido.
- `--output-dir` não pode coincidir com, conter ou ficar dentro do diretório
  das dobras, de `runs/reference/`, dos runs locais oficiais, de `src/`, de
  `data/` ou dos diretórios das entradas. Sem `--force`, uma saída existente
  é preservada antes de qualquer cálculo.
- O SHA-256 dos arrays de pose, DINOv3 e qualidade de cada vídeo lido vai
  para `feature_sha256` e é recalculado com uma nova leitura depois do
  cálculo do relatório.
- O SHA-256 de frames, manifesto, `folds.json`, estatísticas por dobra e
  `config.yaml`, `metrics.json`, `checkpoint.pt` e `fold_run.json` de cada run
  é conferido em seguida.
- As duas verificações rodam antes da gravação. Qualquer diferença falha a
  execução sem publicar relatório, e uma saída existente fica intacta mesmo
  com `--force`.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.visual_neutralization_counterfactual selftest
uv run python -m gatefall.eval.analysis.visual_neutralization_counterfactual analyze \
  --output-dir runs/local/visual_neutralization_counterfactual \
  [--fold-root runs/local/le2i_cv_internal_folds] [--frames PATH] [--manifest PATH] [--force]
```

O `selftest` usa dobras e runs sintéticos treinados com o protocolo das
dobras. Ele cobre a reconciliação das predições originais e a recusa do
contrafactual quando a matriz ou a Macro-F1 divergem, a intervenção só no
DINOv3, a preservação de pose, qualidade, identidades e rótulos pelos
preditores validados, o pareamento das janelas, o isolamento por dobra, a
recusa de neutralizar A, a leitura restrita ao `train`, saídas idênticas ao
repetir, a preservação dos runs e da saída existente, a recusa de saída
protegida e a falha sem publicação, nem sobrescrita com `--force`, quando
features ou arquivos de entrada mudam antes da verificação final. `analyze`
requer o piloto já treinado; GPU é recomendada.

## Limitações

- Exploratório e não canônico. Nada aqui explica a falha observada no `test`
  oficial.
- Uma intervenção fixa e uma seed por arma.
- A reconciliação com `metrics.json` é necessária, mas não prova que cada
  janela recebeu o mesmo rótulo da avaliação histórica, que nunca foi
  persistida por janela.
- Três ambientes retidos, um por dobra.
