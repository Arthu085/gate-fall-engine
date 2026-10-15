# Análise — Viabilidade das dobras internas por sujeito do Le2i-CS (Etapa 4A)

`src/gatefall/eval/analysis/subject_fold_feasibility.py` avalia se o
`train` oficial do Le2i-CS sustenta uma validação interna agrupada por
sujeito. Essa validação serviria a uma tentativa controlada de melhorar a
macro-F1 restrita do Le2i-CS. O valor de 0,800 é um alvo de pesquisa, não um
critério para escolher modelos contra o `test` oficial, que já foi exposto.

A avaliação é **exploratória, não canônica e somente leitura**. Ela não
implementa o protocolo de dobras, não aumenta dados, não treina, não ajusta
estatística de padronização e não altera configurações A/B/C, protocolos,
limiares, checkpoints nem artefatos de referência.

## Entradas e contratos preservados

- Só as linhas `split == train` de `data/processed/le2i/frames.parquet` e
  `manifest.parquet` são carregadas, com filtro na leitura do Parquet, pela
  mesma função da [Etapa 2B](internal-fold-feasibility.md). Linhas de `val`
  e `test` oficiais nunca entram em memória.
- A partição precisa bater com o Le2i-CS pinado: seis ambientes, 133
  vídeos, 22246 janelas em stride 1 e 20740 rotuladas. Os vídeos de frames e
  manifesto precisam coincidir, sem vídeo repetido no manifesto.
- Cada vídeo precisa ter um único sujeito e um único ambiente nos frames, e
  os mesmos valores no manifesto. Sujeito ausente, fracionário ou negativo
  interrompe a execução sem gravar nada.
- As janelas vêm de `build_window_index` com `WINDOW_FRAMES`,
  `TRAIN_STRIDE` (4) e `EVAL_STRIDE` (1). Uma janela é ignorada quando o
  último quadro é `IGNORE_LABEL`.
- As classes avaliadas são as oito de `RESTRICTED_CLASSES`, as mesmas da
  macro-F1 restrita. Como `restricted_macro_f1` atribui F1 zero a uma classe
  sem nenhuma ocorrência, uma dobra com classe ausente não é comparável na
  macro-F1 de oito classes.

## Identidade dos sujeitos

Os IDs brutos de sujeito do Le2i não trazem garantia de identidade global.
O protocolo cross-subject do OmniFall separa `train`, `val` e `test` pelo ID
bruto: a verificação do manifesto mostra `train` com {0, 1, 3, 4, 6, 8},
`val` com {5} e `test` com {2, 7}. As análises do Le2i-CV, por outro lado,
usam o par (ambiente, sujeito). Os metadados não permitem decidir entre as
duas leituras.

As dobras agrupam pelo ID bruto, a unidade mais grossa disponível. Na leitura
global, o sujeito retido sai inteiro do treino. Agrupar por (ambiente,
sujeito) faria o contrário: um ID que aparece em vários ambientes ficaria nos
dois lados da dobra. Na leitura por ambiente, o grupo retido junta pessoas
possivelmente diferentes, todas fora do treino. Mesmo assim, uma delas pode
aparecer no treino com outro ID, risco que já existe no Le2i-CS oficial.

## Contrato proposto das dobras

Cada dobra divide os sujeitos brutos do `train` oficial em treino de
desenvolvimento e lado retido. Todo vídeo fica em exatamente um lado, e a
execução falha se algum vídeo ou sujeito aparecer nos dois lados ou ficar de
fora. Dois esquemas são avaliados:

- `leave_one_subject_out`: cada ID bruto é retido sozinho.
- `pooled_minority_subjects`: um sujeito fica sozinho quando suas janelas em
  stride 1 contêm as oito classes restritas e nenhuma delas vem de um único
  vídeo. Os demais formam um grupo retido único. Enquanto esse grupo não
  tiver as oito classes, o sujeito isolado com menos janelas rotuladas
  passa para ele, com empate decidido pelo menor ID. Menos de dois grupos
  não formam esquema.

O suporte de cada classe é medido no stride em que o lado é usado:
`TRAIN_STRIDE` no treino de desenvolvimento e `EVAL_STRIDE` no lado retido.
Cada dobra registra vídeos, sujeitos, ambientes, câmeras, janelas rotuladas
e ignoradas nos dois strides, suporte por classe em janelas, vídeos e
sujeitos, classes ausentes, classes de um único vídeo, classes de um único
sujeito (só em lados com mais de um sujeito) e riscos de vazamento. Uma
dobra só é comparável na macro-F1 de oito classes quando nenhuma classe
falta em nenhum dos lados. Um esquema sustenta a comparação controlada
quando tem pelo menos duas dobras e todas são comparáveis.

O veredito é `feasible_with_limitations` quando algum esquema sustenta a
comparação, e `not_feasible` caso contrário. Não existe veredito sem
limitações, porque a identidade global dos sujeitos não é verificável.

## Resultados observados

Execução registrada abaixo. Os 133 vídeos do `train` têm seis IDs brutos e
23 pares (ambiente, sujeito). Os IDs 0, 1, 3, 4 e 6 aparecem em mais de um
ambiente, e os IDs 0 e 1 aparecem nos seis.

| Sujeito bruto | Ambientes | Vídeos | Rotuladas stride 4 (fração) |
| ---: | ---: | ---: | ---: |
| 0 | 6 | 67 | 2581 (49%) |
| 1 | 6 | 31 | 1000 (19%) |
| 3 | 5 | 16 | 473 (9%) |
| 4 | 2 | 10 | 467 (9%) |
| 6 | 3 | 6 | 423 (8%) |
| 8 | 1 | 3 | 275 (5%) |

`leave_one_subject_out` tem 3 de 6 dobras comparáveis: as que retêm os
sujeitos 0, 1 e 6. O lado retido não tem `standing` no sujeito 3, não tem
`sit_down` nem `standing` no sujeito 4 e não tem quatro classes no sujeito
8. O sujeito 6 tem as oito classes, mas `fall`, `fallen`, `sit_down` e
`standing` vêm de dois vídeos cada.

`pooled_minority_subjects` isola os sujeitos 0, 1 e 6. Como {3, 4, 8} não
tem `standing`, o sujeito 6 passa para o grupo, e o resultado é
{0}, {1} e {3, 4, 6, 8}, com as três dobras comparáveis:

| Dobra | Treino dev.: sujeitos / vídeos / rotuladas stride 4 | Retido: vídeos / rotuladas stride 1 | Suporte frágil no retido |
| --- | --- | --- | --- |
| `retain_s0` | 5 / 66 / 2638 (51%) | 67 / 10243 | nenhum |
| `retain_s1` | 5 / 102 / 4219 (81%) | 31 / 3961 | `standing` em 5 vídeos |
| `retain_s3-4-6-8` | 2 / 98 / 3581 (69%) | 35 / 6536 | `standing` de um único sujeito (2 vídeos); `sit_down` em 3 vídeos de 2 sujeitos |

No treino de desenvolvimento, a classe com menos suporte é `standing`, na
dobra `retain_s0`: 46 janelas em stride 4, de 7 vídeos e 2 sujeitos. Todas as
dobras compartilham ambientes e câmeras entre os lados.

## Experimento futuro de robustez de pose

O esquema candidato é `pooled_minority_subjects`. Um experimento com pontos
de pose ausentes ou quedas curtas de pose pode ser comparado nas três
dobras, pareado por dobra, seed e receita, desde que respeite as condições
abaixo.

Bloqueios de implementação e protocolo, que exigem aprovação antes de
treinar:

- `gatefall.train.internal_folds` só implementa as dobras por ambiente do
  Le2i-CV. Um modo por sujeito no Le2i-CS é um protocolo novo e desloca a
  próxima etapa exploratória de treino. A linha de base desse modo, sem
  aumento, está na [Etapa 4B](../train/subject-folds.md).
- `PoseWindowDataset` guarda em cache a matriz de features de cada vídeo,
  montada uma vez por `build_pose_features_from_arrays`. Uma queda de pose
  simulada precisa alterar `person_found` e os keypoints em `PoseArrays`
  antes da imputação por forward-fill e do `dt` efetivo. Mascarar a janela já
  montada deixa velocidade, aceleração e orientação do tronco incoerentes.
- As features só têm `kp_conf` por keypoint e `person_found` por quadro, e a
  imputação é por pessoa. A representação de um keypoint ausente simulado
  precisa ser fixada antes de treinar.
- DINOv3 e SAM 3 não são perturbados junto com a pose. O experimento mede
  robustez à falha do detector de pose, não a oclusões.
- Pela invariante 1, o aumento precisa ser idêntico em A, B e C.

Salvaguardas obrigatórias:

- Composição das dobras fixada antes de treinar, com listas de vídeos e
  sujeitos e SHA-256, derivada só do `train` oficial.
- Estatísticas de padronização ajustadas por dobra, só nas janelas do treino
  de desenvolvimento em `TRAIN_STRIDE` e sem aumento, em caminho isolado.
  `src/gatefall/features/stats/pose_le2i_cs.json` e as estatísticas oficiais
  de DINOv3 e SAM 3 foram ajustados em todo o `train`, inclusive nos sujeitos
  retidos, e não podem ser usados nas dobras.
- Aumento aplicado só ao treino de desenvolvimento. O lado retido é avaliado
  em `EVAL_STRIDE`, sem perturbação.
- Configuração do aumento, receita, seeds e critério de decisão fixados
  antes de treinar.
- Proveniência por run (composição da dobra, hashes das estatísticas,
  configuração do aumento, commit e hashes das entradas), com runs só em
  `runs/local/`.
- Macro-F1 restrita comparada só nas dobras comparáveis. A F1 de `standing`
  na dobra agregada é relatada, mas não decide. `val` e `test` oficiais não
  entram em nenhuma seleção.

## Veredito

**`feasible_with_limitations`.** O `train` oficial do Le2i-CS sustenta três
dobras por sujeito bruto com as oito classes restritas nos dois lados e
nenhum vídeo compartilhado. Elas servem para uma comparação pareada e
controlada de um único aumento de robustez de pose, não para estimar o
desempenho no `test` oficial nem para verificar o alvo de 0,800.

As limitações que acompanham o veredito:

- São só três dobras, de tamanhos muito diferentes. O sujeito 0 tem 49% das
  janelas rotuladas, e a dobra que o retém treina com 51% delas.
- Duas das três dobras retêm um único sujeito. A variação entre sujeitos não
  pode ser estimada dentro delas.
- Na dobra agregada, `standing` vem de um único sujeito e de dois vídeos.
- As dobras são por sujeito, não por cena: ambientes e câmeras aparecem dos
  dois lados.
- A identidade global dos IDs brutos não é verificável.
- O pipeline ainda não tem ponto de aumento na pose. O modo por sujeito foi
  implementado depois, na [Etapa 4B](../train/subject-folds.md).

## Garantias de somente leitura

- O SHA-256 de `frames.parquet` e `manifest.parquet` é calculado antes da
  leitura e conferido antes da gravação.
- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, `docs/` ou nos diretórios das
  entradas. Dentro do repositório, só `runs/local/` é aceito. Sem `--force`,
  nada é sobrescrito, e os arquivos são gravados atomicamente.
- Se `frames.parquet` ou `manifest.parquet` não existir, o relatório sai com
  `status: missing_prerequisites`, a lista dos arquivos faltantes e nenhum
  número.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.subject_fold_feasibility selftest
uv run python -m gatefall.eval.analysis.subject_fold_feasibility assess \
  --output-dir runs/local/subject_fold_feasibility/le2i_cs_train \
  [--frames PATH] [--manifest PATH] [--force]
```

`assess` usa por padrão `data/processed/le2i/frames.parquet` e
`manifest.parquet`. Não usa GPU, features, pose nem checkpoints, e leva cerca
de 13 s nesta máquina. A saída tem `subject_fold_feasibility.json` (auditoria
de identidade, esquemas, dobras com isolamento, suporte por classe, riscos e
limitações, avaliação do experimento de pose, veredito e SHA-256 das
entradas) e `subject_fold_feasibility.md`.

`selftest` usa só dados sintéticos e cobre a auditoria de identidade,
identidades ambíguas (vídeo com dois sujeitos, sujeito ausente, fracionário
ou negativo, divergência entre frames e manifesto), isolamento das dobras,
construção dos dois esquemas com desempate, falhas de suporte por classe e
veredito `not_feasible`, leitura só de `train`, pré-requisitos ausentes,
determinismo, preservação das entradas e proteção da saída.

### Execução registrada (não canônica)

2026-10-15, branch `feat/subject-fold-feasibility` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados do checkout principal. Por isso, as chaves de `input_sha256` do JSON
são caminhos absolutos:

| Entrada | SHA-256 |
| --- | --- |
| `frames.parquet` | `2a563bb04d507f19732baf3bcda7aba7e51ba54b826aa3b0054f74cb67631a61` |
| `manifest.parquet` | `9a38620d745aad428918c4f4aa1569f0af1d223ac67fbc37f1c7e697aa2bf34c` |

Os SHA-256 de `runs/reference/`, `data/processed/le2i/`,
`src/gatefall/features/stats/` e das saídas das Etapas 2B e 3C foram iguais
antes e depois. Duas execuções produziram arquivos idênticos. As saídas não
são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `subject_fold_feasibility.json` | `a5b4a5bf2e26f0a14af3f8559d6eb681bfe2e16e5bf4c03917002be6534ad7ca` |
| `subject_fold_feasibility.md` | `43debd22469fb1f621c2e0e0ec79a5ace3183578601c2793bd386fd2904234ea` |

## Limitações

- Contagens por janela usam janelas sobrepostas e não são observações
  independentes. Sujeitos e vídeos são as unidades mais próximas de
  independentes, e o `train` tem só seis sujeitos brutos.
- As regras dos esquemas foram fixadas depois de inspecionar o suporte por
  classe do `train` oficial, sem olhar `val` ou `test`.
- As magnitudes das dobras não são comparáveis às do `test` oficial.
- Qualidade de pose, de features e de rótulos não é avaliada aqui.
