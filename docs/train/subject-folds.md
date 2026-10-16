# Treino — Dobras por sujeito do Le2i-CS (Etapa 4B)

`src/gatefall/train/subject_folds/` implementa a linha de base das dobras
por sujeito propostas na [Etapa 4A](../analysis/subject-fold-feasibility.md):
o braço A treinado com a receita congelada em três dobras do `train` oficial
do Le2i-CS. Esta etapa **não** implementa aumento de pose: só fixa a linha
de base limpa contra a qual o aumento é comparado, pareado por dobra, seed e
receita, na [Etapa 4C](subject-fold-detection-dropout.md).

O protocolo é **exploratório e não canônico**. Ele não muda as
configurações A/B/C, os modelos, os splits oficiais, as features, as
estatísticas oficiais, os runs existentes nem o comportamento padrão das
CLIs de treino. Os resultados ficam em `runs/local/le2i/subject_folds/`, não
versionado.

## Dobras congeladas

Os grupos retidos são os do esquema `pooled_minority_subjects` da Etapa 4A,
congelados em `RETAINED_SUBJECT_GROUPS` e não recalculados a partir dos
dados:

| Dobra | Sujeitos retidos | Sujeitos de desenvolvimento |
| --- | --- | --- |
| `retain_s0` | 0 | 1, 3, 4, 6, 8 |
| `retain_s1` | 1 | 0, 3, 4, 6, 8 |
| `retain_s3-4-6-8` | 3, 4, 6, 8 | 0, 1 |

- Só as linhas `split == train` de `frames.parquet` e `manifest.parquet` do
  Le2i-CS são carregadas, com filtro na leitura, e a partição precisa bater
  com o Le2i-CS pinado (133 vídeos, seis ambientes e as contagens de janelas
  em stride 1). `val` e `test` oficiais nunca entram em memória.
- Cada vídeo precisa ter um único sujeito e um único ambiente, iguais no
  manifesto, com as mesmas guardas da Etapa 4A. Os grupos precisam cobrir
  exatamente os sujeitos do `train`, sem repetição.
- Na tabela de quadros de cada dobra, a coluna `split` vira
  `development_train` ou `retained`. Vídeos e sujeitos ficam em um único
  lado, e cada vídeo é retido em exatamente uma dobra.
- Uma dobra sem alguma das oito classes de `RESTRICTED_CLASSES` em qualquer
  lado (em stride 4 no desenvolvimento, em stride 1 no retido) é recusada,
  porque a macro-F1 restrita de oito classes deixaria de ser comparável.
- Cada lado recebe um loader de pose restrito aos seus vídeos. Uma leitura
  de outro vídeo interrompe a execução.

## O que é reaproveitado sem mudança

- **Receita congelada do braço A.** A configuração sai de
  `BASELINE_A_CONFIG`: arquitetura, AdamW, pesos de classe, 30 épocas e
  checkpoint da última época. Só mudam o caminho e o hash da estatística de
  padronização. A seed é 42, e o contrato desta etapa só aceita o braço A e
  a seed 42.
- **Engine e artefatos.** O treino usa `run_training` do braço A e os mesmos
  validadores de `config.yaml`, `metrics.json` e `checkpoint.pt`, com
  staging, promoção atômica e recusa de run parcial sem `--force`. As
  guardas de `fold_run.json` são as das
  [dobras internas do Le2i-CV](internal-folds.md#diretorio-e-artefatos).
- **Janelamento.** `WINDOW_FRAMES`, `TRAIN_STRIDE` (4) no desenvolvimento,
  `EVAL_STRIDE` (1) no retido e `IGNORE_LABEL` pelo último quadro.
- **Features de pose já extraídas.** Os HDF5 de pose são lidos sem nova
  extração. DINOv3, SAM 3 e qualidade não são lidos.
- **Posições `val` e `test`.** O engine exige as duas. Nas dobras, ambas
  avaliam o mesmo grupo retido, e nenhuma é o split oficial de mesmo nome. A
  curva por época no retido é só registrada; nada é selecionado por ela.

## Padronização por dobra

`prepare` ajusta, para cada dobra, a estatística de pose só nas janelas do
treino de desenvolvimento em stride 4, com a mesma função da estatística
oficial. Cada dobra grava `stats/pose.json` e `stats/provenance.json`, com a
dobra, os sujeitos e vídeos de desenvolvimento, o número de janelas, o
SHA-256 de frames, manifesto e HDF5 de pose e o SHA-256 de `pose.json`.

`src/gatefall/features/stats/pose_le2i_cs.json` foi ajustado em todo o
`train`, inclusive nos sujeitos retidos, e nunca é usado: a configuração
recusa esse caminho, e um `pose.json` idêntico a ele é recusado mesmo com a
proveniência regravada. Estatística de outra dobra, adulterada ou ajustada
sobre entradas ou pose antigas também é recusada.

## Diretório e artefatos

```text
runs/local/le2i/subject_folds/
├── folds.json
├── retain_<sujeitos>/
│   ├── stats/{pose,provenance}.json
│   └── runs/baseline_a/seed_42/
│       ├── config.yaml, metrics.json, checkpoint.pt
│       └── fold_run.json
└── summary/baseline/subject_fold_summary.{json,md}
```

`folds.json` é determinístico e registra o contrato congelado (grupos,
janelamento, classes restritas, braço, seed, mapeamento de avaliação,
política de checkpoint e de seleção), `input_sha256` de frames e manifesto,
`pose_features_sha256` dos HDF5 de pose dos 133 vídeos, `membership_sha256`
e, por dobra, sujeitos, vídeos, janelas rotuladas e suporte por classe de
cada lado, além do número de vídeos e de sujeitos de cada classe restrita no
retido. `train` recusa a execução se o contrato divergir ou se frames,
manifesto, pose ou composição tiverem mudado desde o `prepare`.

`fold_run.json` liga cada run à dobra, ao braço, à seed, aos sujeitos, ao
mapeamento de avaliação, a `input_sha256`, `pose_features_sha256`,
`membership_sha256`, ao hash da estatística e ao SHA-256 de `config.yaml`,
`metrics.json` e `checkpoint.pt`.

`--output-root` não pode coincidir com, conter ou ficar dentro de
`runs/reference/`, `runs/local/le2i_cv/`, `runs/local/le2i_cv_internal_folds/`,
`src/`, `data/`, `docs/`, dos diretórios das entradas ou da raiz de pose.
Dentro do repositório, só `runs/local/le2i/subject_folds/` e seus
subdiretórios são aceitos, o que impede tocar nos runs canônicos de
`runs/local/le2i/`.

## Validação do resumo

`summarize` é somente leitura e não abre frames, manifesto nem features.
Antes de relatar qualquer número, ele confere:

- `folds.json` contra o contrato congelado, com o conjunto exato de campos,
  os papéis de `input_sha256`, a composição gravada, as somas do suporte por
  classe, a presença das oito classes restritas nos dois lados e
  `membership_sha256` recalculado;
- a proveniência de cada estatística contra `folds.json` e os hashes de
  entrada;
- cada run contra a configuração determinística da dobra (`config.yaml`), a
  integridade de `metrics.json` e de `checkpoint.pt` e o `fold_run.json`;
- a igualdade entre `final.val` e `final.test`, que avaliam o mesmo grupo
  retido;
- o suporte por classe de `final.train` e `final.val` contra o registrado em
  `folds.json`.

O resumo relata, por dobra, a macro-F1 restrita no retido, a F1 e o suporte
de cada classe restrita e as limitações da dobra. Não existe escore
agregado entre dobras (`pooled_score` é sempre nulo), e nem o `test`
oficial nem o alvo de 0,800 entram em nenhuma decisão.

## Como executar

Checagens sintéticas, sem dataset real:

```bash
uv run python -m gatefall.train.subject_folds selftest
```

O `selftest` cobre a composição e o isolamento das dobras, vazamentos
(vídeo ou sujeito nos dois lados, linhas de `val`/`test`, grupos
sobrepostos ou incompletos, vídeo com dois sujeitos, leitura fora do lado),
classes restritas ausentes, leitura restrita a `train`, raízes protegidas,
determinação do plano, da receita e de `folds.json`, a padronização por
dobra e sua proveniência, o treino sintético das três dobras, o ciclo de
vida de `fold_run.json` e a recusa do resumo para metadados adulterados,
`val` diferente de `test`, suporte divergente e checkpoint corrompido.

Plano de runs, sem ler dados:

```bash
uv run python -m gatefall.train.subject_folds plan
```

Preparar as dobras e as estatísticas, treinar e resumir:

```bash
uv run python -m gatefall.train.subject_folds prepare
uv run python -m gatefall.train.subject_folds train
uv run python -m gatefall.train.subject_folds summarize
```

`prepare` e `train` aceitam `--frames`, `--manifest` e `--pose-root`; o
padrão são os arquivos do Le2i-CS. `train` aceita `--fold` para dividir o
trabalho e exige GPU CUDA. Sem as entradas ou sem GPU, a CLI lista os
pré-requisitos ausentes e sai sem gravar nada. Runs completos e íntegros são
pulados, então uma execução interrompida pode ser retomada com o mesmo
comando.

## Execução registrada (não canônica)

2026-10-15, branch `feat/le2i-cs-subject-folds` (base
`analysis/gatefall-improvements`), com o código do worktree do branch e os
dados do checkout principal, em GPU CUDA (GTX 1650). `prepare`, `train` e
`summarize` rodaram sem `--force`, e uma segunda chamada de `train` pulou os
três runs íntegros. As composições e contagens de janelas coincidem com as
da Etapa 4A.

| Entrada | SHA-256 |
| --- | --- |
| `frames.parquet` | `2a563bb04d507f19732baf3bcda7aba7e51ba54b826aa3b0054f74cb67631a61` |
| `manifest.parquet` | `9a38620d745aad428918c4f4aa1569f0af1d223ac67fbc37f1c7e697aa2bf34c` |
| HDF5 de pose (`pose_features_sha256`) | `eccffa5789e0f747001362296bdf846c1093660e54b55163e2b9fe596233a618` |
| Composição (`membership_sha256`) | `2aad848422fd926fc7c15e35803a188d4219137eb5a771012d99ebbaf9e9c6e2` |

Macro-F1 restrita no grupo retido, braço A, seed 42, checkpoint da última
época. Cada linha é uma dobra; as linhas não devem ser agregadas nem
comparadas entre si.

| Dobra | Treino dev. (stride 4) | Retido: vídeos / janelas (stride 1) | Macro-F1 restrita | Menor F1 (classe, suporte) |
| --- | ---: | ---: | ---: | --- |
| `retain_s0` | 2638 | 67 / 10243 | 0,558 | `standing` 0,296 (496) |
| `retain_s1` | 4219 | 31 / 3961 | 0,658 | `standing` 0,359 (70) |
| `retain_s3-4-6-8` | 3581 | 35 / 6536 | 0,616 | `standing` 0,146 (107) |

A F1 e o suporte das oito classes restritas de cada dobra estão em
`summary/baseline/subject_fold_summary.md`. `standing` tem a menor F1 nas
três dobras. Na dobra agregada, ela vem de dois vídeos de um único sujeito,
e `sit_down` (F1 0,320, 59 janelas) vem de três vídeos.

Os SHA-256 de `runs/reference/`, `data/processed/le2i/`,
`src/gatefall/features/stats/`, `runs/local/le2i/` fora de `subject_folds/`,
`runs/local/le2i_cv/`, `runs/local/le2i_cv_internal_folds/` e
`runs/local/subject_fold_feasibility/` foram iguais antes e depois. As
saídas não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `folds.json` | `7db0ecc04a9d99f59e5d4275703f1ce61e55749ed4c164a3cd3dbecc3b56f62a` |
| `retain_s0/stats/pose.json` | `bc924795ecf456c7659e25ae364f53365e06567ac8ffa564cee3737398cf1a11` |
| `retain_s1/stats/pose.json` | `5f8916d6907ddb22b300878065b4eb321b65b54df38da18b19e09f85c9ec82b8` |
| `retain_s3-4-6-8/stats/pose.json` | `9d620d466d758076fb792d98a7cef98d9f9d8b61048b7582e2b522e11ff96537` |
| `retain_s0/.../checkpoint.pt` | `c994f116f270f73f18bb7dfa9e1a1129ae0e7427d700488e33831556aacdfe44` |
| `retain_s1/.../checkpoint.pt` | `e728df12d91981ea9b0d345e5ef91b6a3a8ab484f1e6eeecf855642ef7a59374` |
| `retain_s3-4-6-8/.../checkpoint.pt` | `55d2964e8454d06ce6ae276962188db0751707f855ec9eabe75dbbc697081c11` |
| `summary/baseline/subject_fold_summary.json` | `ad26bc6e101c55b4d4f0ea0e48e072184b937e8e8879bf73c1573392f08c83f4` |

## Limitações

- São só três dobras, de tamanhos muito diferentes. O sujeito 0 tem cerca de
  metade das janelas rotuladas do `train`, e a dobra que o retém treina com a
  outra metade.
- Duas dobras retêm um único sujeito, e a variação entre sujeitos não é
  estimável dentro delas. Com uma seed, a variação entre seeds também não.
- No retido da dobra `retain_s3-4-6-8`, `standing` vem de um único sujeito e
  de dois vídeos.
- As dobras são por sujeito bruto, não por cena: ambientes e câmeras
  aparecem nos dois lados, e a identidade global dos IDs brutos não é
  verificável.
- As magnitudes não são comparáveis entre dobras nem com o `test` oficial do
  Le2i-CS.
