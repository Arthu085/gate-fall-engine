# UP-Fall e GMDCSA-24

Datasets do [estudo comparativo 2026](../experiments/comparative-study.md). Ambos usam rótulos
OmniFall na revisão pinada (`68e5cee…`), a mesma grade de 10 fps do Le2i e um adaptador sob o
contrato `DatasetAdapter`. O código fica em `gatefall.data.staged` (preparação) e
`gatefall.datasets.staged` (adaptadores `upfall` e `gmdcsa24`).

## Fontes oficiais

| Dataset | Fonte | Caminho local |
| --- | --- | --- |
| UP-Fall | <https://sites.google.com/up.edu.mx/har-up/> (ZIPs de câmera por sujeito/atividade/ensaio) | ZIPs em `data/scratch/upfall_archives/`; extraído em `data/raw/upfall/` |
| GMDCSA-24 | <https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos> e <https://doi.org/10.5281/zenodo.12921216> | `data/raw/gmdcsa24/GMDCSA24/Subject N/{ADL,Fall}/NN.mp4` |
| CAUCAFall v4 | <https://data.mendeley.com/datasets/7w7fccy7ky/4> | fallback documentado; não implementado |

Os downloads são manuais e nenhum arquivo bruto entra no Git.

## UP-Fall

- **Fonte:** cada vídeo de câmera é uma sequência de imagens com timestamp no nome
  (`2018-07-04T12_04_17.738369.png`). O identificador segue
  `Subject{S}/Activity{A}/Trial{T}/Subject{S}Activity{A}Trial{T}Camera{C}`.
- **Staging:** `stage-archives` extrai cada `Subject{S}Activity{A}Trial{T}Camera{C}.zip`
  usando só os nomes-base, para que nenhum caminho do ZIP escape do destino. A extração passa por
  um diretório `.partial` e termina com rename. ZIPs já preparados são mantidos, e um `.partial`
  restante interrompe a execução.
- **Gravação:** as duas câmeras de um ensaio compartilham o `recording_id`
  (`Subject…/Activity…/Trial…`) e sempre ficam na mesma partição. `audit` mede o deslocamento
  de início entre as vistas (tolerância de 1 s).
- **Grade:** cada passo de 10 fps usa o quadro mais próximo. Quando a sequência tem uma lacuna
  maior que um passo à frente de t, o passo usa o último quadro em ou antes de t, para não
  espiar o futuro. Passos com deslocamento acima de 0,1 s são marcados `source_gap` e mantêm o
  rótulo.
- **Sem rótulo:** os ensaios (8, 11, 2) e (8, 11, 3) não têm anotação OmniFall e ficam fora do
  manifesto, registrados em `unlabeled_sources`.

### Partição `upfall-subject-v1`

| Partição | Sujeitos |
| --- | --- |
| train | 1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 17 |
| val | 11 |
| test | 2, 4, 15, 16 |

O split `up_fall-cs` do OmniFall repete os sujeitos 11, 15 e 16 em `train` e em `val`/`test`, o
que vazaria informação. `audit` reporta esse vazamento em `coverage.json`. A partição do estudo
mantém os sujeitos de avaliação designados pelo OmniFall e os remove de `train`.

## GMDCSA-24

- Os 160 clipes MP4 de 4 sujeitos são decodificados como vídeo comum. O identificador segue
  `Subject_{N}/{ADL,Fall}/{NN}`.
- Todos os sujeitos estão em `test`. O pacote de features contém só essa partição e é usado
  apenas por `evaluate-external`.
- Os diretórios `data/processed/gmdcsa24_external` e `data/features/gmdcsa24_external` são
  separados do experimento de rótulos fracos, que usa `data/processed/gmdcsa24`.

## Rótulos

A tabela completa está em `specs/003-gatefall-comparative-study-2026/contracts/class-mapping.md`.

- As classes OmniFall 0–9 coincidem com as dez classes GateFall.
- As classes 10–15 viram `IGNORE` e são contadas como não suportadas, nunca fundidas em `other`.
- Lacunas de anotação permanecem `IGNORE`.
- Um segmento que termina depois do fim da fonte, além de 0,5 s, falha com erro explícito.

## Artefatos (`data/processed/<dataset>/`)

| Arquivo | Etapa |
| --- | --- |
| `manifest.parquet`, `identities.parquet`, `sequence_index.parquet`, `dataset_provenance.json` | `manifest` |
| `frames.parquet`, `alignment.json` | `timegrid` |
| `coverage.json` | `audit` |

O manifesto falha quando falta uma fonte ou quando as identidades divergem da anotação. A
proveniência registra a partição e seu SHA-256, a revisão OmniFall, o hash do conteúdo do
manifesto e o estado Git.

## Features

Os extratores existentes aceitam `--dataset upfall` e `--dataset gmdcsa24` e gravam em
`data/features/<id>/{pose,quality,dinov3,sam3}/`:

- Para o UP-Fall, os quadros vêm de `decode_image_frames` do adaptador (PIL → RGB), em vez do
  decodificador de vídeo.
- `extract-all` pula vídeos já extraídos.
- `gatefall.study features-report --dataset <id>` lista vídeos faltantes ou desalinhados sem
  carregar modelos.
