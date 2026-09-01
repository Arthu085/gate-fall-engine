# Organização dos dados

O pipeline separa fontes externas, artefatos processados, features derivadas e
execuções. Conteúdo local reconstruível não é versionado.

```text
data/
├── raw/le2i/FallDataset.zip e <ambientes extraídos>/
├── labels/omnifall/{train,val,test,le2i}.csv e PROVENANCE.json
├── processed/le2i/{manifest,frames}.parquet
├── features/le2i/pose/<video_id>.h5
└── scratch/

runs/
├── reference/le2i/baseline_a/   # evidência histórica versionada
└── local/le2i/baseline_a/       # reprodução local ignorada pelo Git
```

| Camada | Responsabilidade |
| --- | --- |
| `raw` | Distribuição original, sem transformação científica |
| `labels` | Anotações externas e sua proveniência |
| `processed` | Manifesto e grade temporal específicos do dataset |
| `features` | Features offline, agrupadas em um HDF5 por vídeo |
| `scratch` | Diagnósticos descartáveis |
| `runs/reference` | Configurações e métricas canônicas, somente leitura |
| `runs/local` | Checkpoints e métricas de reproduções locais |

## Migração dos caminhos legados

`data/manifest.parquet` passou a
`data/processed/le2i/manifest.parquet`, e
`data/labels/le2i/frames.parquet` passou a
`data/processed/le2i/frames.parquet`. Regenere ambos com `ingest` e
`timegrid build`; não é necessário reextrair os HDF5 já existentes em
`data/features/le2i/pose/`.

O campo `relative_path` é a identidade portátil do vídeo e é resolvido pelo
adapter contra `data/raw/le2i`. Caminhos absolutos e caminhos com `..` são
rejeitados. `absolute_path` permanece no schema por compatibilidade, mas as
camadas posteriores não dependem dele.

## Fluxo

O caminho recomendado é o [pipeline A](../runbooks/pipeline-a.md). As etapas
continuam executáveis separadamente pela [referência de comandos](../reference/commands.md):
labels → extração → manifesto → grade → pose → padronização → treino → eventos.

## Responsabilidades

`gatefall.datasets` contém o contrato mínimo e o adapter Le2i. Processamento
tabular, janelamento e leitura de vídeo permanecem genéricos; detalhes de
anotação, extração do arquivo e diagnósticos de referência ficam em
`gatefall.data.le2i`. Veja a [visão de arquitetura](../architecture/overview.md).
