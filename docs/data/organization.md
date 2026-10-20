# Organização dos dados

O pipeline separa fontes externas, artefatos processados, features derivadas e
execuções. Conteúdo local reconstruível não é versionado.

```text
data/
├── raw/le2i/FallDataset.zip e <ambientes extraídos>/
├── raw/figshare_fall/VideoDataset.zip  # opcional; só auditoria de viabilidade
├── raw/gmdcsa24/GMDCSA24/             # opcional; clone v2.1, auditoria e piloto com rótulos fracos
├── labels/omnifall/{train,val,test,le2i}.csv e PROVENANCE.json
├── labels/omnifall_cv/{train,val,test,le2i}.csv e PROVENANCE.json
├── processed/le2i/{manifest,frames}.parquet
├── processed/le2i_cv/{manifest,frames}.parquet
├── processed/gmdcsa24/manifest.json e {videos,frames,window_candidates,windows_t,windows_s}.parquet
├── processed/gmdcsa24_training/{t,s}/{manifest,pose_stats}.json
├── features/le2i/pose/<video_id>.h5
├── features/le2i/dinov3/<env>/<video_name>.h5
├── features/gmdcsa24/pose/manifest.json e Subject_<n>/<ADL|Fall>/<nn>.h5
└── scratch/                         # inclui gmdcsa24/provenance/ (planilha e contact sheets locais)

runs/
├── reference/le2i/{baseline_a,baseline_b0,baseline_b1,baseline_c0,baseline_c1}/
├── local/le2i/<arma>/           # reprodução local ignorada pelo Git
├── local/le2i_cv/<arma>/        # reprodução local do protocolo cv, ignorada pelo Git
└── local/gmdcsa24_weak_label/{t,s}/seed_<n>/ e decision.json  # piloto T/S, ignorado pelo Git
```

`data/labels/omnifall_cv/`, `data/processed/le2i_cv/` e
`runs/local/le2i_cv/<arma>/` são os artefatos isolados do protocolo Le2i
`cv` (cross-environment) e nunca compartilham arquivo com seus equivalentes
`cs` acima. As features de pose, DINOv3, SAM 3 e qualidade em
`data/features/le2i/` são compartilhadas entre os protocolos: somente os
comandos de extração `le2i` podem gravá-las; os consumidores e relatórios
`le2i-cv` podem lê-las. Não existe
`runs/reference/le2i_cv/`: a promoção de um run a
referência nunca é automática. Veja o [relatório de
generalização](../analysis/le2i-cv-generalization.md) para o que o protocolo `cv`
mede.

`data/processed/gmdcsa24/` é a saída isolada do adapter do [piloto com
rótulos fracos](../analysis/gmdcsa24-weak-label-protocol.md#adapter-e-validacao).
Ela só contém janelas de treino do GMDCSA24 e nunca grava em
`data/processed/le2i/`, nas features, nos rótulos ou nos runs.
`data/features/gmdcsa24/pose/` é a saída isolada da [pose do
GMDCSA24](gmdcsa24-pose.md), só dos 145 vídeos elegíveis; ela nunca grava em
`data/features/le2i/` nem em `data/processed/`.
`data/processed/gmdcsa24_training/` e `runs/local/gmdcsa24_weak_label/` são as
saídas isoladas do [treino de T e S](../train/gmdcsa24-weak-label.md#saidas):
padronização e manifesto de cada tratamento e os runs dele, sem nunca gravar
nos artefatos do Le2i nem nos runs do controle. Os runs recebem também os
[eventos só de `val`](../eval/gmdcsa24-weak-label-val.md#saidas), e
`decision.json` registra a decisão GO/NO-GO.

| Camada | Responsabilidade |
| --- | --- |
| `raw` | Distribuição original, sem transformação científica |
| `labels` | Anotações externas e sua proveniência |
| `processed` | Manifesto e grade temporal específicos do dataset |
| `features` | Features offline, agrupadas em um HDF5 por vídeo |
| `scratch` | Diagnósticos descartáveis |
| `runs/reference` | Evidência textual canônica de classificação e, quando disponível, eventos; somente leitura |
| `runs/local` | Checkpoints e métricas de reproduções locais |

A [política de referências finais](../reference/le2i-runs.md) define os arquivos
de cada braço, incluindo a evidência de eventos C0.

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

O caminho recomendado é o [pipeline A](../runbooks/pipelines.md). As etapas
continuam executáveis separadamente pela [referência de comandos](../reference/commands.md):
labels → extração → manifesto → grade → pose → padronização → treino → eventos.

## Responsabilidades

`gatefall.datasets` contém o contrato mínimo e o adapter Le2i. Processamento
tabular, janelamento e leitura de vídeo permanecem genéricos; detalhes de
anotação, extração do arquivo e diagnósticos de referência ficam em
`gatefall.data.le2i`. Veja a [visão de arquitetura](../architecture/overview.md).
