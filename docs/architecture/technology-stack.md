# Tecnologias

| Tecnologia | Estado e papel no GateFall |
| --- | --- |
| Python 3.12 | Linguagem e versão fixada do projeto |
| uv | Ambiente, lockfile e execução reproduzível de comandos |
| FFmpeg / ffprobe | Decodificação sequencial RGB e metadados/contagem de quadros |
| NumPy | Features, métricas e computação numérica |
| Pandas | Tabelas de manifesto, grade temporal e índices |
| PyArrow / Parquet | Persistência tipada do manifesto e da grade |
| HDF5 / h5py | Features de pose agrupadas em um arquivo por vídeo |
| PyTorch | TCN, treino, inferência e checkpoints |
| Ultralytics YOLO-Pose | Backbone congelado de pose executado offline |
| ByteTrack | Associação temporal da pessoa-alvo nas detecções de pose |
| TCN | Codificador temporal causal do braço A |
| MkDocs Material | Site desta documentação |
| Pyright | Verificação estática obrigatória do Python |
| DINOv3 | Backbone congelado ViT-B/16 (LVD-1689M); extração offline de embeddings visuais do braço B, fusão (B0 e B1) e avaliação por eventos implementadas |
| SAM 3 | Backbone congelado; extração offline do descritor `V_t` do braço C implementada, fusão e treino ainda pendentes |

O pipeline é monocular RGB. YOLO-Pose, DINOv3 e SAM 3 são backbones
congelados: nenhum deles é ajustado durante o treino, que alcança apenas as
cabeças de fusão e as TCNs. Do braço A é treinada a TCN sobre features de
pose. Do braço B, além da extração offline de features DINOv3, estão
implementadas e rodando em CI a fusão B0 (concatenação pose+DINOv3 seguida de
TCN dilatada rasa, `gatefall.train.b0_fusion`), a fusão adaptativa B1 (gate
escalar por timestep sobre a mesma TCN, `gatefall.train.b1_gate`) e a
avaliação por eventos de ambas (`gatefall.eval.b0_events` e
`gatefall.eval.b1_events`). Do braço C está implementada apenas a fundação de
extração offline do descritor `V_t` do SAM 3 (ver [Fundação SAM
3](../data/sam3-foundation.md)): C0, C1, fusão, gate, treino e avaliação por
eventos do braço C permanecem pendentes.

## Licenças

A licença MIT do repositório cobre o código do GateFall. Código de
dependências, datasets, anotações, modelos e pesos pré-treinados podem ter
licenças e restrições diferentes. Consulte as fontes oficiais antes de baixar,
redistribuir ou usar esses materiais; a MIT não concede automaticamente
direitos sobre eles.

O código e os pesos do DINOv3 são regidos pela licença própria do DINOv3, não
pela MIT deste repositório, e não são redistribuídos aqui — o repositório de
referência é clonado localmente em `data/scratch/` (git-ignored), nunca
vendorizado. Publicações que usem este trabalho devem reconhecer o uso de
"DINO Materials", conforme exigido pela licença do DINOv3. Veja
[Features DINOv3](../data/dinov3-features.md#licenca-do-dinov3) para os
termos completos.
