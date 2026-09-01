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
| DINOv3 | **Planejado** para embeddings visuais congelados do braço B |
| SAM 3 | **Planejado** para descritores de máscara congelados do braço C |

O pipeline é monocular RGB. YOLO-Pose não é ajustado durante o treino; somente
a TCN do braço A é treinada. Os braços B e C não estão implementados nesta
etapa.

## Licenças

A licença MIT do repositório cobre o código do GateFall. Código de
dependências, datasets, anotações, modelos e pesos pré-treinados podem ter
licenças e restrições diferentes. Consulte as fontes oficiais antes de baixar,
redistribuir ou usar esses materiais; a MIT não concede automaticamente
direitos sobre eles.
