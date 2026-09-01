# Visão da arquitetura

O limite central é um adapter pequeno de dataset:

```text
adapter de dataset (Le2i hoje)
    ↓
dados genéricos e processamento temporal
    ↓
extratores de features offline
    ↓
fonte genérica de janelas
    ↓
treino
    ↓
avaliação
```

`gatefall.datasets.base.DatasetAdapter` expõe apenas identificador, caminhos
canônicos, nomes de classes, dimensão de feature, manifesto, grade, resolução
de vídeo e raízes de features/estatísticas. O primeiro adapter,
`gatefall.datasets.le2i.Le2iDatasetAdapter`, pode delegar a implementações em
`gatefall.data.le2i`; as camadas de pose, features genéricas, treino e
avaliação não importam esses módulos diretamente.

`gatefall.data.pose_dataset.PoseWindowDataset` recebe uma tabela temporal e um
loader de features. Assim, preserva janelamento, padding e identidade
`(video_id, k_end)` sem conhecer Le2i. Contagens e diagnósticos congelados do
Le2i ficam fora do núcleo reutilizável.

Para adicionar outro dataset, implemente o mesmo contrato, registre-o em
`gatefall.datasets.get_dataset` e mantenha no adapter a resolução de caminhos,
rótulos e artefatos específicos. Não copie extração de pose, janelamento,
padronização, TCN ou avaliação.

## Fronteiras de armazenamento

- Dados brutos são entradas externas imutáveis em `data/raw/<dataset>/`.
- Artefatos processados, como manifesto e grade, ficam em
  `data/processed/<dataset>/`.
- Features derivadas ficam em `data/features/<dataset>/`, agrupadas por vídeo.
- Referências científicas versionadas ficam em `runs/reference/<dataset>/`.
- Execuções reproduzidas ficam em `runs/local/<dataset>/` e são ignoradas.

O orquestrador chama as CLIs existentes como subprocessos independentes; não
reimplementa ciência. Consulte o [runbook](../runbooks/pipeline-a.md).
