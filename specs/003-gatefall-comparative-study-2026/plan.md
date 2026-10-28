# Plano de implementação

## Contexto técnico

- Python 3.12, PyTorch, h5py, pandas, `uv`. Não há dependências novas.
- Os backbones congelados são YOLO-Pose, DINOv3 ViT-B/16 e SAM 3; as features ficam em HDF5 por
  vídeo.
- O treino e a avaliação acontecem em `gatefall.experiments.physics`, ponto único para os oito
  métodos, as ablações e o TCNTE. O treino de movimento estende esse mesmo ponto.

## Componentes

| Camada | Módulo | Papel |
| --- | --- | --- |
| Datasets | `gatefall.data.staged`, `gatefall.datasets.staged` | Rótulos OmniFall, manifesto, partição por sujeito, grade causal, auditoria; adaptadores `upfall`/`gmdcsa24`. |
| Fonte de quadros | `gatefall.data.frame_source` | Despacho vídeo × sequência de imagens para os extratores. |
| Extração | `gatefall.pose/dinov3/sam3/features.quality_extract` | Aceitam `upfall` e `gmdcsa24`; cache por vídeo. |
| Pacotes | `gatefall.experiments.physics.data` | Pacote offline por vídeo; o pacote externo contém só `test`. |
| Modelos | `experiments.physics.model`, `experiments.tcnte.model`, `experiments.motion.learning` | Fusão com gates, TCNTE e cabeçotes de movimento. |
| Treino/avaliação | `experiments.physics.training` | Treino com seleção por validação, avaliação causal (predições, eventos, alarmes, tempos) e transferência congelada. |
| Estudo | `gatefall.study` | Protocolo, métodos, plano, registro/executor, métricas, estatística, relatórios, reuso, cobertura e perfil. |

## Fluxo

```text
data (staged) → extract (frozen) → bundle (prepare) → [targets] → train (best_val)
      → evaluate (test) / transfer (gmdcsa24) → report
```

O executor ordena os passos por estágio. Todos os treinos terminam antes de qualquer avaliação de
teste. A proveniência de cada run é gravada uma única vez, logo após o treino.

## Invariantes do repositório

1. Os métodos diferem só pelo vetor por passo e pela fusão. Janela, partição, sementes, épocas,
   batch e protocolo de alarme são compartilhados.
2. A partição é feita por sujeito; vídeos e gravações nunca são divididos.
3. O estudo é RGB monocular.
4. Os backbones ficam congelados; só a fusão, a TCN e as cabeças são treinadas.
5. O HDF5 é organizado por vídeo.

## Riscos

- O UP-Fall de teste tem 4 sujeitos, o que dá IC largo. O IC por gravação é publicado como
  análise secundária.
- Os runs externos de movimento podem ser incompatíveis. Nesse caso são retreinados (150).
- O custo de extração depende do hardware, que fica registrado no perfil.
