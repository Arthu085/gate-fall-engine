# Contrato: protocolo congelado

A fonte única é `gatefall.study.protocol.FROZEN_PROTOCOL`. O SHA-256 do JSON canônico entra na
identidade de cada run e em cada relatório:

```bash
uv run python -m gatefall.study protocol
```

| Bloco | Valor congelado |
| --- | --- |
| Entrada | RGB monocular; 10 fps; janela de 24 quadros terminando em t; passo de treino 4 e de avaliação 1; sem descritores de profundidade. Quadro de origem: o último com timestamp ≤ t nos datasets encenados (`latest_frame_at_or_before_t`); o mais próximo no Le2i histórico (`nearest_frame_historical`). |
| Rótulos | Taxonomia OmniFall na revisão `68e5cee…`; dez classes GateFall; `IGNORE = -1`; classes OmniFall 10–15 ignoradas. |
| Backbones | YOLO-Pose, DINOv3 `dinov3_vitb16` e SAM 3 congelados; features offline em HDF5 por vídeo. |
| Features | `pose138-valid-intervals-v1`, evidence8, normalização só do treino; entrada TCNTE `tcnte-adapted-v1`. |
| Treino | 30 épocas, batch 64, `best_val`, sem `augment`, `order_weight = 0`; seleção pela validação; teste não consultado. |
| Sementes | 42, 43, 44, 45, 46. |
| Alarme | `BASELINE_A_ALARM_PROTOCOL`: positivos `fall` e `fallen`, 3 positivos consecutivos, refratário de 5 s, associação até 2 s após o fim de `fallen`. |
| Partições | `upfall-subject-v1`, `gmdcsa24` só teste, Le2i-CS e Le2i-CV do OmniFall. |
| Transferência | UP-Fall → GMDCSA-24, sem adaptação, com estatísticas do treino de origem. |
| Estatística | Bootstrap por cluster, 2000 reamostras, semente 20260, IC 95%; sujeito como unidade primária e gravação como secundária; pareamento por semente. |

O schema atual é `study2026-v2`. Em relação à v1, ele acrescenta o contrato de alinhamento da
origem e separa as unidades de cluster primária e secundária. Nenhum run real existia na v1.

Qualquer alteração em um desses valores muda o hash, e o relatório recusa misturar runs de hashes
diferentes. Alterar o protocolo exige nova versão da especificação, com o trade-off declarado.
