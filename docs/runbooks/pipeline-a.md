# Pipelines experimentais

## Rota principal do braço A

Após `uv sync`, coloque `FallDataset.zip` em `data/raw/le2i/` e execute:

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm A
```

O orquestrador executa subprocessos com o mesmo Python do ambiente. Uma falha
interrompe imediatamente a sequência, mostra nome, comando e código de saída e
confirma que etapas posteriores não rodaram. Uma nova execução retoma pelo
comportamento idempotente de cada produtor; validações rodam novamente.

`--dataset le2i-cv` executa o mesmo pipeline sobre o protocolo Le2i
cross-environment, em artefatos isolados (`data/labels/omnifall_cv/`,
`data/processed/le2i_cv/`, `runs/local/le2i_cv/baseline_a/`) e com uma etapa
27 adicional que gera o relatório de generalização. Veja [Generalização entre
ambientes (Le2i-CV)](../eval/le2i-cv-generalization.md) para o que esse
protocolo mede e suas ressalvas.

Use `--dry-run` para imprimir os 26 comandos de A sem executá-los. Use `--force`
somente para uma reconstrução deliberada: ele é propagado aos produtores que
o suportam, nunca às validações. A extração de pose exige os pesos do
YOLO-Pose e se beneficia de GPU; treino e avaliação também se beneficiam de
GPU. Aquisições de rede falham explicitamente quando indisponíveis.

## Sequência exata do braço A

1. `python scripts/fetch_labels.py`
2. `python scripts/fetch_labels.py --verify`
3. `python scripts/extract_le2i.py`
4. `python -m gatefall.data.ingest ingest --dataset le2i`
5. `python -m gatefall.data.ingest verify --dataset le2i`
6. `python -m gatefall.data.coverage audit --dataset le2i`
7. `python -m gatefall.data.timegrid selftest --dataset le2i`
8. `python -m gatefall.data.timegrid build --dataset le2i`
9. `python -m gatefall.data.timegrid report --dataset le2i`
10. `python -m gatefall.data.windows selftest --dataset le2i`
11. `python -m gatefall.data.windows report --dataset le2i`
12. `python -m gatefall.data.frames_io selftest --dataset le2i`
13. `python -m gatefall.data.frames_io report --dataset le2i`
14. `python -m gatefall.pose.extract extract-all --dataset le2i`
15. `python -m gatefall.pose.extract report --dataset le2i`
16. `python -m gatefall.pose.kinematics selftest --dataset le2i`
17. `python -m gatefall.pose.kinematics report --dataset le2i`
18. `python -m gatefall.data.pose_dataset selftest --dataset le2i`
19. `python -m gatefall.data.pose_dataset report --dataset le2i`
20. `python -m gatefall.features.standardize selftest`
21. `python -m gatefall.features.standardize build --dataset le2i`
22. `python -m gatefall.features.standardize report --dataset le2i`
23. `python -m gatefall.train.baseline_a selftest`
24. `python -m gatefall.train.baseline_a train --dataset le2i --run-dir runs/local/le2i/baseline_a`
25. `python -m gatefall.eval.baseline_a selftest`
26. `python -m gatefall.eval.baseline_a evaluate --dataset le2i --run-dir runs/local/le2i/baseline_a`

O prefixo real é o interpretador do `uv run` (`sys.executable`), não
necessariamente a palavra literal `python`. O contador exibido é `[01/26]` a
`[26/26]`.

## Armas B0, B1, C0 e C1

Com `--dataset le2i`, o orquestrador também aceita `--arm B0`, `B1`, `C0` ou
`C1`. Somente A aceita `--dataset le2i-cv`; uma combinação incompatível é
recusada antes de qualquer subprocesso.

Cada arma executa os 22 primeiros passos de preparação, pose e padronização
listados acima. Depois, executa o sufixo correspondente, sempre com seu
próprio diretório em `runs/local/le2i/`:

| Arma | Etapas após o passo 22 | Destino local | Total |
| --- | --- | --- | --- |
| B0 | `dinov3.extract`: selftest, extract-all, report; `features.standardize_dinov3`: selftest, build, report; `train.baseline_b0`: selftest, train, report; `eval.b0_events`: selftest, evaluate | `b0_fusion/` | 33 |
| B1 | Extração e padronização DINOv3 de B0; `features.quality_extract`: selftest, extract-all, report; `train.baseline_b1`: selftest, train, report; `eval.b1_events`: selftest, evaluate | `b1_adaptive_gate/` | 36 |
| C0 | `sam3.extract`: selftest, extract-all, report; `features.standardize_sam3`: selftest, build, report; `train.baseline_c0`: selftest, train, report | `c0_fusion/` | 31 |
| C1 | Extração e padronização SAM 3 de C0; `sam3.quality`: selftest; `train.baseline_c1`: selftest, train, report; `eval.c1_events`: selftest, evaluate | `c1_adaptive_gate/` | 34 |

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm B0
uv run python -m gatefall.pipeline run --dataset le2i --arm B1
uv run python -m gatefall.pipeline run --dataset le2i --arm C0
uv run python -m gatefall.pipeline run --dataset le2i --arm C1
```

B0 e B1 exigem os pesos e o runtime locais do DINOv3. B1 extrai também os
sidecars de `q_pose` e `q_visual`. C0 e C1 exigem o runtime isolado e o
checkpoint do SAM 3. C1 calcula `q_pose` e `q_sam3` a partir das features de
pose e dos HDF5 do SAM 3 durante treino e avaliação; o selftest de
`sam3.quality` verifica a fórmula. C0 termina no relatório de classificação,
pois não há avaliação final por eventos para essa arma.

Os comandos `report` de classificação recusam um arquivo já existente sem
`--force`. Portanto, ao repetir um pipeline B0/B1/C0/C1 que já gerou
`classification_report.json`, use `--force` se quiser substituí-lo. O mesmo
flag chega apenas aos produtores que o aceitam, inclusive treino, relatório e
avaliação; as guardas de integridade e isolamento de cada CLI continuam ativas.

## Artefatos e diagnóstico

| Fase | Artefato principal | Diagnóstico relacionado |
| --- | --- | --- |
| Preparação | `data/labels/omnifall/`, `data/raw/le2i/` | [Le2i](../data/le2i.md) |
| Processamento | `data/processed/le2i/{manifest,frames}.parquet` | [Manifesto](../data/manifest-verification.md) e [tempo](../data/temporal-contract.md) |
| Features | `data/features/le2i/pose/<video_id>.h5` | relatórios de pose e janelas |
| Padronização | `src/gatefall/features/stats/pose_le2i_cs.json` | [Padronização](../data/pose-standardization.md) |
| Treino local | `config.yaml`, `metrics.json`, `checkpoint.pt` | [Treino](../train/baseline-a.md) |
| Eventos locais | `alarm_protocol.yaml`, `event_metrics.json` | [Avaliação](../eval/baseline-a-events.md) |

Inspecione a primeira etapa que falhou e rode seu comando isoladamente. Veja
a [referência](../reference/commands.md) para pré-requisitos e efeitos.

## Referência versus reprodução

`runs/reference/le2i/baseline_a/` guarda evidência histórica versionada e não
é destino aceito por treino ou avaliação. O pipeline seleciona exclusivamente
`runs/local/le2i/baseline_a/`, ignorado pelo Git. A
[política de referências finais](../reference/le2i-runs.md) cobre A, B0, B1,
C0 e C1.

O treino publica um run somente quando configuração, checkpoint e métricas são
válidos e coerentes. Ele prepara e valida os artefatos em um diretório de
staging irmão e promove o diretório com `os.replace`; com `--force`, mantém um
backup do run anterior e o restaura se a promoção falhar. O treino não usa lock
nem journal, portanto execuções concorrentes para o mesmo destino não são um
modo suportado. Sem `--force`, artefatos ausentes, inválidos ou inconsistentes
produzem erro preciso; um run completo e válido é preservado.

A avaliação tem lifecycle próprio para publicar conjuntamente protocolo e
métricas: usa lock exclusivo, journal, arquivos de staging e backups para
recuperar uma promoção interrompida e manter o par anterior consistente.
`--force` autoriza substituir saídas locais, nunca as referências.

## Repadronizar após mudar a semântica das features de pose

`standardize build` é idempotente e, sem `--force`, preserva o JSON existente.
A única guarda de obsolescência do arquivo é o SHA-256 de
`data/processed/le2i/frames.parquet` (ver [Padronização de features de
pose](../data/pose-standardization.md)), e uma mudança de semântica das
features — como a [causalidade do prefixo de
pose](../data/temporal-contract.md#imputacao-de-pose-e-causalidade-do-prefixo)
— não toca esse parquet. Um JSON calculado sobre as features antigas passa,
portanto, por todas as checagens existentes sem reclamar.

Depois de qualquer mudança em `gatefall.pose.loading` ou
`gatefall.pose.kinematics` que altere valores de coluna, rode a etapa 21 com
`--force` e a etapa 22 em seguida, e retreine o run local antes de comparar
métricas com qualquer run anterior.

## Validação de desenvolvimento e CI

Pyright e documentação não são etapas científicas do pipeline. A workflow
**CI** roda em pull requests e pushes para `main`; o job/check **validation**
instala FFmpeg e executa todos os selftests sintéticos, `uv run pyright` e
`uv run mkdocs build --strict`, sem dataset real, pesos, GPU ou arquivos
privados. O nome completo exibido como check é **CI / validation**.
