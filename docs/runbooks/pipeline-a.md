# Pipeline completo do braço A

## Rota principal

Após `uv sync`, coloque `FallDataset.zip` em `data/raw/le2i/` e execute:

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm A
```

O orquestrador executa subprocessos com o mesmo Python do ambiente. Uma falha
interrompe imediatamente a sequência, mostra nome, comando e código de saída e
confirma que etapas posteriores não rodaram. Uma nova execução retoma pelo
comportamento idempotente de cada produtor; validações rodam novamente.

Use `--dry-run` para imprimir os 26 comandos sem executá-los. Use `--force`
somente para uma reconstrução deliberada: ele é propagado aos produtores que
o suportam, nunca às validações. A extração de pose exige os pesos do
YOLO-Pose e se beneficia de GPU; treino e avaliação também se beneficiam de
GPU. Aquisições de rede falham explicitamente quando indisponíveis.

## Sequência exata

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
25. `python -m gatefall.eval.baseline_a_events selftest`
26. `python -m gatefall.eval.baseline_a_events evaluate --dataset le2i --run-dir runs/local/le2i/baseline_a`

O prefixo real é o interpretador do `uv run` (`sys.executable`), não
necessariamente a palavra literal `python`. O contador exibido é `[01/26]` a
`[26/26]`.

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
`runs/local/le2i/baseline_a/`, ignorado pelo Git.

O treino publica um run somente quando configuração, checkpoint e métricas são
válidos e coerentes. Lock, journal, diretório temporário, hashes do config e do
checkpoint e promoção atômica impedem que um run parcial pareça completo. Sem
`--force`, artefatos ausentes, inválidos ou inconsistentes produzem erro
preciso; um run completo e válido é preservado. A avaliação aplica o mesmo
princípio ao par protocolo/métricas e mantém o par anterior se a nova
publicação falhar. `--force` autoriza substituir saídas locais, nunca as
referências.

## Validação de desenvolvimento e CI

Pyright e documentação não são etapas científicas do pipeline. A workflow
**CI** roda em pull requests e pushes para `main`; o job/check **validation**
instala FFmpeg e executa todos os selftests sintéticos, `uv run pyright` e
`uv run mkdocs build --strict`, sem dataset real, pesos, GPU ou arquivos
privados. O nome completo exibido como check é **CI / validation**.
