# Avaliação — Braço B0 (protocolo de alarme por evento)

## Como executar

```bash
uv run python -m gatefall.eval.baseline_b0 selftest
```

Roda checagens sintéticas da inferência fundida, das guardas de protocolo e
do lifecycle dos artefatos, sem acessar o dataset real nem um checkpoint.

```bash
uv run python -m gatefall.eval.baseline_b0 evaluate --dataset le2i \
  --run-dir runs/local/le2i/baseline_b0
```

Avalia `val` e `test` de um run B0 completo usando pose e DINOv3 já
extraídos, as duas estatísticas de padronização e o checkpoint treinado. Sem
`--run-dir`, usa `runs/local/le2i/baseline_b0/`. A CLI aceita CS e CV, usa o run local do protocolo escolhido e rejeita o run do braço A.

A avaliação reutiliza `BASELINE_A_ALARM_PROTOCOL`, as mesmas métricas por
evento e por janela e o mesmo lifecycle atômico com lock, journal, staging e
hashes descrito em [Avaliação — Braço A](baseline-a-events.md). Ela
publica `alarm_protocol.yaml` e `event_metrics.json` no próprio run B0. Sem
`--force`, preserva um par de saídas íntegro e falha diante de artefatos
parciais ou inconsistentes; com `--force`, reconstrói e substitui o par
somente após validar os novos arquivos, com rollback em caso de falha. A
operação não retreina nem retuna o modelo ou o protocolo.
