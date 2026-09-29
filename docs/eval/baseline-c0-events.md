# Avaliação — Braço C0 (protocolo de alarme por evento)

## Como executar

```bash
uv run python -m gatefall.eval.baseline_c0 selftest
uv run python -m gatefall.eval.baseline_c0 evaluate --dataset le2i \
  --run-dir runs/local/le2i/baseline_c0
```

`evaluate` exige um run C0 já treinado e íntegro. A CLI aceita somente Le2i CS e usa `runs/local/le2i/baseline_c0/` por padrão. Ela valida `config.yaml`, `metrics.json`, `checkpoint.pt`, a proveniência dos HDF5 do SAM 3 e o frescor das estatísticas de padronização antes da inferência. Não treina o modelo nem extrai novamente features do SAM 3.

A inferência usa a pose e o descritor `V_t` de 10 canais já extraídos, cada um com sua padronização calculada no treino. Avalia todas as janelas de `val` e `test`, inclusive as de rótulo ignorado, com `EVAL_STRIDE`, e preserva `(video_id, k_end)` para o cálculo de eventos. Usa o `BASELINE_A_ALARM_PROTOCOL` congelado, sem ajuste de limiar.

A publicação segue o [contrato compartilhado de eventos](baseline-a-events.md): grava `alarm_protocol.yaml` e `event_metrics.json` juntos no próprio run C0, com lock, journal, staging e hashes. Sem `--force`, preserva um par íntegro e recusa saídas parciais ou inválidas. Com `--force`, reconstrói o par com recuperação transacional. Destinos de A, B0, B1, C1, `runs/reference/` e `le2i-cv` são recusados.

Os cinco checkpoints C0 das seeds 42–46 foram avaliados separadamente sob o protocolo congelado. A evidência textual da seed 42, referência canônica, está em `runs/reference/le2i/baseline_c0/`; o [sumário multi-seed](../analysis/multiseed-summary.md#c0) registra os agregados das cinco seeds. A escolha da referência não depende do desempenho no teste.
