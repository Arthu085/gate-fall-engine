# Avaliação — Braço C1 (protocolo de alarme por evento)

## Como executar

```bash
uv run python -m gatefall.eval.baseline_c1 selftest
uv run python -m gatefall.eval.baseline_c1 evaluate --dataset le2i
```

`evaluate` exige um run C1 já treinado e íntegro. Antes da inferência, confere a configuração, o checkpoint, as métricas de treino, a proveniência dos arquivos SAM 3 e o frescor das fontes de pose, qualidade e padronização. Avalia todas as janelas de `val` e `test`, inclusive as de rótulo ignorado, com pose, `V_t` e `[q_pose, q_sam3]` nas mesmas posições temporais do treino. Usa o protocolo de alarme congelado do braço A, sem seleção pelo teste ou ajuste de limiar.

Grava `alarm_protocol.yaml` e `event_metrics.json` juntos no run C1. Lock e journal permitem recuperar uma publicação interrompida. Um par íntegro é preservado; `--force` reconstrói um par parcial ou inválido. A avaliação recusa destinos de A, B0, B1, C0, `runs/reference/` e do outro protocolo. A execução real exige os artefatos locais de Le2i e do run C1.
