# Avaliação — Braço B1 (protocolo de alarme por evento)

## Como executar

```bash
uv run python -m gatefall.eval.baseline_b1 selftest
```

Roda checagens sintéticas da inferência com gate, das guardas de protocolo e do
lifecycle dos artefatos, sem acessar o dataset real nem um checkpoint.

```bash
uv run python -m gatefall.eval.baseline_b1 evaluate --dataset le2i \
  --run-dir runs/local/le2i/b1_adaptive_gate
```

Avalia `val` e `test` de um run B1 completo. Sem `--run-dir`, usa
`runs/local/le2i/b1_adaptive_gate/`. A CLI aceita somente Le2i CS. As guardas do
`run_dir` são as mesmas do treino e do report — ancoradas em `REPOSITORY_ROOT`,
rejeitam os runs dos braços A e B0 e, para runs não canônicos desses braços, a
precheck sobre a `arm` declarada no `config.yaml` recusa o destino.

A qualidade entra no gate **crua** na inferência: pose e DINOv3 são
padronizados, `[q_pose, q_visual]` não, exatamente como no treino.

A avaliação roda `BASELINE_A_ALARM_PROTOCOL` congelado — não há retuning de
limiar para o B1 — e reutiliza o mesmo schema de `event_metrics.json`, os mesmos
hashes e o mesmo lifecycle atômico com lock, journal, staging e promoção
descrito em [Avaliação — Braço A](baseline-a-events.md). Publica
`alarm_protocol.yaml` e `event_metrics.json` no próprio run B1. Sem `--force`,
preserva um par de saídas íntegro e falha diante de artefatos parciais ou
inconsistentes; com `--force`, reconstrói e substitui o par somente após validar
os novos arquivos, com rollback em caso de falha.
