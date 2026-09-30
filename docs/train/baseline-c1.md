# Treino — Braço C1 (gate adaptativo com SAM 3)

C1 usa a mesma receita de treino e a mesma TCN dos braços A, B0, B1 e C0. Em cada quadro, projeta a pose de 134 para 128 canais e o descritor `V_t` do SAM 3 de 10 para 128 canais com `Linear → LayerNorm → ReLU`. O gate escalar é `g_t = sigmoid(Linear([q_pose_t, q_sam3_t]))`. A entrada da TCN é a concatenação de `g_t × pose_t` e `(1 - g_t) × SAM3_t`, com 256 canais. Os pesos de C1 são treinados do zero; YOLO-Pose e SAM 3 permanecem congelados.

O modelo, os campos da configuração, o loop de treino, a validação das métricas e as guardas genéricas de run ficam em `src/gatefall/train/shared/`. `baseline_c1/` mantém a identidade, as fontes e a validação específicas de C1.

`q_pose` vem de `gatefall.pose.quality.compute_pose_quality`. `q_sam3` vem de `gatefall.sam3.quality.compute_sam3_quality`: score da instância selecionada multiplicado pelo canal `present` de `V_t`. Ambos entram crus no gate. As janelas aplicam os mesmos índices causais e a mesma replicação de borda às três fontes. As estatísticas da pose e do SAM 3 são as mesmas usadas nos outros braços; `train/shared/sam3_inputs.py` valida proveniência e frescor para C0 e C1.

## Execução

```bash
uv run python -m gatefall.train.baseline_c1 selftest
uv run python -m gatefall.train.baseline_c1 train --dataset le2i --seed 42
uv run python -m gatefall.train.baseline_c1 report --dataset le2i
```

O diretório padrão é `runs/local/le2i/baseline_c1` para CS ou `runs/local/le2i_cv/baseline_c1` para CV. `train` grava `config.yaml`, `checkpoint.pt` e `metrics.json`; `report` grava `classification_report.json` após verificar a configuração, os hashes das fontes, a integridade dos artefatos e a compatibilidade do checkpoint. O config registra os hashes das estatísticas, dos arquivos de pose e SAM 3, a proveniência SAM 3 e a identidade das duas fórmulas de qualidade. O destino de C1 não pode coincidir com, conter ou ficar dentro dos runs de A, B0, B1, C0, `runs/reference/` ou do outro protocolo. CV reutiliza os HDF5 do SAM 3 e ajusta a estatística apenas no treino CV. `--force` substitui apenas um run C1 no destino escolhido.

A avaliação por eventos está em [Avaliação — Braço C1](../eval/baseline-c1-events.md).
