# Contrato da CLI experimental

`uv run python -m gatefall.experiments.physics prepare --dataset le2i --visual sam3 --output data/features/physics/le2i-sam3`

Prepara pacote versionado das fontes extraídas. `--visual`: none/dinov3/sam3.
`--seed` fixa corrupção; `--max-gap-seconds` fixa reinício; `--outage-probability`
e `--outage-frames` controlam blocos. Recusa destino existente e falta de fontes.

`uv run python -m gatefall.experiments.physics train --bundle data/features/physics/le2i-sam3 --arm C1 --fusion physics --run-dir runs/local/le2i/physics-c1 --augment --order-weight 0.1 --order-margin 0.05`

`--fusion`: pose/concat/constant/quality_concat/learned/monotonic/physics.
A exige pose; B0/C0 exigem concat; B1/C1 permitem variantes. Defaults: seed42,
30 épocas, checkpoint best_val; `--checkpoint-policy last` fornece controle.
Runs não sobrescrevem diretórios existentes. Treino só avalia train/val.

`uv run python -m gatefall.experiments.physics evaluate --bundle data/features/physics/le2i-sam3 --run-dir runs/local/le2i/physics-c1 --split test`

Confere manifest/config/stats/checkpoint; produz avaliação de classificação e
eventos do split solicitado, incluindo floats de latência sem arredondamento.
`test` não é chamado automaticamente para seleção.

`uv run python -m gatefall.experiments.physics selftest`

Casos analíticos e treino sintético CPU. Erros saem com código não zero.

`cd formal; lake build`

Verifica os teoremas em Lean usando lean-toolchain e dependência Mathlib fixados.
