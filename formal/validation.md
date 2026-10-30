# Validação formal executada

Data: 9 de outubro de 2026 (America/Sao_Paulo).

## Comando e resultado

Executado a partir de `formal/`, usando o binário portátil do workspace:

```powershell
& ../.tools/lean/lean-4.19.0-windows/bin/lake.exe build
```

Resultado: código de saída **0**, sem erros ou avisos. O build declarou:

```text
Built GateFall
Build completed successfully.
```

Foram verificadas **19 declarações de teorema** em `GateFall.lean`. Os 19
comandos `#print axioms` exibiram exclusivamente:

```text
[propext, Classical.choice, Quot.sound]
```

Esses são axiomas padrão do ambiente Lean/Mathlib. Não houve `sorryAx` nem
axioma definido pelo projeto. Uma busca por `sorry`, `admit`, `axiom` e
`unsafe` como palavras inteiras no arquivo Lean não encontrou ocorrências.

SHA-256 de `GateFall.lean` nesta validação:

```text
7B2E1B9911246724CC71707C5AF24E42C06C4A74809B3EED1BD875027F861FD5
```

## Teoremas verificados

- `sigmoid_pos`
- `sigmoid_lt_one`
- `sigmoid_monotone`
- `gate_pose_quality`
- `gate_visual_quality`
- `gate_pose_inconsistency`
- `gate_visual_inconsistency`
- `maskedGate_bounds`
- `maskedGate_pose_inconsistency`
- `maskedGate_visual_inconsistency`
- `invalid_pose_residual_ignored`
- `invalid_visual_residual_ignored`
- `orderHinge_nonneg`
- `constant_gate_positive_loss`
- `zero_hinge_requires_margin`
- `uniform_velocity`
- `uniform_motion_zero_acceleration`
- `midpoint_interval_positive`
- `uniform_acceleration_exact`

## Ambiente e alcance

Lean reportou `4.19.0`, plataforma `x86_64-w64-windows-gnu`, commit
`6caaee842e94`; Lake reportou `5.0.0-6caaee8`. Mathlib e as dependências
transitivas seguem o manifesto versionado.

O build final foi executado no projeto `formal/`, aproveitando os artefatos
Mathlib de um cache local através de junctions ignoradas em `.lake/packages/`.
O fonte de `GateFall.lean` foi compilado neste build. O primeiro protótipo
também foi compilado separadamente com `lake env lean` antes da promoção.

Este resultado valida as proposições reais e suas hipóteses explícitas. A
correspondência com Python, máscaras de pares, comportamento de float32,
gradientes, seleção de checkpoint e desempenho experimental requerem os
testes executáveis e os experimentos descritos na spec. As condições exatas
e o mapeamento para funções Python estão em [`README.md`](README.md).
