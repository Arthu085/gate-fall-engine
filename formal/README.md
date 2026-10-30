# Contratos matemáticos do GateFall

Este projeto Lean 4 verifica propriedades das fórmulas usadas pelo experimento
de fusão orientada por consistência. As provas são verificadas pelo kernel de
Lean e não usam `sorry` ou `admit`.

## Ambiente reproduzível

- Lean: `leanprover/lean4:v4.19.0`, em `lean-toolchain`.
- Mathlib: commit `c44e0c8ee63ca166450922a373c7409c5d26b00b` (`v4.19.0`).
- Dependências transitivas: revisões fixadas em `lake-manifest.json`.

Com Lean/Lake disponíveis, executar neste diretório:

```sh
lake exe cache get Mathlib.Data.Complex.Exponential Mathlib.Tactic.Linarith Mathlib.Tactic.FieldSimp Mathlib.Tactic.Ring
lake build
```

O primeiro comando obtém os artefatos compilados dos imports e de suas
dependências. O segundo verifica `GateFall.lean`. Uma máquina nova precisa de
rede para obter as dependências e o cache; as provas não dependem dos dados de
treinamento. O diretório `.lake/` é ignorado pelo Git.

Na instalação portátil preparada neste workspace, o comando equivalente é:

```powershell
& ../.tools/lean/lean-4.19.0-windows/bin/lake.exe build
```

O cache local foi reutilizado por junctions em `.lake/packages/`; isso é apenas
uma otimização local, não faz parte do projeto versionado. Não é preciso
instalar Lean globalmente para usar a distribuição portátil.

## Correspondência com a implementação

| Declarações em `GateFall.lean` | Implementação Python | Contrato |
| --- | --- | --- |
| `sigmoid_pos`, `sigmoid_lt_one`, `sigmoid_monotone` | `EvidenceGate.forward` em `src/gatefall/experiments/physics/model.py` | A sigmoid real tem saída em `(0,1)` e preserva a ordem. |
| `gate_pose_quality`, `gate_visual_quality` | `EvidenceGate.forward`, modos `monotonic` e `physics` | A qualidade da pose aumenta seu peso; a qualidade visual o reduz, mantendo os demais argumentos fixos. |
| `gate_pose_inconsistency`, `gate_visual_inconsistency` | `EvidenceGate.forward`, modo `physics`, resíduos válidos | A inconsistência da pose reduz seu peso; a inconsistência visual o aumenta. |
| `maskedGate_bounds`, `maskedGate_pose_inconsistency`, `maskedGate_visual_inconsistency` | Multiplicações pelos campos `evidence[..., 4]` e `evidence[..., 5]` | As propriedades continuam válidas com flags `rp_valid` e `rv_valid` não negativas e fixas. |
| `invalid_pose_residual_ignored`, `invalid_visual_residual_ignored` | Mesmo trecho do gate | Um resíduo com flag zero não influencia o gate. |
| `orderHinge_nonneg`, `constant_gate_positive_loss`, `zero_hinge_requires_margin` | `ordering_loss` no mesmo módulo | Cada termo válido de hinge é não negativo; margem positiva penaliza saída constante; termo zero exige a separação indicada. |
| `uniform_velocity`, `uniform_motion_zero_acceleration`, `midpoint_interval_positive`, `uniform_acceleration_exact` | `observed_derivatives` em `src/gatefall/experiments/physics/kinematics.py` | Derivadas entre três observações em tempos crescentes recuperam movimento uniforme e uniformemente acelerado em amostragem irregular. |

Os caminhos Python da tabela são relativos à raiz do repositório.

`maskedGate` usa `γ * rp_valid` e `δ * rv_valid` como coeficientes efetivos;
sobre os reais isso é equivalente a multiplicar o resíduo e seu flag como em
Python. Para as propriedades de ordem, mudar um resíduo significa manter os
flags, os demais sinais e os parâmetros constantes. As provas não afirmam
monotonicidade quando a validade também muda. Flags booleanas codificadas como
0 e 1 satisfazem as hipóteses de não negatividade.

`orderHinge(m, clean, corrupted)` representa o termo da corrupção da pose.
O termo visual troca os dois últimos argumentos. A prova de perda positiva
considera um par efetivamente válido e margem positiva: ela não exige uma
penalidade quando todas as máscaras são zero. A redução mascarada do lote e
a presença de gradientes são verificadas pelos testes executáveis Python.

As provas cinemáticas aplicam-se a cada coordenada e a três observações que
pertençam ao mesmo histórico válido. A lógica de ausência, máscara, reinício
por intervalo longo e causalidade do laço Python é coberta pelos testes
executáveis; ela não é extraída automaticamente para Lean.

## Limites da verificação

As provas usam números reais, não IEEE-754. `torch.sigmoid` em float32 pode
saturar em 0 ou 1, e produtos/somas em ponto flutuante podem introduzir
arredondamento. Os testes numéricos devem exigir valores finitos, limites
fechados `[0,1]` e tolerâncias explícitas, sem transformar o limite real
estrito em uma afirmação falsa sobre float32.

Os teoremas de sinais assumem coeficientes não negativos. A implementação os
parametriza com `softplus`, e os testes Python verificam o comportamento
efetivo. A implementação de `softplus`, a correspondência entre os dois
programas e o treinamento inteiro não são certificados por este projeto.

Nenhum teorema afirma redução de falsos alarmes, melhoria de macro-F1 ou
generalização: isso exige experimentos com as partições e controles definidos
na spec. A inspeção de axiomas e os comandos executados estão em
[`validation.md`](validation.md).
