# Treino — Investigação de determinismo de GPU

Esta página documenta a investigação por trás do item "Determinismo de GPU"
em [Treino — Braço A (TCN)](baseline-a.md).

## Fatos confirmados

- Quatro flags agora são obrigatórias em `_configure_determinism`:
  `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG`
  fixado em um valor suportado pelo cuBLAS, padrão `:4096:8`.
- `runs/reference/le2i/baseline_a` foi treinado antes desta correção e já
  foi regenerado sob o regime determinístico corrigido — ver "Migração de
  referência: determinismo de GPU" em
  [Treino — Braço A (TCN)](baseline-a.md#migracao-de-referencia-determinismo-de-gpu).

## Hipóteses descartadas

- Retreinos controlados com pose/features congeladas e a mesma seed pararam
  de divergir depois de fixar `CUBLAS_WORKSPACE_CONFIG` para um valor
  suportado — ou seja, a causa era não-determinismo de kernel CUDA/cuDNN, não
  as features upstream, e está fechada por esta correção.

## Verificação: três retreinos com hash idêntico

Três retreinos independentes com seed 42 e as mesmas entradas congeladas
produziram checkpoint sha256
`264f4997f0875881f35e20f64a370c955b3488759d5f3714816c6771f12f0ff7` e
test macro-F1 `0.6201067209256219` em todas as execuções.

## Escopo

Esta página documenta a verificação da própria correção; não substitui nem
retoma `docs/tasks/*` (instruções de tarefa efêmeras, ignoradas pelo git,
nunca uma superfície de documentação).
