# CLI
uv run python -m gatefall.experiments.motion <comando>
- prepare --bundle PATH --pose-root PATH --output PATH [--horizons 1 3 --confidence .5]
- train --bundle PATH --targets PATH --run-dir PATH --arm C0 --fusion concat --recipe baseline|motion|motion_physics [--seed 42 --epochs 30 --batch-size 64 --motion-weight .1 --physics-weight .1]
- evaluate --bundle PATH --run-dir PATH [--split test]
- compare --bundle PATH --targets PATH --output PATH --arm C0 --fusion concat [--seeds 42 43 44 45 46]
- summarize --runs PATH [PATH ...] --output PATH
- selftest
Pesos finitos positivos, escala fixa 0,1 diagonal. Baseline desliga perdas auxiliares; motion desliga coerência. Pesos nominais iguais. Saídas existentes recusadas. compare treina matriz inteira antes do teste.
