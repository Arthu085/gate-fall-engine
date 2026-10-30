# Plano de implementação
## Contexto técnico
Python/PyTorch existentes, sem novas dependências. Módulo gatefall.experiments.motion reusa bundle physics, normalização, TCN e avaliação. Pontos de extensão opt-in mantêm comportamento histórico.
## Constitution Check
PASS: separação por vídeo/sujeito, futuro só em alvos, RGB, backbones congelados e teste sem seleção. Dois cabeçotes auxiliares são a extensão solicitada pelo usuário à regra de treino TCN/fusão. Substitui PINN 3D. Provas Lean anteriores não certificam o novo módulo.
## Desenho
- data.py: sidecar somente train, hashes e máscaras; horizontes 1 e 3 quadros a 10 Hz, confiança >=0,5.
- learning.py: cabeçotes independentes no estado final causal; mesma inicialização nas três receitas; RNG original preservado. Huber em unidades espaciais, escala fixa 0,1 diagonal da imagem.
- physics/training.py e model.py: extensão opt-in; perdas auxiliares só treino; seleção original por F1 de validação; inferência sem alvos.
- __main__.py e report.py: prepare/train/evaluate/compare/summarize/selftest; matriz inteira treinada antes de avaliar teste; comparação estrita e deltas por semente.
- selftest.py: analítico, máscaras, gradientes, causalidade, adulteração e integração CPU.
## Validação
Selftests motion e physics, uv run pyright nativo e alvo Linux, MkDocs strict. Registrar limitação fcntl Windows se persistir. Dados reais ausentes; benefício científico fica pendente.
