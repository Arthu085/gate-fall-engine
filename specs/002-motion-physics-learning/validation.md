# Validação da implementação

Data: 2026-10-13. Base preservada: `301b32e42e47f29fed1daf607948d11ddf3b4e5d`.

Implementação publicada no commit `9fb8a8a`, nas branches `main` e
`codex/motion-physics-learning` de
[rsramos30/gate-fall-motion-physics](https://github.com/rsramos30/gate-fall-motion-physics).
É um repositório derivado independente, preservando o histórico; a tentativa
de segundo fork nativo retornou o fork já existente na conta. O checkout e
o repositório anteriores não receberam alterações nesta implementação.

## Resultados de engenharia

| Checagem | Resultado |
|---|---|
| `uv sync --frozen` | PASS, sem adicionar dependências |
| `uv run python -u -m gatefall.experiments.motion selftest` | PASS |
| `uv run python -u -m gatefall.experiments.physics selftest` | PASS, regressão do módulo anterior |
| CLI `motion compare`, fixture sintética C1/physics, semente 42, uma época | PASS nas três receitas, JSON e CSV gerados |
| `uv run pyright --pythonplatform Linux` | PASS, zero erros |
| `uv run pyright` no Windows | Cinco erros preexistentes de `fcntl` em `eval/shared/event_artifacts.py`, linhas 74 e 87 |
| `uv run mkdocs build --strict` | PASS |
| `git diff --check` | PASS |

O selftest verifica unidades e horizontes; máscaras de ausência, confiança e
fim de vídeo; perdas nulas sem alvos; contradição cinemática com perda positiva;
gradientes na TCN e nos dois cabeçotes; preservação da inicialização e do RNG;
inferência independente dos cabeçotes e sem arquivos de alvos futuros; todas as
receitas nos cinco braços; recusa de adulteração e de comparações incompletas,
duplicadas ou com ambientes/código de avaliação diferentes.

Na fixture C0/concat, a baseline reproduziu exatamente os parâmetros
compartilhados do treino anterior com a mesma semente e configuração.
Essa equivalência foi verificada no ambiente sintético local; não é promessa
de igualdade bit a bit entre plataformas ou versões de bibliotecas.

Os testes sintéticos foram executados no Windows em CPU. A checagem com alvo
Linux é estática; não equivale a uma execução do experimento em Linux.
O projeto completo continua recomendando Linux/WSL. Os erros Windows acima
pertencem ao módulo antigo de travamento de arquivos, que não foi alterado.

## Limites da validação

Não foram executados os experimentos reais Le2i CS/CV nem medido ganho de
F1, sensibilidade, falsos alarmes ou latência nesta versão. Os números da
fixture artificial não são resultados científicos e não devem entrar no TCC.
As provas Lean anteriores não foram ampliadas para este módulo.

O próximo passo é executar o protocolo descrito no
[guia do aluno](../../docs/experiments/motion-physics.md). A contribuição física
deve ser avaliada principalmente por `motion_physics - motion`, com todas as
sementes e métricas; a aprovação dos testes de engenharia não comprova melhoria.
