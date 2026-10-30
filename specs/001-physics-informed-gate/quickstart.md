# Quickstart de validação

1. Instalar ambiente: `uv sync --locked --dev`.
2. Rodar `uv run python -m gatefall.experiments.physics selftest`.
3. Rodar `uv run pyright` e `uv run mkdocs build --strict`.
4. Com Lean/elan instalado: `cd formal`, `lake exe cache get`, `lake build --wfail`.
5. Preparar fontes Le2i seguindo os runbooks existentes; executar prepare/train/evaluate
   conforme [contrato da CLI](contracts/cli.md). Nenhum peso novo é baixado pelo runner.

Comparação mínima: A pose; B0/C0 concat; B1/C1 learned; B1/C1 physics;
physics+augment+order; concat+augment. Manter pacote/partições/sementes iguais
em cada família visual. Rodar sementes 42–46. Acrescentar constant e quality_concat
para separar adaptação de informação auxiliar; monotonic separa sinais de resíduos.

Esperado nos testes: aceleração analítica correta, causalidade de prefixo,
gradientes da margem finitos, artefatos reproduzíveis e ausência de alterações
em runs/reference. Não interpretar teste sintético como nova métrica Le2i.
