# Contrato: CLIs

Todos os comandos rodam na raiz do checkout que contém `data/` e `runs/`, com caminhos relativos.

## `gatefall.data.staged`

| Subcomando | Efeito | Custo |
| --- | --- | --- |
| `stage-archives --archives DIR` | Extrai para `data/raw/upfall` os ZIPs de câmera do UP-Fall ainda não preparados (`.partial` + rename). Mantém os destinos que conferem com o ZIP e recusa tudo, sem alterar nada, se algum destino divergir ou restar um `.partial`. | disco |
| `fetch-labels --dataset D [--force]` | Baixa os rótulos OmniFall pinados e os splits de referência; grava `PROVENANCE.json`. | rede |
| `verify-labels --dataset D` | Confere hashes e o esquema dos rótulos locais. | leve |
| `manifest --dataset D [--force]` | Manifesto, identidades, índice de sequência e proveniência; falha se faltar fonte. | lê todas as fontes |
| `timegrid --dataset D [--force]` | `frames.parquet` (10 fps, último quadro ≤ t) e `alignment.json`. Recusa construir sem manifesto e anotações pinadas conferidos. | CPU |
| `audit --dataset D` | Vazamento, sincronia entre vistas, auditoria do split OmniFall, `coverage.json`. | CPU |
| `selftest` | Mundo sintético; sem dados reais. | leve |

## Extratores (sem alteração de interface)

```text
uv run python -m gatefall.pose.extract extract-all --dataset {le2i,upfall,gmdcsa24}
uv run python -m gatefall.features.quality_extract extract-all --dataset ...
uv run python -m gatefall.dinov3.extract extract-all --dataset ...
uv run python -m gatefall.sam3.extract extract-all --dataset ...
```

Sem `--force`, os vídeos já extraídos são pulados.

## `gatefall.experiments.physics`

| Subcomando | Novidade nesta spec |
| --- | --- |
| `prepare --dataset {le2i,le2i-cv,upfall,gmdcsa24}` | Datasets do estudo; o GMDCSA-24 gera um pacote só de `test`. |
| `train --arm {A,B0,B1,C0,C1,TCNTE} --fusion {...,tcnte}` | TCNTE com perda focal e Adam; `timing` em `metrics.json`. |
| `evaluate --split test` | Alarmes e `timing` no relatório; escrita exclusiva. |
| `gatefall.experiments.motion evaluate --equivalence ATESTADO` | Run histórico atestado avaliado sobre o pacote atual; o atestado só é consultado se o run não usa o pacote atual. |
| `evaluate-external --bundle SRC --run-dir RUN --target-bundle DST` | Transferência congelada, sem adaptação. |

## `gatefall.study`

| Subcomando | Efeito |
| --- | --- |
| `protocol` | Imprime o protocolo congelado e o SHA-256. |
| `plan [--matrix ...] [--json NOVO]` | Contagens por matriz e estágio e estado das saídas (`done`/`pending`/`always`). |
| `commands [--matrix ...] [--stage S]` | Comandos `uv run` exatos com recursos, dependências e regra de reuso. |
| `run [--matrix ...] [--stage ...]` | Padrão `--dry-run`: só imprime. Com `--execute`, roda os passos pendentes em ordem e reusa só saídas conferidas (integridade, proveniência, cobertura e dependências). Para na primeira falha ou saída `invalid`, sem sobrescrever, e recusa uma árvore Git suja sem `--allow-dirty`. Com `--stage`, confere antes os pré-requisitos transitivos contra o plano completo, sem executá-los, e recusa se algum estiver ausente, pendente ou não conferir. |
| `report --matrix M` | Agrega avaliações que passam no verificador de integridade, inclusive o pacote externo atual; recusa agregação parcial ou incompatível. |
| `reuse-audit --source DIR [--output NOVO]` | Audita runs de movimento externos (DIR equivale a `runs/local` da origem). |
| `attest-equivalence --dataset {le2i,le2i-cv} --visual {dinov3,sam3} --original-bundle P --original-targets P` | Relê os HDF5/NPZ originais e atuais e grava, uma vez, o atestado que liga pacote e alvos históricos aos do estudo; recusa qualquer diferença fora da allowlist. |
| `features-report --dataset D` | Cobertura e alinhamento das features por vídeo. |
| `profile-extraction --dataset D --videos N` | Sem `--execute`, só descreve. Com `--execute`, extrai N vídeos de teste num diretório temporário e mede s/quadro. |
| `selftest` | Plano, causalidade, métricas, estatística, figuras, reuso, executor e relatório. |
