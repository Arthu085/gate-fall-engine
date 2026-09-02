# Referência de comandos

Execute os comandos na raiz do repositório, após `uv sync`. `--dataset le2i`
é opcional nas CLIs genéricas porque Le2i é o padrão. “Dados” indica acesso ao
dataset real; “GPU/pesos” indica necessidade ou benefício de aceleração e
pesos. Links apontam para o contrato detalhado.

## Orquestração e preparação

| Sintaxe | Propósito e pré-requisitos | Entrada → saída; mutação e idempotência | Dados | GPU/pesos | Detalhes |
| --- | --- | --- | --- | --- | --- |
| `uv run python -m gatefall.pipeline run [--dataset le2i] [--arm A] [--dry-run] [--force]` | Reproduz todo o braço A; ambiente sincronizado e ZIP preparado | Executa os 26 estágios → todos os artefatos locais; muta; rerun preserva saídas válidas; `--force` chega só a produtores compatíveis | Sim, exceto `--dry-run` | Sim nas fases de ML | [Runbook](../runbooks/pipeline-a.md) |
| `uv run python -m gatefall.pipeline selftest` | Valida ordem, falha, dry-run, force e destino local | Entradas sintéticas → stdout; não muta; repetível | Não | Não | [Runbook](../runbooks/pipeline-a.md) |
| `uv run python scripts/fetch_labels.py [--force]` | Baixa o snapshot OmniFall; requer rede | Fonte fixada → `data/labels/omnifall/`; muta; preserva existentes; `--force` baixa/sobrescreve | Não antes do comando | Não | [OmniFall](../data/omnifall.md) |
| `uv run python scripts/fetch_labels.py --verify` | Verifica hashes/proveniência; requer labels baixadas | Labels + `PROVENANCE.json` → stdout; não muta; repetível; sem `--force` | Sim, labels | Não | [OmniFall](../data/omnifall.md) |
| `uv run python scripts/extract_le2i.py [--zip PATH] [--force]` | Extrai o ZIP obtido manualmente | ZIP → `data/raw/le2i/<ambientes>`; muta; preserva extraídos; `--force` reextrai somente destinos conhecidos | Sim | Não | [Le2i](../data/le2i.md) |
| `uv run python scripts/exploratory/explore_le2i.py` | Exploração histórica opcional; requer distribuição local | Le2i → stdout; não é etapa produtiva nem muta artefatos canônicos | Sim | Não | [Le2i](../data/le2i.md) |

## Manifesto, tempo e quadros

| Sintaxe | Propósito e pré-requisitos | Entrada → saída; mutação e idempotência | Dados | GPU/pesos | Detalhes |
| --- | --- | --- | --- | --- | --- |
| `uv run python -m gatefall.data.ingest ingest [--dataset le2i] [--force]` | Casa labels/vídeos e mede metadados; requer FFmpeg | raw + labels → `data/processed/le2i/manifest.parquet`; muta atomicamente; preserva existente; `--force` reconstrói | Sim | Não | [Manifesto](../data/manifest-verification.md) |
| `uv run python -m gatefall.data.ingest verify [--dataset le2i]` | Valida bijeção, splits, arquivos e hashes | manifesto + raw + labels → stdout; não muta; repetível | Sim | Não | [Manifesto](../data/manifest-verification.md) |
| `uv run python -m gatefall.data.coverage audit [--dataset le2i]` | Audita cobertura das anotações | manifesto + labels → stdout; não muta; repetível | Sim | Não | [Cobertura](../data/manifest-verification.md#auditoria-de-cobertura) |
| `uv run python -m gatefall.data.timegrid selftest [--dataset le2i]` | Testa reamostragem e rótulos | Casos sintéticos → stdout; não muta; repetível | Não | Não | [Contrato temporal](../data/temporal-contract.md) |
| `uv run python -m gatefall.data.timegrid build [--dataset le2i] [--force]` | Constrói a grade de 10 fps | manifesto + labels → `data/processed/le2i/frames.parquet`; muta atomicamente; preserva existente; `--force` reconstrói | Sim | Não | [Contrato temporal](../data/temporal-contract.md#como-executar) |
| `uv run python -m gatefall.data.timegrid report [--dataset le2i]` | Valida e relata a grade reconstruída | manifesto + labels → stdout; não muta; repetível | Sim | Não | [Contrato temporal](../data/temporal-contract.md#como-executar) |
| `uv run python -m gatefall.data.windows selftest [--dataset le2i]` | Testa janelamento/padding | Casos sintéticos → stdout; não muta; repetível | Não | Não | [Janelamento](../data/temporal-contract.md#janelamento) |
| `uv run python -m gatefall.data.windows report [--dataset le2i]` | Relata contagens reais de janelas | `frames.parquet` → stdout; não muta; repetível | Sim | Não | [Janelamento](../data/temporal-contract.md#janelamento) |
| `uv run python -m gatefall.data.frames_io selftest [--dataset le2i]` | Testa leitura FFmpeg sintética; requer FFmpeg | Vídeo sintético → stdout/temporários descartados; não muta o dataset | Não | Não | [Contrato temporal](../data/temporal-contract.md) |
| `uv run python -m gatefall.data.frames_io report [--dataset le2i]` | Confere leitura de quadros reais | manifesto + vídeos → stdout; não muta; repetível | Sim | Não | [Contrato temporal](../data/temporal-contract.md) |

## Pose, features e janelas

| Sintaxe | Propósito e pré-requisitos | Entrada → saída; mutação e idempotência | Dados | GPU/pesos | Detalhes |
| --- | --- | --- | --- | --- | --- |
| `uv run python -m gatefall.pose.extract extract --video-id ID [--model MODEL] [--dataset le2i] [--force]` | Extrai pose de um vídeo | vídeo + modelo → HDF5 por vídeo; muta; preserva válido; `--force` reextrai | Sim | Pesos; GPU recomendada | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.extract extract-all [--model MODEL] [--dataset le2i] [--force]` | Extrai pose de todos os vídeos da grade | vídeos + modelo → `data/features/le2i/pose/*.h5`; muta; pula válidos; `--force` reextrai | Sim | Pesos; GPU recomendada | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.extract report [--dataset le2i]` | Valida cobertura e arquivos de pose | grade + HDF5 → stdout; não muta; repetível | Sim | Não | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.smoke report [--video-id ID] [--model MODEL] [--dataset le2i]` | Diagnóstico visual de um vídeo/modelo | vídeo + modelo → relatório/artefatos de scratch; muta apenas scratch | Sim | Pesos; GPU recomendada | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.loading selftest` | Testa carregamento/imputação de pose | Casos sintéticos → stdout; não muta | Não | Não | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.kinematics selftest [--dataset le2i]` | Testa as 134 features cinemáticas | Casos sintéticos → stdout; não muta | Não | Não | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.pose.kinematics report [--dataset le2i]` | Valida features reais | grade + HDF5 → stdout; não muta | Sim | Não | [Contrato temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.data.pose_dataset selftest [--dataset le2i]` | Testa a fonte genérica de janelas | Loader sintético → stdout; não muta | Não | Não | [Dataset de janelas](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.data.pose_dataset report [--dataset le2i]` | Materializa e valida janelas reais | grade + HDF5 → stdout; não muta | Sim | Não | [Dataset de janelas](../data/temporal-contract.md#dataset-de-janelas-de-pose) |
| `uv run python -m gatefall.features.standardize selftest [--dataset le2i]` | Testa layout e z-score; o dataset opcional preserva o contrato uniforme das sub-CLIs | Casos sintéticos → stdout; não muta | Não | Não | [Padronização](../data/pose-standardization.md) |
| `uv run python -m gatefall.features.standardize build [--dataset le2i] [--force]` | Calcula estatísticas só do treino | grade + HDF5 → JSON versionado; muta atomicamente; preserva existente; `--force` recalcula | Sim | Não | [Padronização](../data/pose-standardization.md#como-executar) |
| `uv run python -m gatefall.features.standardize report [--dataset le2i]` | Valida stats e aplicação nos splits | grade + HDF5 + JSON → stdout; não muta | Sim | Não | [Padronização](../data/pose-standardization.md#como-executar) |

## Treino, avaliação e desenvolvimento

| Sintaxe | Propósito e pré-requisitos | Entrada → saída; mutação e idempotência | Dados | GPU/pesos | Detalhes |
| --- | --- | --- | --- | --- | --- |
| `uv run python -m gatefall.train.baseline_a selftest` | Testa TCN e métricas | Casos sintéticos → stdout; não muta | Não | Não | [Treino](../train/baseline-a.md) |
| `uv run python -m gatefall.train.baseline_a train [--dataset le2i] [--run-dir PATH] [--force]` | Treina o braço A; destino deve ser local | grade + HDF5 + stats → config, checkpoint e métricas; staging validado e promoção por diretório; run válido é preservado; parcial falha; `--force` usa backup para rollback | Sim | GPU recomendada | [Treino](../train/baseline-a.md#como-executar) |
| `uv run python -m gatefall.eval.baseline_a_events selftest` | Testa FSM e associação de eventos | Casos sintéticos → stdout; não muta | Não | Não | [Avaliação](../eval/baseline-a-events.md) |
| `uv run python -m gatefall.eval.baseline_a_events evaluate [--dataset le2i] [--run-dir PATH] [--force]` | Avalia checkpoint local completo; requer runtime POSIX para `fcntl.flock` | run + grade + HDF5 → protocolo e métricas de eventos; lock e journal protegem a publicação conjunta; preserva par válido; `--force` substitui o par local | Sim | GPU recomendada e checkpoint | [Avaliação](../eval/baseline-a-events.md#como-executar) |
| `uv run python -m gatefall.eval.qualitative selftest` | Testa desenho de esqueleto/bbox, pose imputada e decodificação de vídeo em uma única passada | Casos sintéticos → stdout; não muta | Não | Não | [Diagnóstico qualitativo](../eval/qualitative-render.md) |
| `uv run python -m gatefall.eval.qualitative render [--dataset le2i] [--run-dir PATH] [--split {val,test,both}] [--force] [--include-false-alarms]` | Renderiza PNGs dos quadros reais nos gatilhos de alarme detectados; fora do pipeline e da CI por depender de vídeo bruto | run + grade + HDF5 + vídeo bruto → PNGs em `figures/`; somente leitura contra artefatos do run; não toca lock/journal; sem `--force` preserva PNG já existente; `--include-false-alarms` só adiciona PNGs de alarme falso (prefixo `falsealarm__`), sem alterar a saída padrão | Sim | GPU recomendada e checkpoint | [Diagnóstico qualitativo](../eval/qualitative-render.md#como-executar) |
| `uv run python -m gatefall.runs_selftest` | Testa proteção de referências e o lifecycle transacional da avaliação, incluindo lock, journal e rollback | Temporários sintéticos → stdout; não muta o projeto | Não | Não | [Runbook](../runbooks/pipeline-a.md) |
| `uv run pyright` | Verificação estática obrigatória após Python | Fontes → stdout; não muta | Não | Não | [Tecnologias](../architecture/technology-stack.md) |
| `uv run mkdocs build --strict` | Valida e constrói o site | `docs/` + `mkdocs.yml` → `site/`; muta saída gerada; repetível; sem `--force` | Não | Não | [Início](../index.md) |
| `uv run mkdocs serve` | Serve documentação local para edição | Docs → servidor local; não altera fontes | Não | Não | [Início](../index.md) |

Na GitHub Actions, o job/check a configurar futuramente como obrigatório para
`main` é **CI / validation**. Ele roda selftests sintéticos, Pyright e o build
estrito, sem dataset real, pesos ou GPU.
