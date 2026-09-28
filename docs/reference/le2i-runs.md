# Referências finais do Le2i

`runs/reference/le2i/` contém a evidência textual versionada dos runs finais
do protocolo `cs`. A promoção usa o run local final de cada braço, com a seed
congelada em `config.yaml`; nenhum braço, época ou artefato foi escolhido pelo
desempenho no teste. Referências são somente leitura e não são destinos de
treino, `report` ou avaliação.

| Braço | Diretório | `run_name` persistido | Classificação | Eventos sob o protocolo congelado |
| --- | --- | --- | --- | --- |
| A | `baseline_a/` | `baseline_a` | Sim | Sim |
| B0 | `baseline_b0/` | `b0_fusion` | Sim | Sim |
| B1 | `baseline_b1/` | `b1_adaptive_gate` | Sim | Sim |
| C0 | `baseline_c0/` | `c0_fusion` | Sim | Não há avaliador final de eventos C0 |
| C1 | `baseline_c1/` | `c1_adaptive_gate` | Sim | Sim |

Os nomes físicos seguem `baseline_*`; os valores de `run_name` mantêm a identidade
experimental original. Um `--run-dir` local explícito com nome legado continua
válido se a configuração persistida corresponder ao braço. Não há migração
automática dos diretórios locais.

## Regra de promoção

Cada diretório contém `config.yaml`, `metrics.json` e
`classification_report.json` do mesmo run final. O par
`alarm_protocol.yaml` e `event_metrics.json` entra apenas quando o braço já
possui avaliação final de eventos sob o protocolo de alarme congelado. A
ausência desse par em C0 significa **avaliação não realizada**, não resultado
zero. As métricas de classificação cobrem `train`, `val` e `test`; as métricas
de eventos cobrem `val` e `test`.

`config.yaml` fixa a receita e os hashes/proveniência das fontes disponíveis
para o braço. `metrics.json` registra o histórico, as métricas finais e os
SHA-256 de `config.yaml` e do checkpoint local. O relatório registra as
matrizes de confusão, métricas por classe, projeção binária e verificação
contra `metrics.json`. `event_metrics.json` registra os SHA-256 do checkpoint,
das métricas de treino e do protocolo. Esses hashes continuam sendo os da
execução original: a promoção não altera valores de métricas nem proveniência.

Os arquivos são copiados dos respectivos `runs/local/le2i/<arma>/`. Apenas
`run_dir` no relatório e `checkpoint_path`/`alarm_protocol_path` nas métricas
de eventos apontam para `runs/reference/le2i/<arma>/`, seguindo a promoção
histórica de A. O caminho do checkpoint na referência é uma indicação de
proveniência; `checkpoint.pt` não é versionado nem está presente ali. A
referência também não contém features, pesos dos backbones, dados brutos,
locks, saídas intermediárias ou diagnósticos locais adicionais.

## Conferência e reprodução

Antes da promoção, cada run local deve passar pelo validador de artefatos da
próprio braço, incluindo compatibilidade e hash do checkpoint. O relatório
deve indicar `verification_against_metrics_json.ok: true`, sem divergências;
seus dados de classificação e hash do checkpoint devem coincidir com
`metrics.json`. Onde há avaliação de eventos, o protocolo e os três hashes
do relatório de eventos devem conferir com os arquivos locais. Após a cópia,
os arquivos sem mudança de caminho devem ser idênticos aos locais; nos JSONs
com caminhos reescritos, somente esses campos podem divergir.

Para reproduzir a inferência ou validar um run completo, prepare o Le2i, as
anotações OmniFall, as features RGB pré-computadas, os arquivos de
padronização e os checkpoints locais correspondentes. Use os comandos das
páginas de cada braço e grave em `runs/local/le2i/`; os validadores de run
exigem `checkpoint.pt` e não operam diretamente sobre a referência textual.
Confira os hashes registrados antes de comparar os resultados. Um retreino
usa a receita de `config.yaml`, mas igualdade bit a bit fora da mesma
máquina/stack de GPU não é garantida. Nenhuma promoção ocorre automaticamente
durante a reprodução.
