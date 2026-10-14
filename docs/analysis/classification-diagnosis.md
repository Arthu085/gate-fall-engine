# Análise — Diagnóstico de erros de classificação

`src/gatefall/eval/analysis/classification_diagnosis.py` é a primeira etapa do
diagnóstico dos erros de classificação das armas A, B0, B1, C0 e C1 em
Le2i-CS (`le2i`) e Le2i-CV (`le2i-cv`). A ferramenta é somente leitura: consome
artefatos já gravados, não roda inferência, não retreina, não reextrai
features, não ajusta hiperparâmetros e não seleciona nem promove checkpoints.
O split `test` é usado apenas de forma descritiva.

Esta etapa não inclui inferência por ambiente ou sujeito, redução de
dimensionalidade, aumento de dados nem mudanças de treino.

## Entradas

| Fonte | Como entra | Protocolo |
| --- | --- | --- |
| Referência canônica | Lida automaticamente de `runs/reference/le2i/<arma>/` (`--reference-root` muda a raiz; `--no-canonical-references` desliga) | Somente Le2i-CS |
| Run local | `--run DATASET ARMA RUN_DIR`, repetível | Declarado pelo usuário |
| Sumário multi-seed | `--multiseed-summary DATASET ARMA JSON`, repetível | Declarado pelo usuário |

Não existe referência canônica versionada para Le2i-CV, e a ferramenta não
supõe que ela exista: em CV, o diagnóstico usa somente runs locais e
sumários fornecidos explicitamente. Uma referência ausente, um protocolo sem
run local, uma arma sem sumário ou uma seed esperada (42–46) ausente
aparecem em `missing_inputs`, sem interromper as demais fontes.

Cada fonte é validada antes de entrar no diagnóstico:

- `config.yaml` deve declarar a arma informada e usar as estatísticas de pose
  do protocolo informado (`pose_le2i_cs.json` ou `pose_le2i_cv.json`);
  `--run` também passa por `validate_local_run_dir`, que recusa referências e
  runs do outro protocolo.
- `metrics.json` deve conter `final.train`, `final.val` e `final.test` com
  `confusion_matrix` 10x10 de inteiros não negativos, `per_class`,
  `support`, `f1_by_class` e `macro_f1_restricted`.
- Todas as métricas gravadas são recalculadas a partir da `confusion_matrix`
  com `classification_summary` e `restricted_macro_f1`
  (`gatefall.train.shared.metrics`); qualquer divergência acima de `1e-9`
  falha. Valores gravados ausentes, não numéricos, `NaN` ou `±Infinity`
  também falham, inclusive no agregado do sumário multi-seed.
- Quando presente, `classification_report.json` deve ter as mesmas matrizes
  e `verification_against_metrics_json.ok: true`.
- O sumário multi-seed deve declarar a arma informada, ter `n_seeds`
  coerente, seeds únicas, `run_dir` por seed que não pertença ao outro
  protocolo e agregado de `macro_f1_restricted` igual ao recálculo. Um
  `run_dir` fora das raízes locais conhecidas não confirma o protocolo; isso
  é registrado em `limitations`.

## Métricas

A Macro-F1 restrita preserva a definição do treino: média simples do F1 das
classes de `RESTRICTED_CLASSES` (`walk`, `fall`, `fallen`, `sit_down`,
`sitting`, `stand_up`, `standing`, `other`). `lie_down` e `lying` têm
precisão, revocação e F1 reportados, mas não entram na média nem no déficit.

- **Por classe:** precisão, revocação, F1, `tp`/`tn`/`fp`/`fn`, suporte e
  fração do suporte, nos splits `train`, `val` e `test`.
- **Déficit da Macro-F1:** `1 − Macro-F1 = Σ_c (1 − F1_c) / 8`. Cada classe
  restrita contribui `(1 − F1_c) / 8`; o ranking dessas parcelas identifica
  as classes que mais derrubam a Macro-F1.
- **Confusões dominantes:** as cinco maiores células fora da diagonal, com
  fração do suporte verdadeiro, fração da coluna predita e fração do total de
  erros. Empates são ordenados pelos ids das classes.
- **Lacunas de generalização:** `train − val`, `train − test` e `val − test`
  da Macro-F1 restrita e do F1 de cada classe restrita.
- **Variação entre seeds** (somente sumários): n, média, desvio-padrão
  amostral, mínimo e máximo, com o mesmo `_aggregate_stats` do
  [sumário multi-seed](multiseed-summary.md), para Macro-F1, métricas por
  classe e lacunas. Também registra o déficit calculado pelo F1 médio, as
  confusões da matriz somada entre seeds e quantas seeds têm cada confusão
  no top-5.

Quando uma referência canônica e uma seed de sumário têm o mesmo checkpoint
SHA-256 ou as mesmas matrizes, a correspondência aparece em
`canonical_reference_matches`.

## Saídas

`--output-dir` recebe três arquivos:

- `classification_diagnosis.json`: definição da Macro-F1, blocos
  `protocols.le2i` e `protocols.le2i-cv` separados por arma e fonte,
  diagnóstico completo por fonte e por seed, `canonical_reference_matches`,
  `missing_inputs`, `observations`, `hypotheses`, `limitations` e
  `input_sha256`, com o SHA-256 de cada arquivo lido.
- `classification_diagnosis.csv`: formato longo com as colunas `dataset`,
  `protocol`, `arm`, `source_kind`, `source`, `seed`, `split`, `record`,
  `entity`, `metric` e `value`. Linhas agregadas de sumário usam `seed=all`
  e métricas como `f1.mean`.
- `classification_diagnosis.md`: resumo legível por protocolo, com tabela
  de Macro-F1 e lacunas, contribuição de cada classe ao déficit em `test`,
  observações confirmadas, hipóteses e entradas ausentes. Com sumário
  multi-seed, o detalhamento usa a média entre seeds; os demais valores ficam
  no JSON e no CSV.

As **observações confirmadas** são valores calculados diretamente dos
artefatos. As **hipóteses** são geradas por regras fixas e não verificadas:
confusão entre fase de transição e estado contíguo, confusão com a classe
residual `other`, suporte em `test` abaixo de 2%, lacuna `train − test` acima
de 0,2 e desvio-padrão do F1 entre seeds acima de 0,05. Esses limiares só
disparam texto; não filtram nem selecionam resultados.

## Garantias de somente leitura

- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, na raiz de referências usada, em `src/`, em `data/`, em
  um `--run` ou no diretório de um sumário de entrada.
- Os arquivos são gravados atomicamente. Sem `--force`, se algum deles já
  existir, o comando não grava nada e informa os arquivos encontrados.
- O SHA-256 de cada entrada é calculado antes da gravação e conferido depois;
  qualquer mudança faz o comando falhar.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.classification_diagnosis selftest
uv run python -m gatefall.eval.analysis.classification_diagnosis diagnose --output-dir PATH \
  [--run DATASET ARMA RUN_DIR ...] [--multiseed-summary DATASET ARMA JSON ...] \
  [--reference-root PATH | --no-canonical-references] [--force]
```

`selftest` usa somente artefatos sintéticos e cobre cálculo das métricas,
interpretação da matriz de confusão, validação de entrada, recusa de
métricas gravadas não finitas, dados ausentes e proteção das referências. Exemplo com os artefatos locais desta máquina:

```bash
uv run python -m gatefall.eval.analysis.classification_diagnosis diagnose \
  --output-dir runs/local/classification_diagnosis \
  --multiseed-summary le2i A runs/local/le2i/baseline_a_multiseed/multiseed_summary.json \
  --multiseed-summary le2i B0 runs/local/le2i/b0_fusion_multiseed/multiseed_summary.json \
  --multiseed-summary le2i B1 runs/local/le2i/b1_adaptive_gate_multiseed/multiseed_summary.json \
  --multiseed-summary le2i C0 runs/local/le2i/baseline_c0_multiseed/multiseed_summary.json \
  --multiseed-summary le2i C1 runs/local/le2i/c1_adaptive_gate_multiseed/multiseed_summary.json \
  --run le2i-cv A runs/local/le2i_cv/baseline_a \
  --multiseed-summary le2i-cv A runs/local/le2i_cv/multiseed_summaries/baseline_a/multiseed_summary.json
```

Repita o par `--run le2i-cv`/`--multiseed-summary le2i-cv` para B0, B1, C0 e
C1. GPU, checkpoints e dataset bruto não são necessários.

## Resultado da execução real

Executado sobre as cinco referências canônicas de Le2i-CS, os cinco runs
locais de Le2i-CV com seed 42 e os dez sumários multi-seed (seeds 42–46).
Todas as métricas gravadas conferiram com o recálculo, e `missing_inputs`
ficou vazio. Em Le2i-CS, cada referência canônica corresponde à seed 42 do
respectivo sumário, com checkpoint e matrizes iguais. Os valores abaixo são
médias entre as cinco seeds.

| Protocolo | Arma | Macro-F1 val | Macro-F1 test | Lacuna train − test | Desvio-padrão em test |
| --- | --- | --- | --- | --- | --- |
| Le2i-CS | A | 0,6484 | 0,6109 | 0,2561 | 0,0137 |
| Le2i-CS | B0 | 0,6802 | 0,6558 | 0,3413 | 0,0143 |
| Le2i-CS | B1 | 0,6885 | 0,6722 | 0,3217 | 0,0089 |
| Le2i-CS | C0 | 0,6617 | 0,6391 | 0,3464 | 0,0071 |
| Le2i-CS | C1 | 0,6813 | 0,6254 | 0,3524 | 0,0179 |
| Le2i-CV | A | 0,5485 | 0,4317 | 0,4513 | 0,0235 |
| Le2i-CV | B0 | 0,5763 | 0,2295 | 0,7690 | 0,0720 |
| Le2i-CV | B1 | 0,5808 | 0,2742 | 0,7217 | 0,0554 |
| Le2i-CV | C0 | 0,5270 | 0,4042 | 0,5819 | 0,0199 |
| Le2i-CV | C1 | 0,5161 | 0,4131 | 0,5672 | 0,0176 |

Observações confirmadas:

- **Le2i-CS:** `standing` e `other` são as duas maiores parcelas do déficit
  em `test` nas cinco armas. Juntas, respondem por 49% a 50% do déficit;
  `standing` tem F1 médio entre 0,02 (C1) e 0,14 (B1), com 43 janelas em
  `test`. A confusão dominante em `test` é `other→walk`, entre 36% e 41% do
  suporte de `other`, e está no top-5 das cinco seeds em todas as armas. Em
  `val`, dominam `standing→walk` (53% a 90% do suporte de `standing`) e
  `sitting→stand_up` (22% a 29% do suporte de `sitting`).
- **Le2i-CV:** a queda de `val` para `test` é maior que no CS em todas as
  armas. Em B0/B1, `stand_up`, `sitting` e `walk` são previstos como `other`
  (43% a 58% do suporte de `stand_up` e `sitting`), e o F1 médio de
  `standing` em `test` é 0. Em A/C0/C1, `sitting→walk` e `stand_up→walk`
  dominam (40% a 50% do suporte de cada classe). Em `val`, a confusão mais
  frequente parte de `other` em todas as armas.

Hipóteses não verificadas: o suporte escasso de `standing` pode dominar seu
F1; confusões com a classe residual `other` podem refletir sua
heterogeneidade; e o colapso de B0/B1 para `other` em Le2i-CV pode indicar
dependência das features DINOv3 em relação ao ambiente. Separar essas causas
exige a análise por ambiente e sujeito, fora desta etapa. O relatório
completo não é versionado; o comando de exemplo acima o reproduz a partir dos
mesmos artefatos.

## Limitações

- As métricas são por janela com `eval_stride` 1. Janelas vizinhas se
  sobrepõem, então as contagens não equivalem a eventos, vídeos ou sujeitos
  independentes.
- A variação entre seeds descreve a inicialização e a otimização, não a
  incerteza amostral entre sujeitos (ver
  [Bootstrap agrupado por sujeito](grouped-bootstrap.md)).
- Le2i-CS retém sujeitos e Le2i-CV retém ambientes; os dois blocos nunca são
  agregados e não são diretamente comparáveis.
