# Análise — Diagnóstico estratificado de erros de classificação

`src/gatefall/eval/analysis/stratified_error_diagnosis.py` é a segunda etapa
do diagnóstico dos erros de classificação (a primeira é o
[diagnóstico agregado](classification-diagnosis.md)). Ela quebra os erros das
armas A, B0, B1, C0 e C1 por **ambiente**, por **(ambiente, sujeito)** e por
**vídeo**, em `val` e `test`, nos protocolos Le2i-CS (`le2i`) e Le2i-CV
(`le2i-cv`).

A análise é **exploratória, não canônica e somente leitura**. Ela não
retreina, não reextrai features, não altera checkpoints, não ajusta
hiperparâmetros, não seleciona nem promove modelos e não chama o ciclo de
publicação de `alarm_protocol.yaml`/`event_metrics.json`. O split `test` é
usado apenas de forma descritiva.

## Por que refazer a inferência

`metrics.json` e `classification_report.json` guardam só matrizes de
confusão agregadas, sem a identidade de cada janela. Para estratificar, a
ferramenta reusa o caminho de inferência já validado de cada avaliador de
eventos (`load_event_evaluation` de `gatefall.eval.baseline_{a,b0,b1,c0,c1}`,
via `load_arm_evaluation` do [bootstrap agrupado](grouped-bootstrap.md)) sobre
o checkpoint congelado e obtém `(video_id, k_end, rótulo verdadeiro, rótulo
predito)` por janela. Diferentemente do bootstrap, A aceita qualquer seed de
treino, como as demais armas.

## Entradas e validação

Cada run entra explicitamente com `--run DATASET ARMA RUN_DIR` (repetível).
Não há run padrão nem referência implícita.

- O run precisa de `config.yaml`, `metrics.json` e `checkpoint.pt`, e o
  protocolo precisa de `data/processed/<protocolo>/frames.parquet`. Se algum
  desses arquivos faltar, o run vai para `missing_inputs` com a lista exata e
  nenhuma métrica é gerada para ele. Um `FileNotFoundError` durante a
  inferência, como um arquivo de features ausente, recebe o mesmo
  tratamento. Combinações protocolo × arma não
  fornecidas também aparecem em `missing_inputs`.
- `config.yaml` deve declarar a arma pedida e as estatísticas de pose do
  protocolo; `validate_local_run_dir` recusa referências e runs do outro
  protocolo; o SHA-256 de `checkpoint.pt` deve ser igual ao
  `checkpoint_sha256` de `metrics.json`.
- `final.val` e `final.test` de `metrics.json` passam pela mesma validação
  do diagnóstico agregado (`diagnose_split`): métricas gravadas precisam
  coincidir com o recálculo a partir da `confusion_matrix`.
- Cada predição é ligada à janela exata de `build_window_index` (stride
  `EVAL_STRIDE`, sem descartar `IGNORE_LABEL`) e aos metadados `env` e
  `subject` do vídeo. A análise falha se houver janela sem predição,
  predição sem janela, identidade `(video_id, k_end)` duplicada, rótulo
  verdadeiro divergente, rótulo predito fora de `0..9`, contagem de janelas
  divergente ou vídeo com split, ambiente ou sujeito ambíguo.

Um erro de integridade interrompe a execução sem gravar nada; só arquivos
ausentes são tratados como disponibilidade parcial.

## Métricas e reconciliação

Janelas `IGNORE_LABEL` ficam fora de todas as métricas e são contadas em
`n_ignored_windows`. Para o split inteiro e para cada grupo de cada nível, a
saída traz janelas totais, rotuladas e ignoradas, número de erros, Macro-F1
restrita, suporte, precisão, revocação, F1, `tp`/`fp`/`fn` por classe, matriz
de confusão 10×10 e as cinco confusões dominantes (`dominant_confusions` do
diagnóstico agregado).

A Macro-F1 restrita mantém a definição do treino: média simples do F1 das
oito classes de `RESTRICTED_CLASSES`, inclusive em grupos sem alguma delas.
Uma classe restrita ausente e não predita entra com F1 0, então a Macro-F1 de
um grupo pequeno é mecanicamente menor e não é comparável à do split. Por
isso cada grupo lista `absent_labels`, `absent_restricted_labels` e
`predicted_absent_labels` (classes ausentes que mesmo assim foram preditas).

Para cada split, a matriz recalculada a partir das predições precisa ser
idêntica à `confusion_matrix` de `metrics.json`, e a soma das matrizes, dos
suportes por classe, das janelas e dos erros de cada nível (ambiente,
ambiente × sujeito, vídeo) precisa reproduzir esse total. Qualquer
divergência faz a análise falhar; o resultado fica em `reconciliation`.

## Sujeitos e ambientes

IDs brutos de sujeito não são globais: o mesmo número aparece em vários
ambientes. O nível intermediário usa sempre o par `(env, subject)`, e
`coverage.subject_ids_in_multiple_environments` lista os IDs repetidos por
split. Em Le2i-CV, diferenças entre ambientes misturam câmera, iluminação,
cenário, sujeitos, ações e suporte por classe; elas não isolam efeito causal
de câmera nem de backbone. Em Le2i-CS, os ambientes de `test` têm poucos
sujeitos, então ambiente e sujeito se confundem.

## Saídas

`--output-dir` recebe três arquivos:

- `stratified_error_diagnosis.json`: `canonical: false`, definição da
  Macro-F1, blocos `protocols.le2i` e `protocols.le2i-cv` por arma, com
  seed, `run_name`, `checkpoint_sha256` e `input_sha256` de cada run, e por
  split `overall`, `coverage`, `reconciliation` e `levels`
  (`environment`, `environment_subject`, `video`); além de
  `missing_inputs`, `observations`, `hypotheses`, `limitations` e
  `input_sha256` global.
- `stratified_error_diagnosis.csv`: formato longo com `dataset`, `protocol`,
  `arm`, `seed`, `source`, `split`, `level`, `group`, `env`, `subject`,
  `video_id`, `entity`, `metric` e `value`. `level=split` é o total; células
  de confusão não nulas usam `metric=confusion_count` e
  `entity=verdadeiro->predito`; classes ausentes usam `metric=absent`.
- `stratified_error_diagnosis.md`: resumo com tabelas de Macro-F1
  (erros/janelas rotuladas) por ambiente e por ambiente × sujeito para cada
  arma, IDs de sujeito repetidos, observações confirmadas, hipóteses,
  entradas ausentes e limitações.

As **observações confirmadas** são valores calculados: Macro-F1 do split e
faixa por ambiente, ambiente com mais erros e sua confusão dominante, e
parcela dos erros nos três vídeos com mais erros. As **hipóteses** vêm de
regras fixas e não verificadas: ambiente com Macro-F1 mais de 0,2 abaixo do
split (com ou sem classes restritas ausentes) e vídeo com 25% ou mais dos
erros do split. Esses limiares só disparam texto.

## Garantias de somente leitura

- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, nos diretórios de frames e features dos
  protocolos usados ou em qualquer `--run`. O mesmo run não pode ser
  fornecido duas vezes.
- Os três arquivos são gravados atomicamente. Sem `--force`, se algum deles
  já existir, nada é gravado.
- O SHA-256 de `config.yaml`, `metrics.json`, `checkpoint.pt` e
  `frames.parquet` de cada run disponível é calculado antes da inferência e
  conferido depois da gravação; qualquer mudança faz o comando falhar.
  Arquivos de features não são hasheados um a um.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.stratified_error_diagnosis selftest
uv run python -m gatefall.eval.analysis.stratified_error_diagnosis diagnose \
  --run DATASET ARMA RUN_DIR [--run DATASET ARMA RUN_DIR ...] \
  --output-dir PATH [--force]
```

`selftest` usa somente dados sintéticos e cobre agrupamento, exclusão de
`IGNORE_LABEL`, Macro-F1 restrita fixa, alinhamento de identidade,
reconciliação com `metrics.json`, classes ausentes, IDs de sujeito ambíguos,
runs parciais e proteção da saída. Execução com os runs locais de seed 42
desta máquina:

```bash
uv run python -m gatefall.eval.analysis.stratified_error_diagnosis diagnose \
  --output-dir runs/local/stratified_error_diagnosis \
  --run le2i A runs/local/le2i/baseline_a \
  --run le2i B0 runs/local/le2i/b0_fusion \
  --run le2i B1 runs/local/le2i/b1_adaptive_gate \
  --run le2i C0 runs/local/le2i/c0_fusion \
  --run le2i C1 runs/local/le2i/c1_adaptive_gate \
  --run le2i-cv A runs/local/le2i_cv/baseline_a \
  --run le2i-cv B0 runs/local/le2i_cv/baseline_b0 \
  --run le2i-cv B1 runs/local/le2i_cv/baseline_b1 \
  --run le2i-cv C0 runs/local/le2i_cv/baseline_c0 \
  --run le2i-cv C1 runs/local/le2i_cv/baseline_c1
```

São necessários os checkpoints, `frames.parquet` dos dois protocolos e as
features da arma: pose para todas, DINOv3 para B0/B1, SAM 3 para C0/C1 e
qualidade para B1/C1. GPU é recomendada; na GTX 1650 desta máquina os dez
runs levaram cerca de 50 s.

### Procedimento de validação

1. Rode o `selftest`, `uv run pyright` e `uv run mkdocs build --strict`.
2. Registre o SHA-256 dos arquivos dos runs antes da execução, por exemplo
   com `find RUN_DIR -maxdepth 1 -type f -exec sha256sum {} + | sort`.
3. Execute `diagnose` e confirme que `reconciliation.ok` é `true` em todo
   split de todo run e que `missing_inputs` contém só o esperado.
4. Compare os SHA-256 depois da execução; eles devem ser idênticos.

## Histórico de execuções (não canônico)

Os resultados abaixo são exploratórios, não foram revisados e não
substituem nenhuma referência. Os relatórios não são versionados; o comando
acima os reproduz.

### 2026-10-14 — smoke descritivo, seed 42

Executado no branch `feat/stratified-error-diagnosis` (base
`analysis/gatefall-improvements`) sobre os dez runs locais de seed 42 listados
acima. Todos os splits reconciliaram exatamente com `metrics.json` nos três
níveis, `missing_inputs` ficou vazio e o SHA-256 de todos os arquivos dos dez
runs ficou idêntico antes e depois.

Cobertura: em Le2i-CS, `val` tem um único sujeito (5) em três ambientes e
`test` tem os sujeitos 2 e 7 em seis ambientes, com o sujeito 7 em todos. Em
Le2i-CV, `val` é só `Office` (sujeitos 0, 1, 3 e 7) e `test` é `Home_01` e
`Home_02` (sujeitos 0, 1, 3, 5 e 7).

Macro-F1 restrita em `test` por ambiente:

| Protocolo | Ambiente | A | B0 | B1 | C0 | C1 |
| --- | --- | --- | --- | --- | --- | --- |
| Le2i-CS | Coffee_room_01 | 0,6005 | 0,5846 | 0,5766 | 0,6113 | 0,5822 |
| Le2i-CS | Coffee_room_02 | 0,7014 | 0,6519 | 0,6878 | 0,6879 | 0,6465 |
| Le2i-CS | Home_01 | 0,4532 | 0,4427 | 0,4569 | 0,4668 | 0,4767 |
| Le2i-CS | Home_02 | 0,3218 | 0,4578 | 0,4709 | 0,2874 | 0,3161 |
| Le2i-CS | Lecture_room | 0,5867 | 0,4480 | 0,4685 | 0,3704 | 0,3739 |
| Le2i-CS | Office | 0,6012 | 0,6292 | 0,6558 | 0,6619 | 0,5921 |
| Le2i-CV | Home_01 | 0,3658 | 0,1139 | 0,2712 | 0,3334 | 0,3352 |
| Le2i-CV | Home_02 | 0,4149 | 0,1216 | 0,2281 | 0,3936 | 0,3834 |

Observações confirmadas:

- **Le2i-CS:** `Coffee_room_02` tem a maior Macro-F1 em `test` nas cinco
  armas. `Lecture_room` concentra de 32% a 34% dos erros de `test` em todas
  as armas, com `other→walk` dominante (44% a 57% do suporte de `other`
  nesse ambiente). `Home_02` não tem `fall`, `fallen` nem `standing` em
  `test`, e sua Macro-F1 fica entre 0,29 e 0,47. Os três vídeos com mais
  erros respondem por 21% a 27% dos erros de `test`.
- **Le2i-CV:** a queda em `test` aparece nos dois ambientes retidos. B0 fica
  entre 0,08 e 0,18 em todos os grupos (ambiente, sujeito) de `test`, e a
  confusão dominante em `Home_02` é `sitting→other` (81% do suporte de
  `sitting`; 64% em B1). Em A e C1, a confusão dominante em `Home_01` é
  `sitting→walk` (72% e 76%). Os erros são difusos: os três vídeos com mais
  erros somam de 8% a 10% dos erros de `test`.

Hipóteses não verificadas: em Le2i-CS, parte da Macro-F1 baixa de `Home_02`
é mecânica, porque três classes restritas estão ausentes. Em Le2i-CV, o
colapso de B0/B1 para `other` é uniforme entre ambientes, sujeitos e vídeos
de `test`, compatível com uma mudança de domínio no nível do ambiente. Como o
ambiente retido também muda sujeitos, ações e suporte, isso não demonstra
efeito causal da câmera nem das features DINOv3.

## Limitações

- As métricas são por janela com `eval_stride` 1; janelas vizinhas se
  sobrepõem e não são observações independentes.
- Cada run é uma única seed; não há variação entre seeds nem intervalo de
  confiança (ver [sumário multi-seed](multiseed-summary.md) e
  [bootstrap agrupado por sujeito](grouped-bootstrap.md)).
- Le2i-CS e Le2i-CV nunca são agregados entre si.
