# Robustez de inferência à perda de detecção nas dobras por sujeito (Etapa 4D)

`src/gatefall/eval/analysis/subject_fold_dropout_robustness.py` é um
diagnóstico **somente leitura, exploratório e não canônico**. Ele mede quanto
os checkpoints já treinados das [dobras por sujeito do
Le2i-CS](../train/subject-folds.md) perdem quando a entrada do grupo retido
sofre a mesma família de perda de detecção usada no treino da [Etapa
4C](../train/subject-fold-detection-dropout.md).

A Etapa 4C avaliou o retido só com entrada limpa. Esta etapa complementa
aquela comparação: ela mede a robustez a perdas simuladas na inferência. Nada
é treinado, ajustado ou selecionado. As configurações A/B/C, os splits
oficiais, os HDF5 de pose, os runs das Etapas 4B e 4C, `folds.json`, as
estatísticas e os resumos não mudam. `val` e `test` oficiais nunca são lidos.

## Modelos e condições

Os seis checkpoints congelados do braço A, seed 42, checkpoint da última
época:

- `baseline` (Etapa 4B) e `detection_dropout` (Etapa 4C) de cada uma das três
  dobras, `retain_s0`, `retain_s1` e `retain_s3-4-6-8`.

Cada checkpoint é avaliado só no grupo retido da própria dobra, em stride 1,
em duas condições:

- **Limpa.** São as mesmas janelas, a mesma estatística da dobra, a mesma
  ordem e o mesmo tamanho de batch de `final.val`.
- **Degradada.** É uma única condição predeclarada. Cada janela retida é
  sorteada com probabilidade 0,20. Quando sorteada, exatamente três quadros
  reais consecutivos da janela de 24 viram detecções ausentes, numa posição
  uniforme entre os deslocamentos válidos.

A condição degradada reusa as funções causais validadas da Etapa 4C
(`valid_offsets`, `mask_person_detections` e `masked_window_features`). A
máscara é aplicada a uma cópia de `PoseArrays` antes da imputação, do dt
efetivo e das 134 features. As features são recalculadas sobre o prefixo do
vídeo até o último quadro da janela.

## Máscara de avaliação

A máscara de cada janela vem só da sua identidade e de uma seed fixa:
`numpy.random.default_rng` é semeado por (`EVALUATION_STREAM_KEY = 0x4D`,
`EVALUATION_SEED = 0`, `video_key`, `k_end`). `video_key` é formado pelos
primeiros 8 bytes, em big-endian, do SHA-256 UTF-8 de `video_id`. O sorteio
não depende do modelo, do tamanho de batch nem da ordem de iteração. Ele
também não consome o RNG global do NumPy, do PyTorch ou do `random`.

As janelas limpas, as janelas degradadas e os rótulos de cada dobra são
materializados uma vez, antes de qualquer modelo. Os dois checkpoints da
dobra recebem os mesmos arrays. O relatório grava os SHA-256 do plano de
máscara, dos rótulos e das duas entradas, e a execução falha se eles
divergirem entre os modelos.

## Validação antes dos resultados

Antes de relatar qualquer número, o diagnóstico:

1. valida a partição oficial do `train` do Le2i-CS e os identificadores de
   sujeito;
2. reconcilia `folds.json` com o contrato congelado, a composição atual e o
   SHA-256 dos HDF5 de pose;
3. carrega a estatística de pose de cada dobra pela proveniência gravada;
4. valida os dois runs de cada dobra com a mesma checagem do resumo da
   Etapa 4C: configuração, `fold_run.json`, integridade do checkpoint,
   suporte por classe, intervenção e ligação do run 4C ao seu controle;
5. confere, para cada vídeo retido, que as features recalculadas da pose
   bruta são idênticas às features limpas;
6. exige que a avaliação limpa reproduza exatamente a matriz de confusão, o
   suporte, a Macro-F1 restrita e a F1 por classe de `final.val`, no mesmo
   dispositivo em que foram gravados.

Qualquer divergência interrompe a análise sem gravar nada. Os SHA-256 de
frames, manifesto, `folds.json`, estatísticas e artefatos dos seis runs são
conferidos de novo antes da publicação.

## O que é relatado

Para cada dobra e cada modelo, separadamente:

- Macro-F1 restrita, F1 e suporte por classe restrita, limpa e degradada;
- a mudança, degradado menos limpo, e a perda, limpo menos degradado;
- o subconjunto afetado, formado pelas janelas efetivamente mascaradas. Ele
  traz acurácia limpa e degradada, predições mudadas, janelas que viraram
  erro ou acerto e a F1 por classe, só quando a classe tem pelo menos 20
  janelas afetadas.

Para cada dobra, a diferença pareada das perdas é a perda de `baseline`
menos a perda de `detection_dropout`. Um valor positivo indica que o modelo
da Etapa 4C perde menos. `pooled_score`, `pooled_difference` e
`significance_test` são sempre nulos.

## Saída

```text
runs/local/le2i/subject_folds/
├── folds.json, retain_*/, summary/        # das Etapas 4B e 4C, só lidos
└── diagnostics/detection_dropout_robustness/
    ├── subject_fold_dropout_robustness.json
    └── subject_fold_dropout_robustness.md
```

A raiz das dobras precisa ficar em `runs/local/le2i/subject_folds/` quando
está dentro do repositório. A saída fica sempre em
`diagnostics/detection_dropout_robustness/`. Sem `--force`, uma saída
existente é preservada.

## Como executar

Checagens sintéticas:

```bash
uv run python -m gatefall.eval.analysis.subject_fold_dropout_robustness selftest
```

Elas cobrem:

- as mesmas entradas degradadas para os dois modelos;
- a máscara determinística, independente da ordem e com o RNG isolado;
- a consistência entre a pose bruta e as features limpas;
- o isolamento do grupo retido de cada dobra;
- a reprodução de `final.val` e a falha quando uma única predição limpa
  diverge;
- a recusa de proveniência corrompida (checkpoint, `fold_run.json` da Etapa
  4C, estatística, `folds.json` e hash de pose);
- a preservação dos caminhos protegidos.

Com os runs das Etapas 4B e 4C, a partir da raiz do checkout que contém os
dados e os runs:

```bash
uv run python -m gatefall.eval.analysis.subject_fold_dropout_robustness analyze
```

Como no treino da Etapa 4C, `config.yaml` registra o caminho relativo da
estatística da dobra. Rode a partir da mesma raiz usada no treino. A
reprodução limpa exige o dispositivo gravado em `metrics.json` (GPU CUDA nos
runs registrados).

## Execução registrada (não canônica)

A execução ocorreu em 2026-10-16, no branch
`feat/subject-fold-dropout-robustness` (base
`analysis/gatefall-improvements`). Ela usou o código do worktree do branch,
com os dados e os runs do checkout principal, a partir da raiz desse checkout,
em GPU CUDA (GTX 1650). O código, incluindo a condição e a seed de avaliação,
foi commitado antes da execução real.

A avaliação limpa reproduziu a matriz, o suporte e a Macro-F1 restrita
gravados nos seis runs. Nenhuma janela sem máscara mudou de predição. Uma
segunda execução sem `--force` foi pulada, e uma execução com `--force`
regravou JSON e Markdown idênticos byte a byte.

Macro-F1 restrita no grupo retido. Cada linha é uma dobra. As linhas não
devem ser agregadas nem comparadas entre si.

| Dobra | Janelas mascaradas | B limpo | B degradado | D limpo | D degradado | Perda B | Perda D | B − D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `retain_s0` | 2053 / 10243 | 0,5584 | 0,5539 | 0,5611 | 0,5568 | 0,0046 | 0,0043 | +0,0003 |
| `retain_s1` | 775 / 3961 | 0,6576 | 0,6515 | 0,6577 | 0,6547 | 0,0061 | 0,0030 | +0,0031 |
| `retain_s3-4-6-8` | 1332 / 6536 | 0,6157 | 0,6152 | 0,6174 | 0,6164 | 0,0005 | 0,0011 | −0,0006 |

B = `baseline` (Etapa 4B); D = `detection_dropout` (Etapa 4C).

A condição degradada custa pouco aos dois modelos: no máximo 0,0061 de
Macro-F1 restrita. Em média, só uma janela em cada cinco recebe a máscara. A
diferença pareada é positiva em `retain_s0` e `retain_s1` e negativa em
`retain_s3-4-6-8`. Ela não passa de 0,0031 em módulo, e não há direção
consistente entre as dobras.

No subconjunto afetado, o modelo da Etapa 4C teve menos janelas que viraram
erro nas três dobras (51 contra 71, 23 contra 30 e 31 contra 34). Em
`retain_s3-4-6-8`, porém, ele também teve menos janelas que viraram acerto,
e sua acurácia degradada ficou abaixo da do controle. `standing`, a classe
restrita mais rara, tem as maiores variações de F1 e não mostra direção
consistente.

Com uma seed por dobra e uma única realização da máscara, essas diferenças
estão na ordem da variação não estimada. Portanto, esta execução não mostra
ganho material de robustez da Etapa 4C nessa condição.

Os SHA-256 de `runs/reference/`, `runs/local/` fora de
`subject_folds/diagnostics/`, `data/processed/`, dos HDF5 de pose em
`data/features/le2i/pose/` e de `src/gatefall/features/stats/` foram iguais
antes e depois. As saídas não são versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `diagnostics/detection_dropout_robustness/subject_fold_dropout_robustness.json` | `3b358b645e0dbf047d6834388e278e92524083089a86f57a1390ba1015cc8a63` |
| `diagnostics/detection_dropout_robustness/subject_fold_dropout_robustness.md` | `d2c86cf9a3cc5f88e1d95851dd03c70fa2abd9d81d291d06924049eae2b292a9` |

## Limitações

- Este diagnóstico é exploratório, somente leitura e não canônico. Ele não
  estima o desempenho no `test` oficial nem verifica o alvo de 0,800.
- Há uma única condição sintética, com lacunas de três quadros em 20% das
  janelas. Ela não reproduz a distribuição real das perdas de detecção, que
  se concentram em `fall` e em Home_01. Ela também não mede lacunas mais
  longas, detecções erradas nem ruído de keypoints.
- O modelo da Etapa 4C foi treinado com a mesma família de perda avaliada
  aqui, com outro fluxo aleatório. Uma perda menor nele mostra robustez à
  condição treinada, não robustez geral.
- Há uma seed por dobra e uma única realização da máscara. Nenhuma das duas
  variações é estimada, e não há teste de significância.
- As três dobras têm tamanhos muito diferentes e não são agregadas.
- As janelas são sobrepostas (stride 1) e não são observações independentes.
  Uma mesma lacuna de quadros pode aparecer em janelas vizinhas mascaradas de
  forma independente.
- O subconjunto afetado é descritivo, e suas classes têm suporte desigual.
- Janelas que terminam antes do terceiro quadro do vídeo ficam limpas, como
  na Etapa 4C.
