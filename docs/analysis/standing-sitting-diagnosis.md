# Análise — Erros `standing→walk` e `sitting→stand_up` em desenvolvimento (Etapa 6A)

`src/gatefall/eval/analysis/standing_sitting_diagnosis.py` pergunta se a
evidência de desenvolvimento do Le2i-CS sustenta uma intervenção de treino
específica e controlada contra os dois erros compartilhados por A, B0, B1, C0
e C1 no `val` oficial: `standing→walk` e `sitting→stand_up` (ver a
[decisão sobre intervenções](intervention-decision.md#evidencia-confirmada-em-train-e-val)).
A alternativa é concluir que os dados são insuficientes ou que o erro é uma
limitação não resolvida. A pergunta é motivada pelo alvo de pesquisa de
0,800 de Macro-F1 restrita, que esta etapa não estima nem verifica.

O diagnóstico é **exploratório, não canônico e somente leitura**. Ele não
retreina, não reextrai features, não ajusta nem seleciona nada e não altera
checkpoints, protocolos, estatísticas de padronização ou artefatos
existentes. Não repete as investigações encerradas de `other→walk` (Etapas
3A–3C), perda simulada de detecção (Etapas 4A–4D) e amostragem balanceada por
sujeito (Etapa 5A).

## Evidência usada

Nenhuma linha, feature, pose ou predição do `test` oficial é lida. O bloco
`final.test` de `metrics.json` só tem a presença conferida pelo validador
compartilhado dos runs.

| Fonte | Unidade | O que é reaproveitado |
| --- | --- | --- |
| `train` oficial | Janela em `TRAIN_STRIDE` (4) | Rótulos por quadro e pose bruta. As janelas precisam coincidir com `build_window_index`, e o suporte por classe precisa coincidir com `final.train` de cada run. Só a matriz agregada de `final.train` descreve o ajuste ao treino. |
| `val` oficial | Janela em `EVAL_STRIDE` (1), 25 runs (cinco braços × seeds 42–46) | Caminho validado do [diagnóstico estratificado](stratified-error-diagnosis.md): reinferência do checkpoint só em `val`, alinhamento exato com `build_window_index` e reprodução exata de `final.val`. |
| Dobras por sujeito | Janela em `EVAL_STRIDE` do grupo retido, controle do braço A, seed 42 | Validadores das [Etapas 4B](../train/subject-folds.md) e [4D](subject-fold-dropout-robustness.md): `folds.json`, estatística da dobra, proveniência, checkpoint e reavaliação limpa que reproduz exatamente `final.val` do run da dobra. |

Os grupos retidos das dobras são sujeitos do próprio `train` oficial, então
servem como sujeitos fora do treino sem ler o `test`.

## Categorias de janela e regras

Cada janela recebe uma única categoria, nesta precedência:

| Categoria | Definição |
| --- | --- |
| Pose ausente | Menos de metade dos 24 índices de `window_frame_indices`, replicados pela borda inclusive, com `person_found` |
| Transição | Janela misturada, que cruza a fronteira inicial do trecho anotado, ou quadro final a menos de 5 quadros (0,5 s) do fim de um trecho seguido por outro |
| Interior | As demais |

O suporte limitado é propriedade da classe, não da janela. A unidade das
regras é o sujeito bruto fora do treino: o sujeito 5 do `val` oficial e cada
sujeito dos grupos retidos.

| Regra | Condição |
| --- | --- |
| `R0_class_support` | Suporte limitado se a classe verdadeira tem menos de 2% das janelas rotuladas do `train` oficial (`LOW_SUPPORT_FRACTION` do [diagnóstico agregado](classification-diagnosis.md)) ou menos de três unidades com 20 ou mais janelas verdadeiras |
| `R1_transition` | Em uma unidade, a taxa de erro do par em transição supera a do interior em pelo menos 0,10, e a transição tem pelo menos metade dos erros do par |
| `R2_pose_absence` | O mesmo, comparando pose ausente com pose presente |

Uma unidade é elegível para R1 ou R2 com 10 ou mais janelas verdadeiras nos
dois estratos. No `val` oficial, a unidade atende quando pelo menos 80% dos
braços atendem com a média entre seeds. Uma regra é sustentada com pelo
menos três unidades elegíveis e dois terços delas atendendo; com menos de
três, é não avaliável.

Veredito por par: R0 limitado leva a `insufficient_data`; R1 sustentada leva
a `consider_controlled_intervention`, com a candidata
`transition_margin_exclusion` (excluir só do treino as janelas de
transição, com a mesma regra nos cinco braços); R1 não avaliável leva a
`insufficient_data`; nos demais casos, `unresolved_limitation`. R2 é só
informativa, porque a robustez à perda de detecção já foi testada e
encerrada. O veredito geral é `consider_controlled_intervention` se algum
par o sustenta, `insufficient_data` se todos os pares têm dados
insuficientes e `unresolved_limitation` nos demais casos.

As regras e os limiares foram commitados antes da execução sobre os dados
reais. Os suportes por classe já eram conhecidos das etapas anteriores; a
divisão por categoria, não.

## Resultados observados

Execução registrada abaixo. Taxas do `val` oficial são médias ± desvio
entre cinco seeds; nas dobras, uma seed.

**Suporte e ajuste ao treino.** `standing` tem 173 janelas no `train`
oficial (3% das rotuladas), de 22 vídeos e só três sujeitos: 0 (127), 1 (18)
e 6 (28). `sitting` tem 696 (13%), de cinco sujeitos. Em `final.train`, os
dois pares têm erro de 0% a 1,7% em todos os runs, inclusive no treino de
cada dobra. No treino da dobra `retain_s0`, `standing` cai para 46 janelas.

**`standing→walk`.** Quase todos os erros de `standing` vão para `walk`.

| Evidência | A | B0 | B1 | C0 | C1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val`, todas as 96 janelas | 64% ± 6% | 90% ± 6% | 84% ± 8% | 53% ± 15% | 55% ± 5% |
| `val`, transição (72) | 57% ± 4% | 93% ± 3% | 87% ± 5% | 50% ± 12% | 46% ± 4% |
| `val`, interior (24) | 86% ± 13% | 82% ± 16% | 73% ± 20% | 64% ± 38% | 81% ± 11% |

No braço A das dobras, a taxa de erro é 41% no sujeito 0 (205 de 496
janelas; transição 46%, interior 34%), 14% no sujeito 1 (70 janelas, todas
de transição) e 47% no sujeito 6 (107 janelas; transição 48%, interior
40%). Nenhuma janela `standing` do `val` ou dos retidos tem pose ausente. No
`val`, erros e acertos têm pose em todos os índices (mediana), e a
confiança média dos keypoints vai de 0,87 a 0,89 nos erros e de 0,85 a 0,91
nos acertos, conforme o braço.

**`sitting→stand_up`.**

| Evidência | A | B0 | B1 | C0 | C1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `val`, todas as 270 janelas | 29% ± 5% | 22% ± 2% | 24% ± 3% | 27% ± 7% | 26% ± 6% |
| `val`, transição (134) | 18% ± 3% | 15% ± 4% | 17% ± 3% | 19% ± 5% | 19% ± 5% |
| `val`, interior (136) | 40% ± 8% | 29% ± 3% | 30% ± 5% | 34% ± 9% | 34% ± 7% |

No braço A das dobras, a taxa de erro é 5% no sujeito 0 (1748 janelas), 5%
no sujeito 1 (409), 19% no sujeito 3 (257), 21% no sujeito 4 (61) e 2% no
sujeito 6 (257). Só 20 janelas `sitting` dos retidos, todas do sujeito 0,
têm pose ausente, nenhuma delas com erro.

**Regras.**

| Par | R0 | R1 (unidades que atendem / elegíveis) | R2 | Veredito |
| --- | --- | --- | --- | --- |
| `standing→walk` | não limitado: 3%, 4 unidades | `not_sustained` (1 / 3) | `not_evaluable` | `unresolved_limitation` |
| `sitting→stand_up` | não limitado: 13%, 6 unidades | `not_sustained` (0 / 5) | `not_evaluable` | `unresolved_limitation` |

Em `standing→walk`, só o sujeito 0 atende R1; no `val`, B0 e B1 atendem e
A, C0 e C1 não, e o sujeito 1 não tem janelas de interior. Em
`sitting→stand_up`, a taxa de erro do `val` é maior no interior que na
transição nos cinco braços.

**Veredito geral: `unresolved_limitation`.** A evidência de desenvolvimento
não sustenta nenhuma intervenção de treino específica, e nenhuma é
recomendada.

## Interpretação

**Observado em desenvolvimento:**

- Os dois pares são erros de generalização para sujeitos fora do treino, não
  de ajuste: o treino quase não os comete, e os sujeitos retidos e o `val`
  sim.
- Pose ausente não acompanha os erros: as janelas dos dois pares quase
  sempre têm pose, e a confiança dos keypoints dos erros é próxima à dos
  acertos.
- A concentração em transições não é consistente. Em `standing→walk`, a
  direção muda entre braços e sujeitos. Em `sitting→stand_up`, os erros do
  `val` ficam mais no interior do trecho.
- A taxa de erro de `standing→walk` varia muito entre sujeitos (14% a 47% no
  braço A das dobras) e entre braços no `val` (53% a 90%). B0 e B1 têm as
  maiores taxas no `val`, como já registrado.

**Hipóteses não testadas aqui:**

- `standing` vem de só três sujeitos no `train`, 73% das janelas do sujeito
  0. Pouca diversidade de sujeitos em `standing` é compatível com o erro, mas
  a regra predeclarada de suporte não marca a classe como limitada, e esta
  etapa não separa essa hipótese de diferenças de ação, câmera ou anotação
  entre sujeitos. A amostragem balanceada por sujeito da Etapa 5A, que não é
  específica de classe, já foi encerrada sem ganho.
- `standing` e `walk` podem diferir, em janelas de 2,4 s, por pistas que a
  pose e os descritores visuais congelados não separam bem, como pausas
  curtas ou passos lentos. Esta etapa não inspeciona os quadros de vídeo.
- Em `sitting→stand_up`, erros no interior do trecho são compatíveis com
  movimento de levantar anotado como `sitting` ou com movimento parecido
  dentro do trecho sentado, sem leitura dos quadros.

**Recomendação:** não implementar nem ajustar intervenção a partir desta
evidência. Uma nova intervenção só caberia com uma pergunta pré-declarada
que desloque outra etapa, aplicada igualmente a A, B e C e avaliada sem o
`test` já exposto.

## Garantias de somente leitura

- Só `train` e `val` oficiais do Le2i-CS entram; runs de Le2i-CV são
  recusados. A reinferência oficial pede só o split `val` ao avaliador, e a
  pose bruta é aberta só para vídeos de `train` e `val`.
- As predições do `val` precisam reproduzir exatamente `final.val` de cada
  run, nos níveis de ambiente, sujeito e vídeo; as dos retidos, a matriz, o
  suporte e a Macro-F1 restrita de `final.val` do run da dobra. As contagens
  de cada par, somadas por categoria, sujeito, ambiente, vídeo e trecho
  vizinho, precisam reproduzir a linha da classe verdadeira na matriz
  gravada.
- Run sem `config.yaml`, `metrics.json` ou `checkpoint.pt`, braço ou seed não
  fornecido, `folds.json` ausente e controle de dobra incompleto viram
  evidência ausente, sem resultado inventado. Divergência de integridade
  interrompe a execução sem gravar nada.
- O SHA-256 de `frames.parquet`, do manifesto, dos arquivos dos runs, de
  `folds.json`, das estatísticas e runs das dobras e dos HDF5 de pose de
  `train` e `val` é calculado antes e conferido antes da gravação.
- `--output-dir` não pode coincidir, conter ou estar contido em
  `runs/reference/`, `src/`, `data/`, `docs/`, nos diretórios de frames e
  pose, na raiz das dobras ou em qualquer `--run`. Dentro do repositório, só
  `runs/local/` é aceito. Sem `--force`, nada é sobrescrito.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.standing_sitting_diagnosis selftest
uv run python -m gatefall.eval.analysis.standing_sitting_diagnosis diagnose \
  --run le2i A runs/local/le2i/baseline_a_multiseed/seed_42 \
  --run le2i B0 runs/local/le2i/b0_fusion \
  ... \
  [--fold-root runs/local/le2i/subject_folds] \
  --output-dir runs/local/standing_sitting_diagnosis/le2i_cs [--force]
```

Os 25 runs seguem a convenção do
[contexto de janela](window-context-diagnosis.md#como-executar): A em
`baseline_a_multiseed/seed_<seed>`; B0, B1, C0 e C1 em `b0_fusion`,
`b1_adaptive_gate`, `c0_fusion` e `c1_adaptive_gate` para a seed 42 e com o
sufixo `_seed43` … `_seed46` para as demais. São necessários os checkpoints,
`frames.parquet`, o manifesto, os HDF5 de pose, as features de cada braço e
os runs da Etapa 4B. Como nas Etapas 4C e 4D, rode a partir da raiz do
checkout que contém os runs das dobras, porque `config.yaml` registra o
caminho relativo da estatística de cada dobra. A reprodução das dobras exige
o dispositivo gravado em `metrics.json`.

A saída tem `standing_sitting_diagnosis.json` (suportes, composição do
`train`, runs do `val` com reconciliação, agregados por braço, dobras,
regras, veredito, evidência ausente, limitações e SHA-256 das entradas),
`standing_sitting_diagnosis.md` e `standing_sitting_windows.csv` (uma linha
por janela focal e por modelo, com categoria, pose e predição; as janelas do
`train` não têm predição).

`selftest` usa só dados sintéticos e cobre categorias e sua precedência,
margem final só com trecho seguinte, fração de detecção com replicação de
borda, contagens e destinos por par, reconciliação com a matriz gravada,
alinhamento entre predições e atributos, limiares e elegibilidade das regras,
concordância entre braços, vereditos e uma execução ponta a ponta com dobras
sintéticas treinadas: leitura só de `train` e `val`, evidência ausente,
determinismo, entradas inalteradas, `--force`, saídas protegidas e falha
após divergência com `metrics.json`.

### Execução registrada (não canônica)

2026-10-16, branch `feat/standing-sitting-diagnostic` (base
`analysis/gatefall-improvements`), com o código do worktree do branch, os
dados e runs do checkout principal, a partir da raiz desse checkout, em GPU
CUDA (GTX 1650), em cerca de 40 s. Por isso, as chaves de `input_sha256` do
JSON são caminhos absolutos. Os 25 runs do `val` e os três controles das
dobras reproduziram exatamente as matrizes gravadas, e o suporte de
`final.train` dos 25 runs coincidiu com as janelas do `train`. Os SHA-256 de
`runs/reference/`, `runs/local/le2i/`, `data/processed/le2i/` e
`src/gatefall/features/stats/` (373 arquivos) foram iguais antes e depois.
Uma segunda execução gerou arquivos idênticos byte a byte. As saídas não são
versionadas:

| Arquivo | SHA-256 |
| --- | --- |
| `standing_sitting_diagnosis.json` | `328588e2e2736bcce7ef35cccbef197a8130f34da72da49157d9e8ce2377a01f` |
| `standing_sitting_diagnosis.md` | `785ed7c7b8ed19ea6dce60e9fb860511a6739266cf4d98e3bc3fbe76c8f98142` |
| `standing_sitting_windows.csv` | `cbae5be8a8b0de5b18c9fb448268c549094a17ca42bdd3f035439c5a5e573957` |

## Limitações

- Associação não é causa. As categorias coincidem com posição no vídeo,
  ação, sujeito, ambiente e decisões de anotação.
- O `val` oficial tem um único sujeito. As dobras só têm o braço A com uma
  seed, então as unidades do `val` e das dobras vêm de modelos diferentes e
  não são comparadas em magnitude.
- Janelas sobrepostas não são observações independentes. A concordância
  entre unidades é descritiva, sem teste de significância.
- Predições por janela do `train` não existem; o ajuste ao treino vem só da
  matriz agregada de `final.train`. Não há predição retida de B0, B1, C0 e C1
  nas dobras.
- A pose descreve o artefato YOLO-Pose bruto. `person_found` e a confiança
  dos keypoints não medem a pose imputada nem a entrada visual de B0, B1, C0
  e C1.
- A reprodução exata das matrizes é necessária, mas não prova que cada
  janela recebeu historicamente o mesmo rótulo.
- Os limiares são convenções fixadas antes da execução; outros limiares
  podem levar a outro veredito.
- Le2i-CV fica fora do escopo, e o conteúdo dos quadros de vídeo não foi
  inspecionado.
