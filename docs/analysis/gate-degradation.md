# Análise pós-hoc de degradação do gate B1/C1

O comando `gatefall.eval.analysis.gate_degradation` analisa checkpoints B1 e C1
já treinados. Ele executa as mesmas condições fixas em `val` e `test`, sem
ajustar pesos, estatísticas de padronização, limiares ou protocolo de alarme.
O resultado de `test` é descritivo; nenhuma severidade ou configuração é
escolhida com esse split. Comportamento não monotônico do gate é mantido como
resultado da medição.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.gate_degradation selftest
uv run python -m gatefall.eval.analysis.gate_degradation analyze \
  --dataset le2i --output-dir data/analysis/gate-degradation \
  --arm both
```

`--arm B1` e `--arm C1` executam apenas uma arma. Os runs padrão são
`runs/local/{dataset}/baseline_b1` e `baseline_c1`; use `--b1-run-dir` e
`--c1-run-dir` para outros runs locais íntegros. B1 usa `--repo-dir` e
`--weights` para o DINOv3 congelado. C1 usa `--runtime-dir` e
`--sam3-checkpoint` para o runtime isolado do SAM 3. Os valores padrão desses
caminhos seguem as variáveis de ambiente e os resolvedores das respectivas
extrações. A análise confere os pesos e a revisão do DINOv3 com os HDF5 de
origem; para C1, confere checkpoint, lock, revisão e tipo de autocast do SAM 3
com a proveniência congelada. É necessário ter vídeo bruto, HDF5, estatísticas,
checkpoints e pesos dos backbones acessíveis. GPU é recomendada.

O diretório de saída é obrigatório e separado dos runs e das fontes
canônicas. Sem `--force`, arquivos de análise existentes interrompem a
execução. `--force` substitui somente os JSON/CSV dedicados dessa análise.
O comando não escreve em `checkpoint.pt`, `classification_report.json`,
`event_metrics.json`, `alarm_protocol.yaml`, HDF5 nem arquivos de referência.

## Execução em shards

A varredura completa de C1 pode exceder o limite de uma sessão de GPU, pois
cada severidade de blur não nula reinfere o SAM 3 em todos os quadros de
`val` e `test`. O subcomando `shard` avalia apenas as condições escolhidas de
uma arma e grava um artefato parcial; `merge` reúne shards completos nas
mesmas saídas canônicas de `analyze`.

Cada condição é identificada por `split:modalidade:severidade`, com
`modalidade` em `clean`, `pose` ou `visual`. `*` seleciona todos os splits,
modalidades ou severidades da grade canônica, e uma condição fora dela
interrompe a execução. Para dividir as quatro severidades caras de C1 entre
as duas T4 do Kaggle, use um processo por GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m gatefall.eval.analysis.gate_degradation shard \
  --arm C1 --dataset le2i --output-dir /kaggle/working/shards/gpu0 \
  --conditions '*:clean:0' '*:pose:*' '*:visual:0' '*:visual:2' '*:visual:3'
CUDA_VISIBLE_DEVICES=1 uv run python -m gatefall.eval.analysis.gate_degradation shard \
  --arm C1 --dataset le2i --output-dir /kaggle/working/shards/gpu1 \
  --conditions '*:visual:6' '*:visual:12'
uv run python -m gatefall.eval.analysis.gate_degradation merge \
  --shards /kaggle/working/shards/gpu*/c1_gate_degradation.shard-*.partial.json \
  --output-dir data/analysis/gate-degradation
```

`shard` aceita `--run-dir` e os mesmos argumentos de backbone, dataset e
`--force` de `analyze`. Cada condição selecionada é avaliada uma única vez e
nenhuma condição fora da seleção reinfere o SAM 3. O runtime do SAM 3 ainda
é aberto em cada split do shard para conferir a proveniência congelada, mesmo
sem severidade não nula. O shard é a unidade atômica: ele só é gravado ao
fim da sua seleção, e uma interrupção não deixa artefato aceito. Para limitar
o trabalho perdido em uma interrupção, use shards menores, como uma
severidade por processo (`'*:visual:2'`) ou um split por severidade
(`'val:visual:2'`), executados em sequência na mesma GPU.

Os arquivos de shard se chamam
`{arma}_gate_degradation.shard-{hash}.partial.json`/`.csv`, em que o hash
identifica a seleção. O JSON marca `canonical: false` e `partial: true`,
registra as condições declaradas, o nome e o SHA-256 do trace e não contém
`delta_from_clean`. O `merge` exige proveniência idêntica entre shards
(dataset, run, checkpoint, configuração, grade, estatísticas, gate,
protocolo, versões e proveniência do backbone/SAM 3) e cobertura completa das
condições limpas, de pose e visuais de `val` e `test`. Ele rejeita shards
com condição declarada ausente, campo de métrica faltando, trace truncado ou
com hash divergente, e duplicatas conflitantes; duplicatas idênticas são
aceitas uma vez. Em seguida, restaura a ordem canônica de condições e linhas
do trace e só então calcula os deltas contra a condição limpa. O resultado é
idêntico ao de uma execução `analyze` sem interrupção. As regras de
`--force`, a escrita atômica e a proteção contra diretórios de fontes
canônicas valem para `shard` e `merge`.

## Condições congeladas

Cada split tem uma condição limpa, cinco condições de pose e cinco condições
visuais. A severidade zero em cada modalidade repete exatamente as entradas
limpas e permite conferir os deltas de referência.

| Modalidade | Degradação | Severidades | Processamento |
| --- | --- | --- | --- |
| Pose | Dropout de keypoints | `0, 4, 8, 12, 16` pontos por quadro | Zera coordenadas e confiança de um subconjunto determinístico e aninhado dos 17 pontos no dado bruto, em memória; recalcula os 134 descritores e `q_pose`. |
| Visual B1 | Box blur | raios `0, 2, 3, 6, 12` em 224 × 224 | Aplica o blur existente ao quadro RGB redimensionado; recalcula DINOv3 e `q_visual`. |
| Visual C1 | Box blur | raios `0, 2, 3, 6, 12` em pixels nativos | Aplica o blur existente ao quadro RGB nativo; reinfere SAM 3 com prompt `person`, seleção causal por continuidade, descritor `V_t` e `q_sam3`. |

Uma modalidade permanece limpa enquanto a outra é degradada. A condição
limpa usa as fontes e a inferência normais da avaliação B1/C1. A qualidade
entra no gate sem padronização, na ordem `[q_pose, q_visual]` (C1 usa
`q_sam3` no segundo canal). O peso `g_t` multiplica a projeção da pose;
`1-g_t` multiplica a projeção visual. As normas após o gate são diagnósticos
mecânicos, não medidas causais de importância de modalidade.

## Saídas e interpretação

O diretório escolhido recebe `b1_gate_degradation.json`/`.csv` e/ou
`c1_gate_degradation.json`/`.csv`. Cada linha do CSV representa um quadro
único de vídeo em uma condição: identificadores, qualidade crua dos dois
canais, `g_t`, `1-g_t`, normas `||g_t E_P||₂` e `||(1-g_t) E_V||₂`, pesos
lineares e viés do gate. Os diagnósticos por quadro são suficientes porque
as projeções e o gate operam por timestep antes da TCN; um quadro repetido
em janelas sobrepostas produz os mesmos valores. As estatísticas do JSON
agregam quadros únicos, e o campo `gate_statistics_unit` registra essa
unidade.

Cada condição no JSON traz média, mediana, p05 e p95 de qualidade, peso e
normas, além de macro-F1 restrito às classes do protocolo, sensibilidade
por evento, número de alarmes falsos, FA/h, latência de evento e deltas
contra a condição limpa do mesmo split. Quando nenhum evento é detectado,
a latência média e seu delta são `null`. O relatório registra hashes do
checkpoint, da configuração, da grade e das estatísticas, parâmetros do
gate, proveniência do backbone, protocolo de alarme e grades fixas. O
protocolo e a implementação das métricas são os mesmos da avaliação
congelada B1/C1.

Esta análise exige muitas reinferências do backbone e pode demorar; para C1,
prefira a execução em shards. Os selftests usam somente dados sintéticos. Os resultados reais dependem dos
artefatos locais e não são produzidos pelo comando `selftest`.
