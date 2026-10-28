# Estudo comparativo 2026

Infraestrutura do estudo comparativo do plano de 23/10/2026. A especificação congelada está em
`specs/003-gatefall-comparative-study-2026/`. Esta página explica o desenho; os comandos
reais ficam no [roteiro de execução](comparative-study-runbook.md).

!!! warning "Resultados parciais"
    Só as matrizes `le2i` ([resultados](#resultados-matriz-le2i)) e `motion`
    ([resultados](#resultados-matriz-motion)) foram executadas com dados reais, e seus
    resultados são exploratórios. As matrizes `upfall`, `transfer` e `ablations` e o perfil de
    extração seguem pendentes. Os selftests validam apenas contratos sintéticos e não medem
    desempenho.

## Datasets

| Dataset | Papel | Partição | Fonte |
| --- | --- | --- | --- |
| Le2i-CS / Le2i-CV | comparação com o trabalho A/B/C | OmniFall `cs` / `cv` (inalteradas) | [Le2i](../data/le2i.md) |
| UP-Fall | primário | `upfall-subject-v1`, por sujeito | [UP-Fall e GMDCSA-24](../data/staged-datasets.md) |
| GMDCSA-24 | teste externo reservado | só `test` | [UP-Fall e GMDCSA-24](../data/staged-datasets.md) |
| CAUCAFall v4 | fallback documentado | — | não implementado |

## Métodos

Os oito métodos primários treinam pelo mesmo `gatefall.experiments.physics train`. Todos usam o
backend pose138 + evidence8 e diferem só pela fonte visual e pela fusão:

| Método | Braço / fusão | Visual |
| --- | --- | --- |
| A | `A` / `pose` | — |
| B0 | `B0` / `concat` | DINOv3 |
| B1 learned | `B1` / `learned` | DINOv3 |
| C0 | `C0` / `concat` | SAM 3 |
| C1 learned | `C1` / `learned` | SAM 3 |
| B1 physics | `B1` / `physics` | DINOv3 |
| C1 physics | `C1` / `physics` | SAM 3 |
| TCNTE adaptado | `TCNTE` / `tcnte` | — |

Learned e physics compartilham a mesma representação (pose138). Por isso, a diferença entre
eles mede apenas o gate e nunca uma mudança de features. As cinco configurações históricas
sobre pose134 permanecem como referência congelada em [Referências do Le2i](../reference/le2i-runs.md)
e não são retreinadas.

As ablações de gate (`constant`, `quality_concat`, `learned`, `monotonic`, `physics`, para B1 e C1)
ficam fora da tabela primária e são comparadas com a variante `constant`. O
[gate físico](../train/physics-gate.md) e o [aprendizado de movimento](motion-physics.md) têm
páginas próprias.

### TCNTE adaptado ao protocolo do estudo

A implementação é independente, a partir de Yu et al. (2025) e da arquitetura publicada em
`SomeOtherScenery/fall_detection` (commit `23b7d32`). Esse repositório não declara licença,
então nenhum código foi copiado. O relatório usa sempre o rótulo “TCNTE adaptado ao protocolo do
estudo”.

| Aspecto | Publicado | Adaptado |
| --- | --- | --- |
| Janela | 18 quadros | 24 quadros a 10 fps, terminando em t |
| Entrada | 12 articulações (x, y), min–max por quadro | as mesmas, calculadas do bloco `kp_xy` do pose138 (equivalente ao min–max em pixels) |
| Ausência | não tratada | quadro zerado |
| Convolução | não causal | preenchimento só à esquerda |
| Saída | binária, softmax antes da focal | dez classes; focal (γ = 3) sobre log-softmax |
| Seleção | melhor acurácia de teste | melhor macro-F1 restrito (8 classes) de validação, como todos os métodos |
| Treino | Adam 1e-4, 100 épocas | Adam 1e-4, 30 épocas, batch 64 |

## Protocolo congelado

`uv run python -m gatefall.study protocol` imprime o protocolo e seu SHA-256. O protocolo fixa:

- janela de 24 quadros a 10 fps;
- passo 4 no treino e 1 na avaliação;
- sementes 42–46;
- 30 épocas, batch 64 e `best_val`;
- o protocolo de alarme do braço A: `fall` e `fallen` positivos, 3 quadros consecutivos,
  refratário de 5 s;
- as partições;
- a transferência sem adaptação;
- os parâmetros do bootstrap.

Qualquer mudança altera o hash, e o relatório recusa misturar hashes.

## Métricas

As métricas são recalculadas das predições por janela de cada `evaluation-*.json`:

- **Por classe:** precisão, revocação, F1 e suporte.
- **Macro-F1:**
    - `macro_f1_supported` (primário): média sobre as classes com suporte no teste, conjunto
      fixado pelos rótulos e idêntico entre os métodos;
    - `macro_f1_restricted8`: só para comparação histórica.
- **Eventos:** detectados, perdidos e sensibilidade, separando a detecção **durante a queda**
  (alarme dentro do segmento `fall`) da detecção **só após a queda** (alarme na janela de
  associação depois do fim de `fall`).
- **Falsos alarmes:** contagem, FA por hora de vídeo avaliado (denominador: janelas de passo 1
  divididas por 10 fps) e FA por hora rotulada.
- **Latência:** média e mediana dos detectados, mais o número de perdidos. A mediana sobre todos
  os eventos trata as perdas como latência infinita e só existe quando mais da metade dos
  eventos foi detectada.
- **Custo:**
    - extração em s/quadro, medida por `profile-extraction` em diretório isolado e com
      aquecimento excluído;
    - classificação em s/janela, medida na avaliação;
    - ponta a ponta, que é a soma das duas.

## Incerteza

- **Diferenças pareadas por semente:** tratamento − referência na mesma semente e no mesmo
  pacote, com média, desvio e contagem de sinais.
- **Bootstrap por cluster:**
    - A unidade primária é o sujeito em todos os datasets. A gravação é a unidade secundária e é
      sempre relatada ao lado (`*_recording_ci_*`, `cluster_role = secondary`).
    - O sujeito de cada vídeo vem dos metadados existentes, a coluna `subject` de
      `frames.parquet`. Nos datasets encenados, ela é conferida contra o identificador. Um vídeo
      com mais de um sujeito, ou um conjunto de teste diferente do avaliado, é recusado.
    - No UP-Fall, as duas câmeras de um ensaio formam uma única gravação.
    - O mesmo sorteio de clusters vale para os dois métodos e para todas as sementes. Assim,
      janelas sobrepostas nunca contam como observações independentes.
    - O cálculo usa 2000 reamostras e IC percentil de 95%.
- **Poucos sujeitos:**
    - O teste tem 2 sujeitos no Le2i-CS (7 e 31 vídeos), 5 no Le2i-CV, 4 no UP-Fall e 4 no
      GMDCSA-24.
    - Com 2 a 5 clusters, o bootstrap tem poucas réplicas distintas e o IC percentil por sujeito
      subestima a variabilidade entre populações. No Le2i-CS, ele é dominado por um único sujeito.
    - O IC por sujeito deve ser apresentado como largo e descritivo. O IC por gravação é mais
      estreito porque ignora a correlação entre gravações do mesmo sujeito.
- **Sem teste de significância automático.**

## Registro e proveniência

- `gatefall.study plan` monta todos os passos com saídas, dependências e identidade de run.
- `run` executa só com `--execute`. Nessa execução:
    - todo treino termina antes de qualquer avaliação de teste;
    - um passo só é pulado quando suas saídas conferem: integridade, proveniência, cobertura e
      identidade das dependências (a preparação de cada ZIP, as anotações pinadas, os hashes do
      manifesto e da grade, os arquivos do pacote e o hash da grade de origem, o hash do pacote
      no treino e na avaliação, o checkpoint e `selection_performed = false`);
    - uma saída existente que não confere interrompe a execução sem sobrescrever nada;
    - uma saída ausente é refeita, de modo que uma execução interrompida retoma do primeiro passo
      pendente;
    - com `--stage`, os pré-requisitos transitivos fora dos estágios escolhidos são conferidos
      contra o plano completo antes de qualquer execução, cada um uma única vez, e nunca são
      executados. Um pré-requisito ausente, pendente ou que não confere recusa a execução com o
      passo e o comando a rodar. Um treino conferido sem `study-provenance.json` também a recusa;
    - a execução para na primeira falha;
    - cada execução gera um ledger JSONL.
- Logo após o treino, cada run recebe um `study-provenance.json` escrito uma única vez. Ele
  registra a identidade, o Git, os pesos (SHA-256), o pacote, o hash da partição, os hashes de
  artefatos, o ambiente e os tempos.
- `report` reaplica o verificador de integridade de cada avaliação antes de agregar. Assim,
  uma transferência cujo pacote GMDCSA-24 atual não confere com `target_manifest_sha256` é
  recusada.
- `report` gera:
    - `summary.md`, `summary.csv`, `per_class.csv`, `contrasts.csv`, `runs.csv` e `cost.csv`;
    - figuras SVG de macro-F1 por semente e de sensibilidade × FA/h;
    - linhas do tempo qualitativas (verdade, predições e alarmes);
    - `report-manifest.json`.

## Movimento: runs externos

Os 150 runs históricos da matriz de movimento foram treinados sobre pacotes e alvos anteriores ao
estudo. Seus `manifest_sha256` e `targets_sha256` apontam para esses manifestos, e não para os do
estudo; por isso o reuso comum os recusa. `reuse-audit` compara cada run candidato com o passo
planejado (schema, receita, semente, pacote, alvos e hashes):

- só runs **compatíveis**, copiados para o destino planejado, são reusados;
- os **incompatíveis** permanecem como evidência histórica separada.

### Atestado de equivalência (opt-in)

Um run histórico só fica compatível com um atestado explícito, criado por
`gatefall.study attest-equivalence`. O atestado liga o pacote e os alvos originais aos atuais do
estudo e fica em `runs/local/<protocolo>/study2026/equivalence/motion-<visual>.json`, escrito uma
única vez. Nenhum manifesto, hash registrado ou artefato do run é reescrito.

Ao atestar, o comando relê do disco todo HDF5 dos dois pacotes e todo NPZ dos dois alvos. Ele
recusa o atestado inteiro quando há:

- arquivo ausente, arquivo não listado ou SHA-256 real diferente do registrado;
- registro por vídeo diferente: identidade, partição, `K`, caminho, hash, hash da pose de
  origem ou ordem;
- `video_id` ou caminho repetido (mapeamento ambíguo);
- alvos que não apontam (`bundle_sha256`) para o pacote do mesmo lado;
- qualquer diferença de metadados fora da allowlist. A comparação cobre schema, dataset,
  modalidade, dimensão visual, classes, evidências, fps, `preparation_config`, fontes
  (`frames_sha256`, papéis e hashes em `files`, proveniência SAM 3) e os parâmetros dos alvos
  (horizontes, confiança, juntas, coordenadas).

Os alvos atuais também precisam carregar sobre o pacote atual (`load_targets`).

| Manifesto | Diferença admitida |
| --- | --- |
| pacote | `code_provenance` pode diferir |
| pacote | `splits` pode ser adicionado se for `train/val/test`, as partições que o carregador exigia sem o campo |
| pacote | `videos[].bytes` pode ser adicionado; o valor é conferido contra o tamanho real |
| pacote | `sources.pose_extractor` e `sources.dinov3_extractor` podem ser adicionados se ausentes no original |
| alvos | `code_provenance` pode diferir |
| alvos | `bundle_sha256` é remapeado: cada manifesto aponta para o seu pacote |

O atestado embute o texto dos quatro manifestos. Assim, o uso o reconfere sem os artefatos
originais e sem confiar no resumo gravado:

- os hashes dos textos embutidos;
- a comparação refeita, que precisa dar a mesma lista de diferenças admitidas;
- os hashes do pacote e dos alvos atuais em disco;
- o `manifest_sha256` e o `targets_sha256` do run, que precisam ser os originais atestados.

O atestado entra no fluxo do estudo assim:

- a verificação de treino, `reuse-audit` e o executor aceitam o run como `reused-verified`. O
  `study-provenance.json` registra o resumo e o SHA-256 do atestado, e o executor recusa
  registrar um run atestado como `executed`;
- `evaluate --equivalence` do módulo de movimento carrega o run sobre o pacote atual. O plano
  sempre passa esse caminho, que só é consultado quando o run não usa o pacote atual. Sem
  atestado válido, a recusa estrita de sempre continua valendo;
- a avaliação registra `equivalence` (resumo do atestado), e a verificação de avaliação exige
  esse registro igual ao atestado atual. Por isso, uma avaliação histórica, feita sobre o pacote
  original, é recusada: copie só `config.json`, `metrics.json`, `stats.json`, `checkpoint.pt` e
  `last.pt`, e refaça a avaliação;
- `report` aplica as mesmas verificações e publica `origin` e `equivalence_sha256` em `runs.csv`
  e em `report-manifest.json`.

Limites da garantia:

- a garantia são bytes idênticos de entrada do treino, conferidos no momento do atestado;
- depois de atestar, mudanças nos artefatos originais não são detectadas, porque não são mais
  lidos;
- os HDF5 atuais são relidos por `load_bundle` a cada avaliação, e a verificação de passo confere
  os seus tamanhos;
- as fontes em `data/features` não são relidas. Vale o hash registrado nos dois manifestos, que
  precisa ser igual;
- o código, o ambiente e os hiperparâmetros do treino histórico ficam fora da garantia, além dos
  campos da identidade planejada que o reuso comum já confere.

## Resultados: matriz `le2i`

Execução real do usuário, consolidada em
`specs/003-gatefall-comparative-study-2026/validation.md`. O relatório usado é
`runs/local/study2026/reports/le2i-2026-10-28T191059Z/`:

- protocolo `d61f7769cb5c…` e revisão `380a31f` sem alterações locais;
- 80 treinos executados pelo estudo e 80 avaliações de teste novas, todos `executed`, que formam
  16 grupos (8 métodos × 2 protocolos) com 5 sementes (42–46) cada.

Todos os números vêm de `summary.csv`, `contrasts.csv` e `per_class.csv` desse relatório. Um Δ é
a média, entre as 5 sementes, da diferença pareada por semente. A coluna "sementes +/−" conta as
sementes com Δ positivo e negativo; empates ficam de fora.

Os resultados são exploratórios. Nenhum método, gate, semente ou checkpoint foi escolhido por
eles: o checkpoint é sempre o `best_val` congelado, e os alarmes seguem o protocolo congelado do
braço A, sem limiar ajustado. Estes runs usam pose138 e não substituem as
[referências canônicas A/B/C](../reference/le2i-runs.md), treinadas sobre pose134.

### Limites estatísticos

- **Le2i-CS:** o teste tem só 2 sujeitos (7 e 31 vídeos), 38 gravações e 22 quedas. O bootstrap
  por sujeito (primário) tem apenas três reamostragens distintas. Nos oito métodos, o limite
  superior do IC por sujeito é a própria média observada. Ele não serve como evidência, nem
  quando exclui zero.
- **Le2i-CV:** o teste tem 5 sujeitos, 60 gravações e 37 quedas. O IC por sujeito continua
  grosseiro. O `cv` separa ambiente e câmera, não sujeitos: identificadores de sujeito se repetem
  entre treino e teste ([relatório de generalização](../analysis/le2i-cv-generalization.md)).
- **IC por gravação:** é secundário. Ele trata como independentes gravações do mesmo sujeito e
  tende a ser estreito demais.
- **Sementes:** o desvio entre sementes chega a 0.064 de macro-F1 (B1-learned no CV), da ordem
  de várias diferenças entre métodos. Cinco sementes estimam mal essa variabilidade.
- **Eventos:** uma queda vale 0.045 de sensibilidade no CS e 0.027 no CV. Cada protocolo tem
  cerca de 0.17 h de vídeo avaliado, então um falso alarme vale cerca de 6 FA/h.
- **Comparações múltiplas:** são 7 contrastes × 4 métricas × 2 protocolos, cada um com dois IC,
  sem correção e sem teste de significância. Alguns IC que excluem zero são esperados por acaso.
- **Escopo:** CS e CV são duas partições do mesmo Le2i. O CV é uma mudança composta de ambiente
  dentro do Le2i (ambiente, câmera, resolução, fps e qualidade de pose variam juntos), não uma
  validação entre datasets, e não isola nenhum desses fatores.
- **`lie_down` no CV:** a classe tem 5 janelas de teste e nenhuma de treino, e todos os métodos
  têm F1 zero nela. Por isso o macro-F1 com suporte do CV (9 classes) fica abaixo do restrito a 8
  classes, também listado. No CS, os dois coincidem.

### Desempenho por método

Médias entre as 5 sementes. O F1 de `fall` e de `fallen` vem de `per_class.csv` e é descritivo,
sem IC. A latência é a mediana dos eventos detectados, em segundos; "perdidos" é a média de quedas
não detectadas.

**Le2i-CS**

| Método | macro-F1 (suporte) ± dp | F1 `fall` | F1 `fallen` | sens. evento | durante a queda | FA/h | latência (s) | perdidos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.594 ± 0.029 | 0.694 | 0.754 | 0.818 | 0.791 | 208.9 | 0.34 | 4.0 |
| B0 | 0.640 ± 0.005 | 0.758 | 0.881 | 0.827 | 0.827 | 51.4 | 0.30 | 3.8 |
| B1-learned | 0.657 ± 0.020 | 0.772 | 0.860 | 0.927 | 0.927 | 60.7 | 0.29 | 1.6 |
| B1-physics | 0.644 ± 0.012 | 0.749 | 0.829 | 0.891 | 0.891 | 68.9 | 0.30 | 2.4 |
| C0 | 0.629 ± 0.016 | 0.745 | 0.853 | 0.955 | 0.955 | 75.9 | 0.30 | 1.0 |
| C1-learned | 0.628 ± 0.027 | 0.744 | 0.846 | 0.927 | 0.927 | 105.1 | 0.28 | 1.6 |
| C1-physics | 0.633 ± 0.010 | 0.739 | 0.854 | 0.900 | 0.900 | 91.1 | 0.30 | 2.2 |
| TCNTE | 0.321 ± 0.041 | 0.427 | 0.655 | 0.909 | 0.827 | 130.7 | 1.26 | 2.0 |

**Le2i-CV**

| Método | macro-F1 (suporte) ± dp | macro-F1 (8) | F1 `fall` | F1 `fallen` | sens. evento | durante a queda | FA/h | latência (s) | perdidos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.403 ± 0.019 | 0.454 | 0.643 | 0.696 | 0.827 | 0.816 | 168.5 | 0.40 | 6.4 |
| B0 | 0.264 ± 0.022 | 0.297 | 0.531 | 0.693 | 0.681 | 0.600 | 234.7 | 0.50 | 11.8 |
| B1-learned | 0.268 ± 0.064 | 0.302 | 0.540 | 0.704 | 0.778 | 0.665 | 158.9 | 0.43 | 8.2 |
| B1-physics | 0.248 ± 0.037 | 0.279 | 0.438 | 0.686 | 0.676 | 0.503 | 182.9 | 0.60 | 12.0 |
| C0 | 0.417 ± 0.017 | 0.470 | 0.684 | 0.742 | 0.811 | 0.811 | 92.7 | 0.40 | 7.0 |
| C1-learned | 0.424 ± 0.029 | 0.478 | 0.685 | 0.740 | 0.795 | 0.778 | 118.0 | 0.34 | 7.6 |
| C1-physics | 0.413 ± 0.014 | 0.465 | 0.697 | 0.714 | 0.881 | 0.865 | 73.4 | 0.35 | 4.4 |
| TCNTE | 0.209 ± 0.030 | 0.235 | 0.369 | 0.564 | 0.735 | 0.638 | 140.8 | 1.15 | 9.8 |

### Contrastes pareados predefinidos

| Contraste | Protocolo | Δ macro-F1 (suporte) ± dp | sementes +/− | IC sujeito | IC gravação | Δ sens. evento | Δ FA/h |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 − A | CS | +0.045 ± 0.030 | 5/0 | [+0.027, +0.059] | [+0.007, +0.083] | +0.009 | −157.6 |
| B0 − A | CV | −0.140 ± 0.033 | 0/5 | [−0.158, −0.077] | [−0.169, −0.097] | −0.146 | +66.2 |
| C0 − A | CS | +0.035 ± 0.038 | 4/1 | [−0.018, +0.051] | [+0.009, +0.062] | +0.136 | −133.1 |
| C0 − A | CV | +0.014 ± 0.020 | 3/2 | [−0.021, +0.033] | [−0.014, +0.036] | −0.016 | −75.8 |
| B1-learned − B0 | CS | +0.017 ± 0.020 | 4/1 | [+0.008, +0.021] | [−0.001, +0.037] | +0.100 | +9.3 |
| B1-learned − B0 | CV | +0.004 ± 0.071 | 4/1 | [−0.007, +0.008] | [−0.009, +0.014] | +0.097 | −75.8 |
| C1-learned − C0 | CS | −0.002 ± 0.014 | 3/2 | [−0.004, +0.018] | [−0.014, +0.011] | −0.027 | +29.2 |
| C1-learned − C0 | CV | +0.007 ± 0.020 | 4/1 | [−0.005, +0.014] | [−0.011, +0.025] | −0.016 | +25.3 |
| B1-physics − B1-learned | CS | −0.013 ± 0.018 | 2/3 | [−0.016, −0.006] | [−0.029, +0.001] | −0.036 | +8.2 |
| B1-physics − B1-learned | CV | −0.020 ± 0.068 | 1/4 | [−0.036, −0.002] | [−0.032, −0.006] | −0.103 | +24.1 |
| C1-physics − C1-learned | CS | +0.005 ± 0.028 | 3/2 | [−0.001, +0.005] | [−0.010, +0.018] | −0.027 | −14.0 |
| C1-physics − C1-learned | CV | −0.011 ± 0.015 | 2/3 | [−0.015, +0.004] | [−0.031, +0.005] | +0.086 | −44.5 |
| TCNTE − A | CS | −0.273 ± 0.042 | 0/5 | [−0.273, −0.256] | [−0.311, −0.237] | +0.091 | −78.2 |
| TCNTE − A | CV | −0.194 ± 0.045 | 0/5 | [−0.228, −0.144] | [−0.224, −0.152] | −0.092 | −27.7 |

**Inversão de `B0 − A`.** O sinal da concatenação DINOv3 muda entre os protocolos:

- no CS, B0 ganha 0.045 de macro-F1 (5/0 sementes, os dois IC acima de zero) e reduz FA/h em
  157.6 (0/5; o IC por gravação fica abaixo de zero, o IC por sujeito cruza zero). A sensibilidade
  por evento quase não muda (+0.009);
- no CV, B0 perde 0.140 de macro-F1 (0/5, os dois IC abaixo de zero). A sensibilidade por evento
  cai 0.146, a sensibilidade durante a queda cai 0.216 (0/5), e FA/h sobe 66.2 (os dois IC acima
  de zero). O F1 de `fall` cai de 0.643 para 0.531;
- nas médias do CV, os três métodos DINOv3 ficam abaixo de A (0.248 a 0.268 contra 0.403), e os
  três SAM 3 ficam perto dele (0.413 a 0.424). `C0 − A` no CV é +0.014 (3/2, os dois IC cruzam
  zero), com FA/h 75.8 menor (os dois IC abaixo de zero).

Como a mudança do CV é composta, o relatório não diz qual fator produz a inversão.

**`B1-physics − B1-learned` negativo nos dois protocolos:**

- **CS:** −0.013 (2/3 sementes). O IC por sujeito fica abaixo de zero, mas é degenerado; o IC por
  gravação cruza zero ([−0.029, +0.001]). A sensibilidade por evento cai 0.036 (0/3, dois empates),
  e FA/h sobe 8.2.
- **CV:** −0.020 (1/4), com os dois IC abaixo de zero. A sensibilidade durante a queda cai 0.162
  (0/5; IC por gravação [−0.263, −0.065]), e parte das detecções passa para depois da queda (de
  0.114 para 0.173). FA/h sobe 24.1, com dp de 172.3 entre sementes, e o F1 de `fall` cai de 0.540
  para 0.438.

Em nenhum dos dois protocolos o gate físico supera o aprendido em B1.

**C1 no CV: operação × macro-F1.** Em `C1-physics − C1-learned`:

- o macro-F1 cai 0.011 (2/3), e os dois IC cruzam zero;
- a sensibilidade por evento sobe 0.086 (4/1; IC por sujeito [+0.043, +0.123], por gravação
  [+0.044, +0.135]), e as quedas perdidas caem de 7.6 para 4.4;
- FA/h cai 44.5 (1/4; IC [−51.8, −30.0] e [−64.3, −26.8]), de 118.0 para 73.4.

O gate físico em C1 detecta mais quedas com menos falsos alarmes no CV, sem ganho de macro-F1 por
janela. No CS, o mesmo contraste não tem direção comum: +0.005 de macro-F1 (3/2), −0.027 de
sensibilidade (2/3) e −14.0 FA/h (2/2).

**Déficit do TCNTE.** `TCNTE − A` é −0.273 no CS e −0.194 no CV, com 0/5 sementes e os dois IC
abaixo de zero nos dois protocolos:

- o F1 de `fall` é 0.427 no CS e 0.369 no CV, contra 0.694 e 0.643 de A;
- a latência mediana sobe para 1.26 s e 1.15 s, contra 0.34 s e 0.40 s;
- o déficit é sobretudo de classificação por janela. No CS, a sensibilidade por evento é 0.091
  maior (4/1, o IC por gravação cruza zero), mas 0.082 da sensibilidade do TCNTE vem de
  detecções só após a queda, contra 0.027 em A. FA/h fica abaixo do de A nos dois protocolos
  (−78.2 e −27.7), e A tem o maior FA/h do CS (208.9).

O resultado vale para o TCNTE adaptado ao protocolo do estudo (12 articulações, dez classes e 30
épocas), não para o modelo publicado.

**Outros contrastes:**

- **`C0 − A` no CS:** +0.035 (4/1; o IC por sujeito cruza zero, o por gravação não), +0.164 de
  sensibilidade durante a queda (5/0) e −133.1 FA/h (0/5).
- **`B1-learned − B0`:** +0.017 no CS (4/1), com +0.100 de sensibilidade (5/0). No CV, +0.004 (os
  dois IC cruzam zero), com +0.097 de sensibilidade e −75.8 FA/h.
- **`C1-learned − C0`:** macro-F1 perto de zero nos dois protocolos (−0.002 e +0.007, os IC
  cruzam zero), com FA/h maior (+29.2 e +25.3).

**Custo.** O `cost.csv` deste relatório traz só a classificação por janela e o treino. A extração
e o custo ponta a ponta dependem de `profile-extraction`, que não foi executado.

## Resultados: matriz `motion`

Execução real do usuário, consolidada em
`specs/003-gatefall-comparative-study-2026/validation.md`. O relatório usado é
`runs/local/study2026/reports/motion-2026-10-28T164408Z/`:

- protocolo `d61f7769cb5c…` e revisão `b4463d9` sem alterações locais;
- 150 avaliações de runs `reused-verified`, que formam 30 combinações com 5 sementes cada.

Todos os números vêm de `summary.csv`, `contrasts.csv` e `per_class.csv` desse relatório. Um Δ é
a média, entre as 5 sementes, da diferença pareada por semente. A coluna "sementes +/−" conta as
sementes com Δ positivo e negativo; empates ficam de fora.

Cuidados de leitura:

- **Le2i-CS:** o teste tem só 2 sujeitos e 38 gravações. O bootstrap por sujeito (primário) tem
  apenas três reamostragens distintas, e seu IC é degenerado: muitas vezes termina na própria
  média observada. Não serve como evidência.
- **Le2i-CV:** o teste tem 5 sujeitos e 60 gravações. O IC por sujeito continua grosseiro.
- **IC por gravação:** é secundário. Ele trata como independentes gravações do mesmo sujeito e
  tende a ser estreito demais.
- **Escopo:** CS e CV são duas partições do mesmo Le2i, não dois datasets. Nada aqui generaliza
  para outro dataset nem estabelece superioridade de uma receita. Nenhuma receita foi
  selecionada por estes resultados, e eles não substituem as referências canônicas A/B/C.

### Contraste primário: `motion_physics − motion`

| Protocolo | Braço | Δ macro-F1 (suporte) ± dp | sementes +/− | IC sujeito | IC gravação | Δ FA/h |
| --- | --- | --- | --- | --- | --- | --- |
| CS | A | +0.004 ± 0.015 | 3/2 | [−0.034, +0.012] | [−0.010, +0.014] | −2.3 |
| CS | B0 | +0.015 ± 0.013 | 4/1 | [+0.006, +0.015] | [+0.002, +0.028] | −5.8 |
| CS | B1-physics | +0.013 ± 0.009 | 5/0 | [−0.056, +0.019] | [−0.010, +0.037] | +7.0 |
| CS | C0 | +0.011 ± 0.007 | 5/0 | [−0.005, +0.016] | [−0.001, +0.021] | −28.0 |
| CS | C1-physics | −0.015 ± 0.021 | 1/4 | [−0.018, −0.003] | [−0.025, −0.008] | +9.3 |
| CV | A | −0.009 ± 0.037 | 3/2 | [−0.015, 0.000] | [−0.020, +0.001] | +19.3 |
| CV | B0 | +0.007 ± 0.014 | 3/2 | [+0.001, +0.017] | [−0.001, +0.015] | −32.5 |
| CV | B1-physics | −0.015 ± 0.078 | 2/3 | [−0.028, −0.001] | [−0.026, −0.005] | +86.7 |
| CV | C0 | −0.012 ± 0.027 | 1/4 | [−0.017, −0.003] | [−0.020, −0.004] | +8.4 |
| CV | C1-physics | −0.003 ± 0.023 | 3/2 | [−0.008, +0.009] | [−0.011, +0.007] | −15.6 |

**Sem direção comum.** O sinal do contraste muda com o braço e com o protocolo:

- no CS, as médias de A (+0.004, 3/2 sementes, os dois IC cruzam zero), B0, B1-physics e C0
  são positivas. Só C1-physics fica negativo;
- no CV, só a média de B0 é positiva;
- em B1-physics e C0 no CS, as cinco sementes melhoram, mas os dois IC cruzam zero.

Uma média positiva é uma observação, não um efeito estabelecido. No CS, com dois sujeitos, nem um
IC por sujeito acima de zero é evidência.

**B0, melhora localizada.** B0 é o único braço com Δ macro-F1 positivo nos dois protocolos:

- +0.015 no CS (4/1 sementes) e +0.007 no CV (3/2);
- o IC por sujeito fica acima de zero nos dois. O IC por gravação também fica no CS, mas não no
  CV;
- a melhora é relativa a `motion`, que tinha piorado B0 (`motion − baseline` = −0.019 no CS e
  −0.015 no CV). A média de `motion_physics` continua abaixo do `baseline`: 0.635 contra 0.640
  no CS e 0.256 contra 0.264 no CV.

**Efeitos adversos:**

- **C1-physics no CS:** −0.015 (1/4), com os dois IC abaixo de zero, +9.3 FA/h e F1 de `fall` de
  0.741 para 0.710.
- **B1-physics no CV:** −0.015 (2/3), com os dois IC abaixo de zero. FA/h sobe de 167.3 para
  254.0 (Δ +86.7, com dp entre sementes de 196.8) e a sensibilidade por evento cai 0.070.
- **C0 no CV:** −0.012 (1/4), com os dois IC abaixo de zero e +8.4 FA/h.

**F1 de `fall` × macro-F1.** As médias por classe de `per_class.csv` são descritivas, sem IC, e
não acompanham o macro-F1:

- no CS, B0 e B1-physics ganham macro-F1, mas o F1 de `fall` cai 0.003 e 0.008. Os ganhos vêm de
  outras classes;
- no CS, C0 (+0.020) e A (+0.013) ganham em `fall`;
- no CV, o F1 de `fall` sobe 0.026 em B0, com dp 0.050 entre sementes, e cai 0.017 em A e 0.019
  em B1-physics.

**Falsos alarmes:**

- **C0 no CS:** as 5 sementes reduzem FA/h (Δ −28.0, os dois IC abaixo de zero), com +0.045 de
  sensibilidade por evento. Isso só desfaz o aumento de `motion` sobre `baseline` (+32.7), e FA/h
  fica em 80.5 contra 75.9 do `baseline`.
- **B0 no CV:** reduz FA/h em 32.5, mas com dp de 173.4 entre sementes, e continua acima do
  `baseline` (254.0 contra 234.7).
- **A no CV:** aumenta FA/h em 19.3 (4/1 sementes, os dois IC acima de zero) e perde 0.059 de
  sensibilidade por evento.

**Diante do `baseline`.** Na comparação descritiva das médias de `summary.csv`, as diferenças de
macro-F1 entre `motion_physics` e `baseline` nas 10 combinações vão de −0.0244 a +0.0054. O
maior ganho é o de A no CS (0.5941 para 0.5995). Ele é menor que o desvio entre sementes desses
métodos (0.029 e 0.015) e não tem contraste pareado nem IC neste relatório. Nas outras nove
combinações, a diferença é no máximo +0.0021.
