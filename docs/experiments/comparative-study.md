# Estudo comparativo 2026

Infraestrutura do estudo comparativo do plano de 23/10/2026. A especificação congelada está em
`specs/003-gatefall-comparative-study-2026/`. Esta página explica o desenho; os comandos
reais ficam no [roteiro de execução](comparative-study-runbook.md).

!!! warning "Resultados pendentes"
    Nenhuma matriz foi executada com dados reais. Os selftests validam apenas contratos
    sintéticos e não medem desempenho.

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
