# Treino — Dobras internas por ambiente do Le2i-CV (Etapa 2C)

`src/gatefall/train/internal_folds/` implementa o
[candidato 1 da Etapa 2A](../analysis/intervention-decision.md#candidato-1-validacao-interna-por-ambiente-em-le2i-cv-instrumento):
um protocolo de desenvolvimento que treina A, B0 e B1 em três dobras do
`train` oficial do Le2i-CV. Cada dobra retém um ambiente e treina nos outros
dois. A estrutura das dobras foi avaliada sem treino na
[Etapa 2B](../analysis/internal-fold-feasibility.md).

O protocolo é **exploratório e não canônico**. Ele não muda a receita, os
splits oficiais, as features, os runs existentes nem o comportamento padrão
das CLIs de treino. Os resultados das dobras ficam em diretório local próprio
e só servem para decidir se o deslocamento para `other` de B0/B1 aparece em
dados de desenvolvimento.

## O que é reaproveitado sem mudança

- **Receita congelada.** A configuração de cada run sai de
  `BASELINE_A_CONFIG`, `B0_FUSION_CONFIG` e `B1_ADAPTIVE_GATE_CONFIG`. Só
  mudam a seed e os caminhos e hashes das estatísticas de padronização. Em
  B1, `quality_features_sha256` cobre só os 97 vídeos do `train` oficial,
  porque nenhum outro vídeo é lido.
- **Engine e artefatos.** Os runs usam os loops de treino de A, B0 e B1 e os
  mesmos validadores de `config.yaml`, `metrics.json` e `checkpoint.pt`, com
  staging, promoção atômica e recusa de run parcial sem `--force`.
- **Janelamento e rótulos.** `WINDOW_FRAMES`, `TRAIN_STRIDE` (4) no treino de
  desenvolvimento, `EVAL_STRIDE` (1) no ambiente retido, `IGNORE_LABEL` pelo
  último quadro e `RESTRICTED_CLASSES` fixas. Os pesos de classe vêm só das
  janelas de desenvolvimento.
- **Checkpoint da última época.** A curva por época no ambiente retido é só
  registrada; nada é selecionado por ela.
- **Features já extraídas.** Pose, DINOv3 e qualidade são lidos dos HDF5
  existentes. Não há nova extração.

## Dobras e isolamento

- Só as linhas `split == train` de `frames.parquet` e `manifest.parquet` do
  `le2i-cv` são carregadas, com filtro na leitura, e a partição precisa bater
  com o Le2i-CV pinado, com as mesmas guardas da Etapa 2B. `val` e `test`
  oficiais nunca entram em memória.
- As dobras são `retain_Coffee_room_01`, `retain_Coffee_room_02` e
  `retain_Lecture_room`. Na tabela de quadros de cada dobra, a coluna `split`
  vira `development_train` ou `retained`, e cada vídeo fica em um único lado.
- Cada lado recebe loaders de feature restritos aos seus vídeos. Uma leitura
  de outro vídeo interrompe a execução. Os HDF5 de DINOv3 e de qualidade são
  conferidos só para os vídeos do `train` oficial, com os atributos do
  manifesto filtrado.
- O engine exige as posições `val` e `test`. Nas dobras, as duas avaliam o
  mesmo ambiente retido; nenhuma é o split oficial de mesmo nome. Cada run
  grava `fold_run.json` com `evaluation_split_mapping`, a dobra, a arma, a
  seed e os hashes das estatísticas, e o resumo confere que `final.val` e
  `final.test` são idênticos.

## Padronização por dobra

`prepare` ajusta, para cada dobra, as estatísticas de pose e DINOv3 só nas
janelas de desenvolvimento em stride 4, com as mesmas funções das
estatísticas oficiais. Cada dobra grava `stats/pose.json`, `stats/dinov3.json`
e `stats/provenance.json`. A proveniência registra a dobra, os ambientes e
vídeos de desenvolvimento, o número de janelas, o SHA-256 das entradas e o
SHA-256 dos dois arquivos de estatística.

Antes de treinar, a proveniência precisa bater com a dobra atual, com os
hashes atuais dos arquivos e das entradas e com o número de janelas de
desenvolvimento. Estatística de outra dobra, adulterada ou ajustada sobre
entradas antigas é recusada. Os arquivos oficiais `*_le2i_cv.json` não são
lidos nem gravados.

## Diretório e artefatos

O destino padrão é `runs/local/le2i_cv_internal_folds/`, não versionado:

```text
runs/local/le2i_cv_internal_folds/
├── folds.json
├── retain_<ambiente>/
│   ├── stats/{pose,dinov3,provenance}.json
│   └── runs/{baseline_a,baseline_b0,baseline_b1}/seed_<N>/
│       ├── config.yaml, metrics.json, checkpoint.pt
│       └── fold_run.json
└── summary/{pilot,assessment}/internal_fold_summary.{json,md}
```

`folds.json` registra a composição das dobras, o hash das entradas e dos
sidecars de qualidade, a política de checkpoint e o critério congelado.
`train` recusa a execução se frames, manifesto, composição ou qualidade
tiverem mudado desde o `prepare`.

`--output-root` não pode coincidir com, conter ou ficar dentro de
`runs/reference/`, `runs/local/le2i/`, `runs/local/le2i_cv/`, `src/`, `data/`
ou dos diretórios das entradas. Sem `--force`, estatísticas e runs completos e
íntegros são preservados, e estatísticas inválidas, runs parciais e
`fold_run.json` divergente interrompem a execução. O resumo não sobrescreve
uma saída existente sem `--force`.

## Como executar

Checagens sintéticas, sem dataset real:

```bash
uv run python -m gatefall.train.internal_folds selftest
```

O `selftest` cobre a composição das dobras, a leitura restrita a `train`, os
loaders restritos, a proveniência e a recusa de estatísticas, a proteção das
saídas, a determinação do plano e das configurações e o cálculo do critério.
Ele também treina A, B0 e B1 nas três dobras com features sintéticas, para
conferir que o treino só vê o desenvolvimento e que a avaliação só vê o
ambiente retido.

Plano de runs, sem ler dados:

```bash
uv run python -m gatefall.train.internal_folds plan --seed-set pilot
```

Preparar as dobras e as estatísticas, uma vez:

```bash
uv run python -m gatefall.train.internal_folds prepare
```

**Piloto de 9 treinos (seed 42).** Valida o pipeline das dobras. Não aplica o
critério, porque com uma seed não existe variação entre seeds:

```bash
uv run python -m gatefall.train.internal_folds train --seed-set pilot
uv run python -m gatefall.train.internal_folds summarize --seed-set pilot
```

**Avaliação de 45 treinos (seeds 42 a 46).** Os runs da seed 42 do piloto
são reaproveitados se estiverem íntegros. O resumo aplica o critério:

```bash
uv run python -m gatefall.train.internal_folds train --seed-set assessment
uv run python -m gatefall.train.internal_folds summarize --seed-set assessment
```

`train` aceita `--fold` e `--arm` para dividir o trabalho, por exemplo
`--fold retain_Lecture_room --arm B0 B1`. Uma execução interrompida pode ser
retomada com o mesmo comando, porque runs completos são pulados. GPU é
recomendada. `prepare` e `train` aceitam `--frames` e `--manifest`; o padrão
são os arquivos do `le2i-cv`.

## Critério de reprodução congelado

O critério é o da Etapa 2A, fixado antes de treinar e sem opção de CLI:

> O deslocamento é considerado reproduzido se, em pelo menos dois dos três
> ambientes retidos, a menor fração predita como `other` entre as seeds de B0
> (ou B1) superar a maior entre as seeds de A, com o mesmo cálculo da
> consolidação multi-seed.

A fração é a soma da coluna `other` fora da linha `other` da matriz de
confusão do ambiente retido, em stride 1, dividida pelas janelas rotuladas
fora de `other`. A comparação é estrita e feita dentro de cada dobra, para B0
e B1 separadamente. O resumo relata as dobras que satisfazem a comparação por
arma e se B0 ou B1 alcança duas das três.

O critério só é aplicado ao conjunto `assessment`. No `pilot`, o resumo traz
as frações por run e marca o critério como `not_applicable`.

## Estado

A implementação foi validada só com dados sintéticos. Nenhum treino real foi
executado. O piloto e a avaliação exigem aprovação e execução explícitas.

## Limitações

- Cada dobra treina com dois ambientes em vez de três, com cerca de dois
  terços das janelas e outra composição de classes, sujeitos e cenas. As
  magnitudes não são comparáveis às do Le2i-CV oficial.
- A proporção de `other` no ambiente retido vai de 7% a 39% entre dobras. Por
  isso o critério só compara armas dentro da mesma dobra.
- A F1 por classe no ambiente retido depende de um a três vídeos para algumas
  classes e não deve ser lida como resultado. A macro-F1 do retido aparece no
  resumo só como contexto.
- As frações são calculadas por janela, sobre janelas sobrepostas.
- Todo o `train` oficial tem 320×240 e 25 fps. As dobras não testam a
  hipótese de resolução ou fps do `test` oficial. Um resultado negativo não
  identifica a causa da falha no `test`.
- IDs brutos de sujeito se repetem entre ambientes. As dobras são disjuntas
  por ambiente, mas não comprovadamente por sujeito.
- O `test` oficial já foi exposto. Mesmo um resultado positivo nas dobras
  continua exploratório e não confirma, de forma independente, a
  generalização.
- C0 e C1 não fazem parte deste protocolo.
