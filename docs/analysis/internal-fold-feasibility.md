# Análise — Viabilidade das dobras internas do Le2i-CV (Etapa 2B)

`src/gatefall/eval/analysis/internal_fold_feasibility.py` avalia se o
[candidato 1 da Etapa 2A](intervention-decision.md#candidato-1-validacao-interna-por-ambiente-em-le2i-cv-instrumento)
é viável com os dados existentes. Cada uma das três dobras propostas retém um
ambiente do `train` oficial do Le2i-CV e usa os outros dois como treino de
desenvolvimento.

A avaliação é **exploratória, não canônica e somente leitura**. Ela não
implementa o protocolo de dobras, não treina, não cria split nem estatística
de padronização, não seleciona hiperparâmetro e não altera splits oficiais,
features, checkpoints, referências nem resultados existentes.

## Entradas e contratos preservados

- Só as linhas `split == train` de `data/processed/le2i_cv/frames.parquet` e
  `manifest.parquet` são carregadas, com filtro na leitura do Parquet. Linhas
  de `val` e `test` oficiais nunca entram em memória. O SHA-256 dos dois
  arquivos inteiros é calculado antes e conferido depois, como nas demais
  análises; isso lê os bytes do arquivo, mas não decodifica nenhuma linha
  fora de `train`.
- A partição precisa bater com o Le2i-CV pinado: ambientes
  `Coffee_room_01`, `Coffee_room_02` e `Lecture_room`, 97 vídeos, 17905
  janelas em stride 1 e 16595 rotuladas, os mesmos valores verificados pelo
  relatório de janelas. O manifesto filtrado não pode repetir `video_id`, os
  vídeos de frames e manifesto precisam coincidir, cada vídeo precisa ter
  ambiente e sujeito únicos nos frames, e esse ambiente e esse sujeito
  precisam ser iguais aos do manifesto. Qualquer divergência interrompe a
  execução sem gravar nada.
- As janelas vêm de `build_window_index` com `WINDOW_FRAMES`,
  `TRAIN_STRIDE` (4) e `EVAL_STRIDE` (1). Uma janela é ignorada quando o
  último quadro é `IGNORE_LABEL`, como no treino e na avaliação.
- As classes avaliadas são as oito de `RESTRICTED_CLASSES`. `lie_down` e
  `lying` aparecem só em `excluded_class_support`.
- Identidades são o par `(ambiente, sujeito)`. IDs brutos de sujeito não são
  globais no Le2i.

## O que é medido

Para o `train` oficial (referência) e para cada lado de cada dobra:
ambientes, câmeras, resolução e fps, vídeos, identidades, janelas totais,
rotuladas, ignoradas e rotuladas fora de `other` nos dois strides, e o
suporte de cada classe restrita em janelas, vídeos e identidades.

Cada lado tem um **stride de suporte**, o stride em que ele é usado: o treino
de desenvolvimento e o `train` oficial usam `TRAIN_STRIDE` (4), porque o
treino só vê essas janelas, e o ambiente retido usa `EVAL_STRIDE` (1), como
`val`. A presença de cada classe, os vídeos e as identidades que a contêm,
a fração de janelas e as listas de classes ausentes ou de suporte único são
medidos no stride de suporte do lado. Uma classe que só aparece em quadros
fora dos fins de janela de stride 4 conta como ausente no treino de
desenvolvimento, mesmo que exista em stride 1. As contagens de janelas nos
dois strides continuam no JSON. O critério da Etapa 2A usa a fração predita
como `other` entre as janelas fora de `other` do ambiente retido.

As limitações por dobra são fatos estruturais, sem limiar novo:

- classe restrita sem nenhuma janela, presente em um único vídeo ou em uma
  única identidade;
- parcela das janelas rotuladas do `train` oficial usada pelo treino de
  desenvolvimento;
- razão entre a maior e a menor classe presente, comparada com a do `train`
  oficial;
- fração de janelas `other` em cada lado;
- IDs brutos de sujeito presentes nos dois lados;
- câmera única no ambiente retido.

`criterion_blockers` só registra impossibilidades lógicas do critério: o
ambiente retido sem janela fora de `other`, que deixa a fração sem
denominador, ou o treino de desenvolvimento sem `other`, caso em que a
fração predita deixa de medir o deslocamento.

## Saídas

`--output-dir` recebe dois arquivos, gravados atomicamente. Sem `--force`,
nada é gravado se algum deles já existir.

- `internal_fold_feasibility.json`: `canonical: false`, `read_only: true`,
  `status`, `official_splits_read`, contrato de janelas, classes restritas,
  `support_strides`, `official_train`, `folds` (com `development_train`,
  `retained`, `criterion_blockers` e `limitations`), `missing_prerequisites`,
  `limitations`, `approval_required` e `input_sha256`. Cada lado traz
  `support_stride` e, por classe, `windows_train_stride` e
  `windows_eval_stride` (contagens nos dois strides) mais
  `windows_at_support_stride`, `share_at_support_stride`,
  `videos_at_support_stride` e `identities_at_support_stride`. As listas
  `missing_restricted_classes_at_support_stride`,
  `single_video_classes_at_support_stride` e
  `single_identity_classes_at_support_stride` usam o mesmo stride.
- `internal_fold_feasibility.md`: tabelas por dobra, bloqueios, limitações e
  as restrições que exigem aprovação.

Se `frames.parquet` ou `manifest.parquet` não existir, o relatório sai com
`status: missing_prerequisites`, lista os arquivos ausentes e não contém
nenhum número. `--output-dir` não pode coincidir com `runs/reference/`,
`src/`, `data/` nem com os diretórios das entradas, nem estar dentro ou
acima deles.

## Como executar

```bash
uv run python -m gatefall.eval.analysis.internal_fold_feasibility selftest
uv run python -m gatefall.eval.analysis.internal_fold_feasibility assess \
  --output-dir runs/local/internal_fold_feasibility \
  [--frames data/processed/le2i_cv/frames.parquet] \
  [--manifest data/processed/le2i_cv/manifest.parquet] [--force]
```

O `selftest` usa apenas dados sintéticos e cobre a construção das dobras,
as contagens nos dois strides com `IGNORE_LABEL`, o suporte por classe e por
identidade no stride de cada lado, inclusive uma classe presente em stride 1
e ausente em stride 4 no treino de desenvolvimento, as classes ausentes e de
suporte único, os bloqueios do critério, a leitura restrita a `train`, as
guardas da partição oficial e dos metadados do manifesto, os
pré-requisitos ausentes e a proteção da saída.

## Resultado local (não canônico)

Executado em 2026-10-14 no branch `feat/le2i-cv-internal-fold-feasibility`
(base `analysis/gatefall-improvements`, `311c6af`), com suporte do treino de
desenvolvimento medido em stride 4 e guardas de metadados do manifesto.
As guardas passaram sem divergência. SHA-256 das entradas:
`frames.parquet` `b65bedc3…b640` e `manifest.parquet` `4dcad0c5…a94a`. O
relatório não é versionado; o comando acima o reproduz.

| Dobra (ambiente retido) | Treino dev.: vídeos / identidades / janelas rotuladas (stride 4) | Retido: vídeos / identidades | Retido: rotuladas / ignoradas (stride 1) | Retido: fora de `other` (stride 1) |
| --- | --- | --- | --- | --- |
| `Coffee_room_01` | 49 / 13 / 2726 (65%) | 48 / 8 | 5731 / 19 | 5339 |
| `Coffee_room_02` | 75 / 13 / 2867 (69%) | 22 / 8 | 5176 / 71 | 3167 |
| `Lecture_room` | 70 / 16 / 2743 (66%) | 27 / 5 | 5688 / 1220 | 4431 |

O `train` oficial tem 4168 janelas rotuladas em stride 4, 97 vídeos e 21
identidades.

Fatos confirmados:

- Nenhuma classe restrita falta em nenhum lado de nenhuma dobra, no stride
  de suporte de cada lado, e nenhuma dobra tem bloqueio do critério.
  `lie_down` e `lying` têm zero janelas em todos os lados, como no `train`
  oficial.
- No treino de desenvolvimento, em stride 4, toda classe restrita aparece em
  pelo menos 5 vídeos. O menor suporte é `sit_down` na dobra `Lecture_room`:
  44 janelas de 5 vídeos e 3 identidades. Nenhuma classe depende de um
  único vídeo ou de uma única identidade nesse lado.
- Suporte pequeno no lado retido: `sit_down` em `Coffee_room_01` vem de um
  único vídeo (24 janelas). Em `Coffee_room_02`, `standing` vem de 3 vídeos
  e `sit_down` de 4 vídeos de 2 identidades. `Lecture_room` tem só 5
  identidades, e `standing` vem de 2 delas.
- `Lecture_room` tem 1220 das 6908 janelas em stride 1 ignoradas (18%), por
  causa do prefixo de sala vazia.
- A fração de janelas `other` muda entre os lados: 30% no treino de
  desenvolvimento e 7% no retido na dobra `Coffee_room_01`, 14% e 39% na
  dobra `Coffee_room_02`, e 22% e 22% na dobra `Lecture_room`. No `train`
  oficial, a fração é 22%.
- A razão `walk`/`sit_down` em stride 4 no treino de desenvolvimento é 16,3,
  36,8 e 21,0, contra 22,6 no `train` oficial.
- Os IDs brutos 0, 1, 2, 3, 4, 6 e 7 aparecem nos dois lados das dobras de
  `Coffee_room_01` e `Coffee_room_02`, e 0, 1, 3, 6 e 7 na de
  `Lecture_room`. As dobras são disjuntas por ambiente, mas não há como
  provar que sejam disjuntas por sujeito.
- Cada ambiente tem uma única câmera (1, 2 e 5), e todo o `train` oficial é
  320×240 a 25 fps.

## Recomendação

**As três dobras são estruturalmente viáveis como instrumento para o
critério da Etapa 2A, mas não como medida de desempenho por classe no
ambiente retido.** Todo ambiente retido tem milhares de janelas fora de
`other`, distribuídas em 22 a 48 vídeos, e todo treino de desenvolvimento
contém as oito classes restritas. A fração predita como `other` é, portanto,
calculável nas três dobras.

Por outro lado, a F1 de `sit_down` e `standing` no ambiente retido depende
de um a três vídeos em algumas dobras e não deve ser lida como resultado.
A proporção de `other` também varia de 7% a 39% entre ambientes retidos, por
isso o critério só deve comparar armas dentro da mesma dobra. Essa é a forma
já definida na Etapa 2A.

O piloto de 9 treinos, com uma seed, não aplica o critério como ele foi
escrito. O critério compara a menor fração entre seeds de B0 ou B1 com a
maior entre seeds de A, e com uma seed isso vira uma comparação única, sem
variação entre seeds. O piloto serviria só para validar o pipeline das
dobras. Para decidir se o deslocamento se reproduz, são necessários os 45
treinos ou outro critério aprovado antes de treinar.

As dobras não testam a hipótese de resolução ou fps, porque todo o `train`
oficial é homogêneo nesses fatores. Um resultado negativo continua com a
leitura limitada descrita na Etapa 2A.

## Exige aprovação explícita antes de implementar ou treinar

- Um modo de split por dobra interna no pipeline de treino. É um protocolo
  de desenvolvimento novo e desloca a próxima etapa exploratória de treino,
  não se soma a ela.
- Estatísticas de padronização de pose, DINOv3 e SAM 3 ajustadas só nos dois
  ambientes de treino de cada dobra, em caminhos isolados, sem sobrescrever
  os arquivos `*_le2i_cv.json` oficiais.
- Orçamento de runs (9 ou 45), armas (A, B0 e B1) e receita congelada, com
  checkpoint da última época e sem nova extração.
- Critério de reprodução fixado antes de treinar e não ajustado depois de
  ver as dobras.
- `RESTRICTED_CLASSES` fixo em todas as dobras, com as classes de suporte
  único relatadas e não compensadas.
- Runs das dobras em diretório local próprio, nunca em `runs/reference/` nem
  nos runs locais do Le2i-CV oficial, e nenhum uso de `val` ou `test`
  oficiais para seleção.

## Limitações

- As contagens por janela usam janelas sobrepostas e não são observações
  independentes. Vídeos e identidades são as unidades mais próximas de
  independentes.
- Com três ambientes, há só três dobras, e nenhuma variação entre ambientes
  pode ser estimada dentro de uma dobra.
- Cada dobra treina com cerca de dois terços das janelas e outra composição
  de classes, sujeitos e cenas. As magnitudes não são comparáveis às do
  Le2i-CV oficial.
- Qualidade de pose, de features e de rótulos não é avaliada.
