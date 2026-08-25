# Proveniência dos dados

Esta página documenta de onde vêm as anotações do Le2i usadas neste projeto,
sob qual licença, e como reproduzir o download com garantia de integridade.
Ela existe porque os CSVs de anotação **não são versionados** neste
repositório (ver `.gitignore` e a seção "Dados" do README) — a
reprodutibilidade vem da revisão fixada do dataset upstream e dos checksums
abaixo, não de um arquivo commitado. Ela também documenta como os vídeos
brutos do Le2i são casados com essas anotações e verificados, já que
`data/manifest.parquet` — o resultado desse casamento — também é gerado
localmente, nunca commitado.

## Fonte e revisão fixada

As anotações são obtidas do dataset agregado
[`simplexsigil2/omnifall`](https://huggingface.co/datasets/simplexsigil2/omnifall)
no HuggingFace Hub, que resolve a correspondência entre vídeos, splits e
rótulos de queda entre os datasets originais que compõem o OmniFall.

`scripts/fetch_labels.py` fixa a revisão do dataset em um SHA de commit
específico do repositório HuggingFace, em vez de seguir a `main` (`HEAD`)
móvel:

| Campo               | Valor                                      |
| ------------------- | ------------------------------------------ |
| `dataset_repo_id`   | `simplexsigil2/omnifall`                   |
| `OMNIFALL_REVISION` | `68e5cee56a4bad38cca4aea791cac248f96e79a0` |
| Configs usados      | `le2i-cs`, `labels`                        |

Os nomes desses configs já foram reestruturados uma vez no histórico do
dataset upstream. Fixar a revisão garante que uma futura reestruturação não
altere silenciosamente as colunas, os splits ou o conteúdo baixado por este
repositório.

## Contagens esperadas

Para a revisão fixada acima, o config `le2i-cs` (splits oficiais por sujeito)
e o config `labels` (filtrado para `dataset == "le2i"`) produzem:

| Arquivo     | Origem                        | Linhas |
| ----------- | ----------------------------- | -----: |
| `train.csv` | `le2i-cs`, split `train`      |    670 |
| `val.csv`   | `le2i-cs`, split `validation` |     94 |
| `test.csv`  | `le2i-cs`, split `test`       |    203 |
| `le2i.csv`  | `labels`, filtrado para Le2i  |    967 |

`670 + 94 + 203 = 967`, ou seja, `le2i.csv` cobre exatamente a união dos três
splits de `le2i-cs` — é usado para conferência cruzada, não como split
adicional.

## Reprodução e verificação de integridade

Baixar as anotações (fixadas em `OMNIFALL_REVISION`):

```bash
uv run python scripts/fetch_labels.py
```

O script grava, junto dos CSVs, um `data/labels/omnifall/PROVENANCE.json`
com `dataset_repo_id`, `revision`, `configs`, o timestamp UTC do fetch e,
para cada CSV, seu nome, hash SHA-256 e número de linhas.

Para conferir que os arquivos em disco continuam batendo com o que foi
registrado em `PROVENANCE.json` (sem baixar nada de novo):

```bash
uv run python scripts/fetch_labels.py --verify
```

O comando termina com código de saída diferente de zero e lista cada arquivo
ausente ou com hash divergente, caso encontre alguma inconsistência.

## Casamento dos vídeos com as anotações (`data/manifest.parquet`)

A árvore de vídeos extraída do Le2i não segue o mesmo layout dos paths
gravados nos CSVs de anotação do OmniFall, então associar um vídeo à sua
label exige normalizar os dois lados antes de compará-los. Exemplos reais:

| Path na label (OmniFall) | Arquivo local (Le2i)                                                |
| ------------------------- | -------------------------------------------------------------------- |
| `Coffee_room_01/video_1`  | `Coffee_room_01/Videos/video (1).avi`                                 |
| `Lecture_room/video_1`    | `Lecture room/video (1).avi` (sem subpasta `Videos/`, espaço em vez de `_` no nome do ambiente) |
| `Office/video_1`          | `Office/video (1).avi` (sem subpasta `Videos/`)                      |

`gatefall.data.ingest` normaliza os dois lados antes de comparar: remove
qualquer componente `Videos/` do path, remove a extensão `.avi`, converte
`video (N)` para `video_N` (e vice-versa), normaliza espaços e underscores no
nome do ambiente, e compara tudo em minúsculas.

Construir o manifesto:

```bash
uv run python -m gatefall.data.ingest ingest [--force]
```

O comando exige bijeção estrita entre vídeos locais e labels: exatamente 190
vídeos locais devem casar com exatamente 190 paths únicos de label. Qualquer
vídeo sem label correspondente, label sem vídeo, ou colisão de normalização
(dois arquivos distintos normalizando para a mesma chave) interrompe o
comando com código de saída diferente de zero e a lista completa de todas as
entradas não casadas de ambos os lados — nunca descarta uma linha
silenciosamente. `--force` é necessário para sobrescrever um manifesto já
existente; sem ele, `ingest` recusa e termina sem erro.

Para cada vídeo casado, o comando sonda o arquivo via `ffprobe` (frame rate,
contagem de frames do header e contagem autoritativa via `-count_frames`,
duração, resolução, codec) e calcula o sha256, gravando uma linha por vídeo
em `data/manifest.parquet`:

| Coluna                                      | Conteúdo                                                                                                     |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `video_id`                                   | Chave normalizada (`env/video_N`)                                                                              |
| `dataset`                                    | Sempre `"le2i"`                                                                                                 |
| `relative_path`, `absolute_path`             | Caminho do vídeo em `data/raw/le2i/`                                                                            |
| `env`, `subject`, `cam`, `split`             | Metadados vindos da label do OmniFall                                                                          |
| `fps`, `fps_source`                          | Frame rate e de qual campo do `ffprobe` ele foi lido                                                            |
| `n_frames_header`, `n_frames_counted`        | Contagem do header do container vs. contagem real via `-count_frames`                                          |
| `duration_s`, `width`, `height`, `codec`     | Metadados de vídeo do `ffprobe`                                                                                 |
| `sha256`                                     | Hash do arquivo, para detectar corrupção ou reextração parcial                                                 |
| `pose_status`, `dino_status`, `sam_status`   | `pending`/`done`/`failed`, status de extração de features por branch; todos `pending` nesta etapa — a extração ainda não foi implementada |

`data/manifest.parquet` é um artefato gerado (gitignored, nunca commitado),
assim como os CSVs de anotação — pelo motivo inverso: não é uma questão de
licença, é que ele pode sempre ser reconstruído a partir dos vídeos brutos e
das labels.

**O fps não é uniforme entre ambientes do Le2i** — `Home_01` e `Home_02`
rodam a ~23.9997 fps (`500000/20833`), todos os demais ambientes a
exatamente 25 fps (`25/1`) — por isso o fps de cada vídeo é sempre lido do
próprio arquivo (preferindo `avg_frame_rate`, caindo para `r_frame_rate` só
quando o primeiro não está disponível ou é zero), nunca assumido fixo.

Verificar o manifesto já construído:

```bash
uv run python -m gatefall.data.ingest verify
```

`verify` refaz a checagem de bijeção do zero, recalcula o sha256 de cada
vídeo referenciado no manifesto e compara com o valor gravado, e imprime um
relatório: disjunção de paths entre splits (falha o comando se violada),
disjunção de subjects entre train/val/test (informativo — o relatório declara
explicitamente se o `le2i-cs` é de fato cross-subject), distribuição completa
de resolução com todo vídeo fora da moda listado por nome (na prática a moda
é 320x240, com alguns vídeos de `Home_01`/`Home_02` em 320x180), distribuição
de fps por ambiente, um cruzamento `cam` × `env` (na prática confirma que
`cam` é função 1:1 de `env` — o Le2i tem uma câmera por ambiente, `cam`
identifica o local, não um ângulo de câmera), estatísticas de duração de
segmento por classe, contagem de segmentos por classe e por split
(sinalizando qualquer combinação com zero segmentos, principalmente em
`test`), e a duração total do dataset com a projeção do número de frames a
10/12.5/25 fps. Só a falha de bijeção, a falha de disjunção de splits e a
divergência de sha256 encerram o comando com erro — o restante do relatório é
informativo.

A escolha da taxa de amostragem e do tamanho da janela de treino depende
dessas estatísticas de duração e ainda não foi feita; conversão de rótulo por
frame e a própria janela não fazem parte deste comando.

## Licenças

Este repositório mistura três escopos de licença **distintos**, sobre
artefatos diferentes — é exatamente por isso que os CSVs de anotação não
podem ser versionados junto do código:

| Artefato                     | Licença                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| Código deste repositório     | MIT                                                                                       |
| Anotações do OmniFall (CSVs) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — uso não comercial |
| Vídeos originais do Le2i     | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) — uso não comercial |

CC BY-NC-SA é incompatível com a distribuição junto de código MIT em um
repositório público: exige atribuição, proíbe uso comercial e obriga
compartilhamento pela mesma licença para qualquer redistribuição — condições
que não fazem sentido aplicadas ao código do projeto. Por isso as anotações
são sempre baixadas sob demanda a partir da fonte original, nunca commitadas.
