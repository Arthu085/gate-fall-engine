# Análise — Viabilidade do dataset de vídeo do Figshare 28596332

Esta página avalia se o dataset público "Video-Based Fall Detection Dataset
with 2017 Activities from 29 Subjects" (Figshare 28596332) pode entrar no
GateFall. É um estudo de viabilidade, sem novo experimento. Nenhum adapter foi
registrado e nada foi treinado ou extraído. Não houve download completo. Os
contratos, checkpoints, features e referências de Le2i-CS e Le2i-CV não
mudaram.

## Fontes e garantias

- **Registro.** API pública do Figshare (`/v2/articles/28596332` e suas
  versões 1 e 2), consultada em 2026-10-19.
- **Estrutura do arquivo.** O diretório central do `VideoDataset.zip` foi lido
  por requisições HTTP `Range`: cerca de 2,9 MB, num arquivo de 2,53 GB. Ele
  dá nome, tamanho e CRC-32 de todos os 22 397 membros. As contagens abaixo
  marcadas como **verificadas** vêm dele.
- **Amostra.** Também por `Range`, foram lidos seis `metadata.json`, três logs
  de sensor, nove MP4 (um por par categoria × local) e o par de MP4 suspeito de
  duplicata. O CRC-32 de cada MP4
  bateu com o do diretório central. Os arquivos ficaram fora do repositório e
  de `data/`. Propriedades medidas só nessa amostra estão marcadas como
  **amostra**.
- **Código.** `gatefall.datasets.base.DatasetAdapter`,
  `gatefall.data.manifest`, `gatefall.data.ingest`,
  `gatefall.data.video_metadata`, `gatefall.data.resampling`,
  `gatefall.data.windowing` e `gatefall.config`.

## Fonte, licença e aquisição

| Campo | Valor |
| --- | --- |
| Título (v2) | Video-Based Fall Detection Dataset with 2017 Activities from 29 Subjects |
| Título (v1) | Sensor-Based Fall Detection Dataset with 2017 Activities from 29 Subjects |
| Autor | Ivan Ursul (ORCID 0009-0002-9879-8008) |
| DOI | `10.6084/m9.figshare.28596332.v2` (v1 com o mesmo arquivo) |
| Publicação | 2025-03-14 |
| Licença | CC BY 4.0: reuso permitido com atribuição |
| Arquivo | `VideoDataset.zip`, id 52990358, 2 529 520 868 bytes |
| MD5 publicado | `c784167d08f2fa94e3afd36cec758e1f` |
| Download | `https://ndownloader.figshare.com/files/52990358` |

A versão 2 só mudou o título, de "Sensor-Based" para "Video-Based". A URL
pública ainda usa o slug antigo. O registro não tem artigo associado, livro de
códigos das ações, protocolo de split, README nem declaração de consentimento
ou ética. Os vídeos mostram o rosto dos participantes. A licença permite o
reuso. O consentimento para publicação de imagem não está documentado no
registro.

O download é manual. O Figshare publica apenas MD5. Por isso, `verify` também
imprime o SHA-256 local, para registro como evidência:

```bash
mkdir -p data/raw/figshare_fall
curl -L -o data/raw/figshare_fall/VideoDataset.zip \
  https://ndownloader.figshare.com/files/52990358
```

`data/raw/` e `*.zip` já são ignorados pelo Git. O arquivo não precisa ser
extraído: a auditoria lê os membros direto do zip.

## Estrutura do arquivo (verificada)

```text
VideoDataset/
├── timelapse.mp4
├── ADL/
│   ├── SBJ_<sujeito>_LOC<local>/ACT<ação>_<lado>_<tentativa>/
│   │   ├── <timestamp>.mp4
│   │   ├── <timestamp>.txt      # log do sensor vestível (IMU)
│   │   └── metadata.json        # subjectId, locationId, actionId, side, attempt
│   ├── all_csvs/<timestamp>.csv  # sinais de aceleração filtrados
│   └── all_plots/<timestamp>.png
└── Fall/
    ├── (mesma árvore de ADL)
    ├── timelapse_20x.mp4
    ├── timelapse_20x_v2.mp4
    └── timelapse_20x_v2.mp4.zip
```

O zip também tem 10 233 membros `__MACOSX/` e `.DS_Store`, gerados pelo
macOS. Todos os membros usam deflate, e o átomo `moov` dos MP4 fica no fim do
arquivo. Por isso, codec, FPS e duração de todos os vídeos só podem ser medidos
com o zip local.

Cada `.txt` registra `time`, `DeviceName`, aceleração, velocidade angular,
ângulo, temperatura, versão e bateria de um sensor `HC-06`, com timestamp
absoluto em milissegundos. Os `metadata.json` lidos repetem os campos do
caminho.

## A discrepância 2017 / 999 / 1017 (verificada)

A descrição anuncia 2 017 gravações, com 999 quedas e 1 017 ADL. A soma dá
2 016. Os três números são contagens brutas de arquivos `.mp4`, não de
gravações:

| Pasta | `.mp4` | Composição |
| --- | ---: | --- |
| `ADL/` | 1 017 | 1 016 gravações com vídeo, uma delas com dois vídeos |
| `Fall/` | 999 | 992 gravações com vídeo, cinco delas com dois vídeos, e dois timelapses |
| raiz | 1 | `timelapse.mp4` |
| **Total** | **2 017** | |

A unidade utilizável é a gravação, ou seja, um diretório `ACT…`:

| Medida | ADL | Fall | Total |
| --- | ---: | ---: | ---: |
| Diretórios de gravação | 1 016 | 1 008 | 2 024 |
| Com ao menos um vídeo | 1 016 | 992 | 2 008 |
| Sem vídeo | 0 | 16 | 16 |
| Com dois vídeos | 1 | 5 | 6 |

- Pelo diretório central, o único par de MP4 com mesmo CRC-32 e mesmo tamanho
  é `Fall/SBJ_02_LOC2/ACT1_L_1/20240912_101331.mp4` e
  `20240912_101331 (1).mp4`, na mesma gravação. Isso indica uma provável cópia,
  mas não prova igualdade, porque o CRC-32 colide. Os dois membros foram então
  baixados por `Range`, cerca de 0,7 MB cada, e têm SHA-256 idêntico
  (`dd255576…d93a41`). Com isso, a cópia está verificada. As outras cinco
  gravações com dois vídeos têm CRC-32 diferentes, com timestamps separados
  por 4 a 47 s. O registro não diz qual tomada é a válida.
- Oito das nove gravações da ação 12 (Fall) não têm vídeo.
- Os três timelapses são compilações aceleradas e não são gravações.
- Restam 2 013 vídeos distintos dentro de diretórios de gravação. A contagem
  é exata: CRC-32 ou tamanho diferentes já provam conteúdo diferente, e o
  único par coincidente foi confirmado por SHA-256. O número de quedas
  utilizáveis depende de como as gravações duplas forem tratadas: são 987 com
  exatamente um vídeo, ou 992 com ao menos um.

## Sujeitos, locais e granularidade do rótulo (verificada)

Há 29 sujeitos distintos (`SBJ_01` a `SBJ_29`), como anunciado. Os IDs são
compartilhados entre `ADL/` e `Fall/`. Os seis `metadata.json` lidos confirmam
o mesmo `subjectId`. O registro não garante que o mesmo ID nas duas pastas seja
a mesma pessoa.

| Grupo de sujeitos | Sujeitos |
| --- | --- |
| Fall e ADL (13) | 1–10, 13, 14, 20 |
| Só Fall (8) | 11, 12, 15–18, 21, 22 |
| Só ADL (8) | 19, 23–29 |

| Local | ADL | Fall |
| ---: | ---: | ---: |
| 1 | 509 | 348 |
| 2 | 151 | 356 |
| 3 | 263 | 254 |
| 4 | 41 | 0 |
| 5 | 52 | 50 |

Quatro sujeitos (1, 2, 3 e 6) mudam de local entre Fall e ADL. As ações de
Fall (1–12) e de ADL (13–25) são disjuntas. O registro não publica o
significado de cada código de ação. Lado (`L`, `R`, `F`, `B`) e tentativa (1 a
8) completam a chave.

**O rótulo é por gravação.** Não há início ou fim de queda, rótulo por quadro
nem classes além de Fall e ADL. Os nove MP4 amostrados duram cerca de 1,9 s. Na
amostra de Fall, a pessoa caminha, cai e termina deitada no colchão dentro do
mesmo clipe. Projetar o rótulo da gravação sobre todos os quadros confundiria
`walk`, `fall` e `fallen` da taxonomia de dez classes do OmniFall. Não existe
mapeamento derivável para essa taxonomia.

O log IMU tem timestamp absoluto, mas o nome do MP4 só tem resolução de 1 s. Na
amostra, os logs cobrem 4,4 a 9,1 s, mais que o vídeo. O deslocamento
sub-segundo entre IMU e vídeo é desconhecido. Derivar eventos do IMU seria um
protocolo de anotação novo, e o IMU não pode entrar em nenhum ramo do pipeline
monocular RGB.

## Propriedades de vídeo (amostra de 9)

| Propriedade | Observado |
| --- | --- |
| Codec / pixel | H.264, `yuv420p`, sem tag de rotação |
| Resolução | 1920×1080 em todos |
| FPS | 30/1 constante em 8; um vídeo (`ADL/SBJ_07_LOC3/ACT15_F_1`) com `r_frame_rate=120/1` e `avg_frame_rate=6840/223` (≈30,67), PTS irregulares de 25–33 ms |
| Quadros | 56–57, iguais a `nb_frames` do cabeçalho |
| Duração | 1,858–1,900 s |
| Decodificação | `ffprobe -count_frames` sem erro; PTS monotônicos |

Com `TARGET_FPS = 10`, a grade de `build_time_grid` tem `K = 18` ou `19`
quadros. Isso fica abaixo de `WINDOW_FRAMES = 24`. Nenhuma janela de avaliação
cabe sem replicação de borda: toda janela tem ao menos 24 − 19 = 5 posições
preenchidas com o quadro 0. Em resolução, os vídeos têm 6 vezes a largura e
4,5 vezes a altura dos 320×240 do Le2i.

## Compatibilidade com os contratos do GateFall

| Contrato | Situação |
| --- | --- |
| `DatasetAdapter` / manifesto | Viável com decisões: `env` ← local, `subject` ← `subjectId`; `cam` não é publicado; `split` não existe |
| Metadados do manifesto (`fps`, `n_frames_*`, `duration_s`, resolução, codec, `sha256`) | Mensuráveis após o download, como faz `probe` |
| Taxonomia de dez classes do OmniFall | **Incompatível**: só Fall/ADL por gravação |
| Grade temporal e `labels_for_grid` | **Incompatível**: não há segmentos `[start, end)` abaixo do vídeo inteiro |
| Janela de 24 quadros a 10 fps | **Incompatível** na amostra: K = 18–19. Reduzir a janela só para este dataset viola a invariante 1 |
| Split por sujeito | Possível (29 IDs), mas confundido com classe e local (ver abaixo) |
| Monocular RGB | Os MP4 são RGB. IMU, `all_csvs` e `all_plots` precisam ficar fora |
| Backbones congelados / HDF5 por vídeo | Sem impedimento |

## Vazamento e viabilidade de split

- **Cópias e compilações.** A cópia `(1).mp4`, confirmada por SHA-256, e os
  três timelapses precisam ser excluídos. Os timelapses provavelmente reúnem quadros de várias gravações e
  vazariam entre splits. Isso não foi verificado.
- **Sujeito confundido com classe.** 16 dos 29 sujeitos têm uma só classe. Um
  split por sujeito deixaria aparência e identidade predizerem o rótulo.
- **Local confundido com classe.** O local 4 só tem ADL. O local 1 tem 59% ADL
  e o local 2 tem 70% Fall. Na amostra, o cenário é um ginásio com colchões.
  Um split por sujeito não desacopla o local.
- **Tentativas repetidas.** Tentativas 2 a 8 do mesmo sujeito, ação e lado
  ficam juntas num split por sujeito. Um split por gravação ou janela vazaria.
- **Granularidade.** Com rótulo só por gravação, a única avaliação honesta
  seria por clipe, por exemplo alarme sim/não por gravação. Isso cria um
  protocolo de avaliação novo.

## Conclusão

O dataset é bem identificado, licenciado (CC BY 4.0) e verificável, e a
estrutura do arquivo está íntegra. Ele **não é
compatível** com os contratos atuais de treino e avaliação do GateFall: os
rótulos são binários e por gravação, e os clipes de cerca de 1,9 s não cabem na
janela de 24 quadros a 10 fps. A forma menos invasiva de uso seria um teste
externo binário por clipe, com split por sujeito e as exclusões acima. Isso
seria um novo protocolo de avaliação, com seu próprio tratamento de padding.
Pela regra de escopo do projeto, essa adição só procede se alguém nomear
explicitamente o que ela desloca.

## Propriedades não verificadas

- MD5 e SHA-256 do arquivo completo baixado.
- Codec, resolução, FPS, duração e decodificabilidade dos 2 005 MP4 de
  gravação fora da amostra e dos timelapses.
- Coerência de todos os 2 024 `metadata.json` com o caminho.
- Conteúdo dos timelapses e se reúnem quadros das gravações.
- Qual tomada vale nas cinco gravações com dois vídeos distintos.
- Se o mesmo `SBJ_xx` em `ADL/` e `Fall/` é a mesma pessoa.
- Significado dos códigos de ação 1–25, número de câmeras e consentimento de
  imagem.

## Como reproduzir a validação

Com o zip baixado conforme acima:

```bash
uv run python -m gatefall.data.figshare_fall.audit verify
uv run python -m gatefall.data.figshare_fall.audit inventory
uv run python -m gatefall.data.figshare_fall.audit probe [--limit N]
uv run python -m gatefall.data.figshare_fall.audit selftest
```

Todos aceitam `--archive PATH`. O padrão é
`data/raw/figshare_fall/VideoDataset.zip`.

- `verify` confere tamanho e MD5 com o registro e imprime o SHA-256 local.
  Falha em divergência.
- `inventory` lê o diretório central e os `metadata.json`. Ele relata
  contagens anunciadas × arquivo, gravações sem vídeo ou com vídeos duplos,
  ações, sujeitos e locais. Também lista as prováveis duplicatas, ou seja,
  vídeos com mesmo CRC-32 e tamanho. Para cada grupo, compara o SHA-256 do
  conteúdo descompactado e informa se o conteúdo é idêntico ou diferente.
  Falha se algum `metadata.json` faltar ou divergir do caminho.
- `probe` extrai cada MP4 para um diretório temporário e roda o mesmo
  `probe_video` do Le2i. Relata codec, resolução, FPS, quadros, duração, `K` da
  grade a 10 fps e vídeos com `K < WINDOW_FRAMES`. Sem `--limit`, sonda todos
  os vídeos de gravação. Com `--limit N`, sonda uma amostra determinística
  espalhada, e `N` precisa ser positivo. Falha se algum vídeo não decodificar
  ou se nenhum vídeo for sondado.
- `selftest` monta um zip sintético com os desvios observados e não precisa do
  dataset. Requer FFmpeg.

Nenhum comando grava em `data/`. O módulo não está em
`SUPPORTED_DATASET_IDENTIFIERS` nem no `gatefall.pipeline`.
