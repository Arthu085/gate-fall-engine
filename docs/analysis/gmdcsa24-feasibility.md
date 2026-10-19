# Análise — Viabilidade do dataset GMDCSA24

Esta página avalia se o GMDCSA24 ("A Dataset for Human Fall Detection in
Videos", Alam et al., 2024) pode servir a um experimento controlado de aumento
de dados no GateFall. É um estudo de viabilidade, sem novo experimento. Nenhum
adapter foi registrado, nada foi treinado e nenhuma feature de backbone foi
extraída. Os contratos, checkpoints, features, protocolo de alarme e
referências de Le2i-CS e Le2i-CV não mudaram.

## Fontes e garantias

As propriedades abaixo têm uma de duas origens, marcadas em cada seção:

- **Repositório público.** Lido sem baixar os vídeos, em 2026-10-19: API do
  GitHub (tags, árvore Git do commit da tag `v2.1` com nome, tamanho e SHA-1 de
  blob de cada arquivo, comparação entre tags), os oito CSVs e o README pela
  URL `raw` do commit, a API do Zenodo e o snapshot do OmniFall já pinado no
  GateFall.
- **Download local.** Clone completo da tag `v2.1` (1,11 GB de MP4; 2,1 GB em
  disco com `.git`) fora de `data/`, auditado com
  `gatefall.data.gmdcsa24.audit`. Decodificação, FPS, duração, grade e janela
  causal só existem nessa origem.

Código reutilizado: `gatefall.data.video_metadata`,
`gatefall.data.resampling` (`build_time_grid`, `labels_for_grid`),
`gatefall.data.intervals`, `gatefall.data.omnifall` e `gatefall.config`.

## Fonte, licença e proveniência (repositório público)

| Campo | Valor |
| --- | --- |
| Repositório | `ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos` |
| Tag pinada | `v2.1`, commit `5abac7693229900cf80f722e878fbb119211fc1c` (2024-08-21) |
| DOI Zenodo da `v2.1` | `10.5281/zenodo.13354453` (zip de 1 107 545 615 bytes, MD5 `3d36f2c5c1a666b99639e4e9fd843efb`) |
| Licença no GitHub | MIT ("Software") |
| Licença no Zenodo | CC BY 4.0 |
| Anotações OmniFall | `simplexsigil2/omnifall@68e5cee`, config `labels` (`dataset == "GMDCSA24"`) e `gmdcsa24-{cs,cv}`; licença CC BY-NC 4.0 do card |
| Citação | Data in Brief, 2024; o README da `v2.1` ainda diz "communicated" |

- A `v2.1` só altera o README e os oito CSVs em relação à `v2.0`; os vídeos são
  os mesmos. Entre `1.1.1` e `v2.0`, arquivos dos sujeitos 2, 3 e 4 foram
  adicionados, removidos ou alterados. Por isso, a identidade de um vídeo só
  vale com a tag pinada.
- As duas licenças do autor permitem reuso com atribuição. A licença das
  anotações do OmniFall é não comercial, como a das anotações Le2i já usadas.
- Os vídeos mostram o rosto dos participantes e têm faixa de áudio AAC. O
  repositório não documenta consentimento de imagem. O pipeline só lê a faixa
  de vídeo.
- Os quatro sujeitos são `Subject 1` a `Subject 4`. Não há identificador de
  ambiente nem de câmera: o `cam` do OmniFall é sempre `1`.

O download é manual, fora do Git do GateFall:

```bash
git clone --depth 1 --branch v2.1 \
  https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos.git \
  data/raw/gmdcsa24/GMDCSA24
```

`data/raw/` já é ignorado pelo Git. `verify` confere o commit e a ausência de
alterações; o GitHub não publica checksum de arquivo único.

## Inventário e identidade (repositório público e download local)

```text
Subject N/
├── ADL.csv, Fall.csv   # File Name, Length (seconds), Time of Recording, Attire, Description, Classes
├── ADL/NN.mp4
└── Fall/NN.mp4
```

| Sujeito | ADL | Fall | Total |
| ---: | ---: | ---: | ---: |
| 1 | 16 | 16 | 32 |
| 2 | 23 | 25 | 48 |
| 3 | 22 | 21 | 43 |
| 4 | 20 | 17 | 37 |
| **Total** | **81** | **79** | **160** |

- **Vídeos utilizáveis: 160.** Os mesmos 160 identificadores aparecem no clone,
  nos CSVs originais, no config `labels` do OmniFall e nos splits `cs` e `cv`.
  `Subject 1/ADL/01.mp4` ↔ `Subject_1/ADL/01`. Não há linha repetida nem
  arquivo fora do layout.
- **Duplicatas: nenhuma.** Na árvore pública, os 160 SHA-1 de blob são
  distintos. No clone, nenhum par tem o mesmo SHA-256.
- O `subject` do OmniFall bate com o caminho em todos os 458 segmentos.

## Propriedades de vídeo (download local, 160 de 160)

| Propriedade | Observado |
| --- | --- |
| Codec | H.264 em todos, sem rotação, com faixa AAC |
| Resolução | 1280×720 em 142, 640×480 em 18, nos quatro sujeitos |
| FPS | Taxa variável: `r_frame_rate` 60/1 em 131 e 15000/1001 em 29; `avg_frame_rate` 14,95–29,93 (mediana 29,82) |
| Quadros | 85–523; `nb_frames` igual aos quadros decodificados em todos |
| Duração | 2,98–17,52 s (p5 4,32; mediana 7,31; p95 12,40) |
| Decodificação | Sem erro nos 160 |
| Desvio de PTS | Maior \|PTS − índice/fps\| por vídeo: mediana 0,014 s, p95 0,069 s, máximo 0,131 s; 4 vídeos acima de 0,1 s |

O desvio de PTS mede o erro de tratar o vídeo como taxa constante, que é o que
`build_time_grid` faz. No pior vídeo, um quadro da grade de 10 fps pode vir de
um instante até 1,3 passo de grade distante do rótulo.

`Length (seconds)` do CSV é inteiro e fica, em mediana, 0,49 s abaixo da
duração medida. `Subject_2/ADL/04` declara 70 s e dura 7,84 s.

## Janela causal de 24 quadros a 10 fps (download local)

| Medida | GMDCSA24 (160 vídeos) | Le2i-CS (referência) |
| --- | ---: | ---: |
| Vídeos com grade `K >= 24` | 160 de 160 (100%) | — |
| Quadros da grade | 12 799 | 22 246 no treino |
| Quadros com janela sem replicação de borda | 9 119 (71,2%) | — |
| Quadros `fall` | 1 543 | 1 993 no treino |
| Quadros `fall` com janela sem replicação de borda | 979 (63,4%) | 2 629 de 2 697 (97,5%, três splits) |
| Quedas cujo primeiro quadro `fall` tem índice >= 23 | 25 de 78 (32,1%) | 119 de 130 (91,5%, três splits) |

A coluna Le2i-CS vem de `data/processed/le2i/frames.parquet`, gerado pelo
pipeline canônico e só lido aqui.

Todo vídeo comporta ao menos uma janela causal completa (menor `K` = 29).
O problema está na posição da queda: no GMDCSA24, a mediana do início do
primeiro `fall` é 1,66 s. Em 53 das 78 quedas (68%), o primeiro quadro `fall`
da grade tem índice menor que 23, e as janelas do início dessas quedas contêm
cópias do quadro 0. No Le2i, isso ocorre em 8,5% das quedas. Se esses vídeos entrassem no treino sem tratamento, a replicação de
borda ficaria correlacionada com a classe `fall`, um atalho que não existe nos
splits de avaliação do Le2i.

## Anotações originais do autor (repositório público)

A coluna `Classes` é texto livre, multirrótulo e com onze nomes: `Falling`
(com direção `SW`, `FW` ou `BW`), `Fall` (variante de `Falling`), `Sitting`,
`Standing`, `Walking`, `Sleeping`, `Reading`, `Writing`, `Eating`, `Drinking`
e `Exercising`. O parser tolerante extrai 357 segmentos e relata 9 desvios de
formato:

| Vídeo | Desvio |
| --- | --- |
| `Subject_2/ADL/06` | `Sitting[9.7 to 3]`: fim antes do início |
| `Subject_2/Fall/11` | `Falling (BW[5.9 to 10]`: parêntese não fechado |
| `Subject_2/Fall/19`, `Subject_4/Fall/09` | Intervalo sem `to` (`0.9 4.6`, `2.3 6`) |
| `Subject_3/ADL/04` | Colchete não fechado em `Standing` |
| `Subject_3/ADL/12`, `Subject_4/Fall/15` | `Reading` e `Falling` sem intervalo |
| `Subject_3/Fall/01`, `Subject_3/Fall/02` | `Fall` em vez de `Falling` |

Sobre `[0, Length)`, 94 dos 160 vídeos têm trecho sem rótulo; 30 têm um gap de
1 s ou mais. Em 53 vídeos, uma atividade se sobrepõe a uma postura, por
desenho. Em 5, duas posturas se sobrepõem.

**As classes originais não equivalem às dez do OmniFall.** A matriz abaixo
mostra a fração do tempo de cada classe original coberta por cada classe
OmniFall nos mesmos vídeos:

| Original | Destino principal no OmniFall |
| --- | --- |
| `Falling` | `fallen` 57%, `fall` 41% |
| `Sleeping` | `lying` 53%, `lie_down` 39%, `stand_up` 9% |
| `Sitting` | `sitting` 57%, `other` 26%, `sit_down` 8% |
| `Standing` | `other` 54%, `standing` 33% |
| `Walking` | `walk` 73%, `other` 22% |
| `Reading`, `Eating`, `Writing` | `sitting` ou `other` |
| `Drinking`, `Exercising` | `other` 98% e 89% |

`Falling` vai do início da queda até o fim do vídeo e absorve `fallen`. Não há
classe para `sit_down`, `lie_down` ou `stand_up`. Nenhum mapeamento por nome
reproduz a taxonomia do OmniFall. O CSV original só serve para auditoria.

## Anotações do OmniFall (snapshot pinado)

São 458 segmentos densos e de rótulo único nas dez classes do GateFall:

| Classe | Segmentos | Segundos | S1 | S2 | S3 | S4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `walk` | 65 | 120,9 | 22,8 | 44,1 | 27,7 | 26,3 |
| `fall` | 78 | 154,4 | 34,0 | 52,1 | 43,3 | 25,1 |
| `fallen` | 77 | 209,8 | 39,1 | 69,6 | 54,1 | 47,0 |
| `sit_down` | 22 | 43,4 | 6,6 | 21,0 | 11,4 | 4,4 |
| `sitting` | 63 | 235,6 | 19,8 | 44,9 | 92,0 | 78,8 |
| `lie_down` | 13 | 41,6 | 8,4 | 16,4 | 3,7 | 13,1 |
| `lying` | 14 | 37,6 | 15,5 | 10,2 | 1,5 | 10,4 |
| `stand_up` | 17 | 39,9 | 5,0 | 13,0 | 17,4 | 4,6 |
| `standing` | 37 | 72,9 | 5,9 | 35,9 | 28,8 | 2,2 |
| `other` | 72 | 328,2 | 43,4 | 145,2 | 64,5 | 75,2 |

- **Limites temporais.** O último fim de cada vídeo fica entre 0,56 s antes e
  0,06 s depois da duração medida (mediana −0,003 s; p5 −0,019; p95 +0,027). Isso confirma que o
  OmniFall anotou os mesmos vídeos da `v2.x`, não só os mesmos nomes.
- **Gaps.** Na grade de 10 fps, 80 de 12 799 quadros ficam como
  `IGNORE_LABEL`, em 23 vídeos. Os maiores são o início de `Subject_1/ADL/05`
  (1,0 s) e um trecho de 1,26 s em `Subject_2/ADL/21`.
- **Sobreposições.** 24 vídeos têm alguma sobreposição. Em 22, ela é só de
  arredondamento (< 0,1 s). Há dois casos reais. Em `Subject_3/ADL/09`, `stand_up [0; 7,78)` cobre
  `sitting` e `standing`; pela regra de menor `start`, `stand_up` vence e
  apaga o resto. Em `Subject_3/Fall/09`, `fall` e `sitting` se sobrepõem por
  0,1 s. No total, `labels_for_grid` resolve 46 quadros em 3 vídeos.
- **Quedas.** Cada vídeo `Fall` tem no máximo um `fall`, e nenhum vídeo `ADL`
  tem `fall`. `Subject_2/Fall/03` (queda para a frente sobre a cama) foi
  anotado como `lie_down`/`lying`, sem `fall`. Em `Subject_4/Fall/15`, o CSV
  original não tem intervalo de queda, mas o OmniFall marca `fall` em 1,18 s.
- **Início da queda.** Nos 77 vídeos com as duas fontes, o `fall` do OmniFall
  começa em mediana 0,23 s antes do `Falling` original, com até 1,70 s de
  diferença. Em 22 vídeos, a diferença passa de 0,5 s. Isso é coerente com a
  definição do OmniFall: a queda começa no primeiro quadro em que o movimento
  diverge.

## Sujeitos, ambientes e splits

| Split OmniFall | Treino | Validação | Teste |
| --- | --- | --- | --- |
| `gmdcsa24-cs` | S1, S3 (75 vídeos) | S2 (48) | S4 (37) |
| `gmdcsa24-cv` | S2, S3 (91) | S1 (32) | S4 (37) |

- Os dois protocolos separam por sujeito. O "cv" não separa por câmera ou
  ambiente: só troca S1 e S2, e o teste é o mesmo S4 com 17 quedas.
- Não há rótulo de ambiente. Num quadro de amostra por sujeito, S2 e S3
  aparecem no mesmo quarto (mesmo tapete, cortina e porta), e S1 e S4 em
  quartos próprios. S1 tem ângulos de câmera diferentes entre ADL e Fall. Isso
  vem de inspeção visual de uma amostra, não de metadado.
- `Time of Recording` confunde iluminação com sujeito: só S2 e S3 têm `Night`
  (15 e 14 vídeos), só S4 tem `Lights Off` (18).
- O sujeito 2 tem 30% dos vídeos e 35% dos quadros. `sitting` se concentra em
  S3 e S4, e `standing` em S2 e S3.

## Compatibilidade com os contratos do GateFall

| Contrato | Situação |
| --- | --- |
| Taxonomia de dez classes | **Compatível** via OmniFall; o CSV original é incompatível |
| Segmentos `[start, end)` e `labels_for_grid` | **Compatível**; 80 quadros ignorados e 1 vídeo com sobreposição real |
| Identidade e proveniência | **Verificada**: 160 = 160 = 160, tag e revisão pinadas, durações coerentes |
| Grade de 10 fps | Funciona, mas com taxa variável: desvio de PTS até 0,13 s |
| Janela causal de 24 quadros | Cabe em 100% dos vídeos; só 32% das quedas têm histórico completo no início |
| Split por vídeo ou sujeito | Possível por sujeito, com quatro sujeitos e ambiente confundido |
| Monocular RGB | Sim; o áudio fica fora |
| Backbones congelados / HDF5 por vídeo | Sem impedimento |
| `DatasetAdapter` / manifesto | Viável: `subject` ← sujeito, `cam` = 1, `env` sem fonte (proposta: sala inferida, ver abaixo) |

## Decisão

**GO condicionado** para um PR de integração isolado. Os vídeos, a
identidade, os limites temporais e a taxonomia do OmniFall foram verificados
contra a tag `v2.1` e batem. O dataset adiciona 160 vídeos, 12 799 quadros e
1 543 quadros `fall`, o equivalente a 58% dos quadros e 77% dos quadros
`fall` do treino Le2i-CS.

O GO vale somente com estas condições:

1. **Rótulos somente do OmniFall** (`labels`, revisão `68e5cee`). O CSV
   original continua só como auditoria.
2. **Somente no treino.** GMDCSA24 nunca entra na validação ou no teste
   Le2i-CS/Le2i-CV, nem no cálculo de estatísticas de padronização usadas
   nesses testes, nem na calibração do alarme. As métricas de referência
   continuam sendo as do Le2i.
3. **A, B e C juntos.** O aumento entra igual nas três configurações, com a
   mesma seleção de vídeos e janelas, e as três são executadas de novo.
4. **Replicação de borda tratada antes do treino.** O PR precisa medir e
   tratar a correlação entre a replicação de borda e `fall`, por exemplo
   descartando janelas de treino do GMDCSA24 com `k_end < 23` ou registrando
   a fração delas por classe. A escolha vale para as três configurações.
5. **Exclusões explícitas.** `Subject_3/ADL/09` fica de fora ou recebe
   correção documentada da sobreposição. `Subject_2/Fall/03` fica como o
   OmniFall anotou, com nota.
6. **Escopo declarado.** Pela regra de risco dominante, o PR precisa dizer o
   que o experimento desloca antes de começar.

## Limitações

- Quatro sujeitos e no máximo três ambientes. Um teste com GMDCSA24 isolado
  teria 17 quedas e não sustenta conclusão estatística.
- O ambiente vem de inspeção visual de oito quadros, não de metadado.
- Taxa de quadros variável: a grade assume taxa constante. Quatro vídeos
  passam de um passo de grade de desvio.
- Resolução e FPS diferem do Le2i (320×240 ou 320×180, 24–25 fps). O efeito disso nas
  features congeladas não foi medido.
- A licença do Zenodo (CC BY 4.0) e a do GitHub (MIT) divergem. O
  consentimento de imagem não está documentado.
- O artigo do Data in Brief não foi lido; esta análise não usa nada dele.

## Estratégia de validação proposta para o PR de integração

1. Adapter e manifesto isolados (`data/processed/gmdcsa24/`), com `sha256` por
   vídeo e `verify` desta auditoria como pré-requisito.
2. Selftests sintéticos no padrão do Le2i para o adapter e a montagem do treino
   aumentado. Os selftests e as referências existentes do Le2i precisam
   continuar idênticos.
3. Verificação automática de que nenhum `video_id` do GMDCSA24 aparece em
   validação, teste ou estatísticas de padronização do Le2i.
4. Relatório de janelas por classe e da fração com replicação de borda, antes e
   depois do tratamento da condição 4.
5. Comparação pareada com as referências atuais, com o bootstrap agrupado por
   sujeito já existente e várias seeds, nas três configurações.

## Como reproduzir a validação

Com o clone feito conforme acima:

```bash
uv run python -m gatefall.data.gmdcsa24.audit verify
uv run python -m gatefall.data.gmdcsa24.audit inventory
uv run python -m gatefall.data.gmdcsa24.audit annotations
uv run python -m gatefall.data.gmdcsa24.audit probe
uv run python -m gatefall.data.gmdcsa24.audit selftest
```

Todos aceitam `--root DIR`. O padrão é `data/raw/gmdcsa24/GMDCSA24`.

- `verify` confere se `HEAD` é o commit da tag `v2.1` e se não há arquivo
  modificado ou não rastreado. Falha caso contrário.
- `inventory` relata vídeos por sujeito e categoria, CSVs, arquivos fora do
  layout e duplicatas por SHA-256. Compara a identidade dos vídeos entre clone,
  CSV original, `labels` e splits `cs`/`cv` do OmniFall. Falha em divergência,
  duplicata ou arquivo fora do layout.
- `annotations` relata colunas, classes, desvios de formato, gaps e
  sobreposições do CSV original. Também relata classes por sujeito, cobertura e
  splits do OmniFall, a matriz original × OmniFall e o início da queda nas duas
  fontes.
- `probe` decodifica cada MP4 no lugar com o mesmo `probe_video` do Le2i e mede
  codec, resolução, FPS, desvio de PTS, duração contra `Length`, `K` da grade
  de 10 fps e vídeos com `K < WINDOW_FRAMES`. Aplica os segmentos OmniFall com
  `labels_for_grid` e relata quadros ignorados, sobreposições resolvidas e
  quadros e inícios de `fall` sem replicação de borda. Falha se algum vídeo
  não decodificar ou se nenhum for sondado.
- `selftest` monta um clone sintético com os desvios observados e não precisa
  do dataset nem de rede. Requer FFmpeg e Git.

`inventory`, `annotations` e `probe` leem o OmniFall pelo
`load_annotation_config` existente, na revisão pinada, pelo cache do Hugging
Face. Nenhum comando grava em `data/`. O módulo não está em
`SUPPORTED_DATASET_IDENTIFIERS` nem no `gatefall.pipeline`.
