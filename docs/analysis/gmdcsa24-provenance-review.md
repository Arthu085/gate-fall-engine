# Análise — Revisão visual da proveniência do GMDCSA24

Esta página registra o fluxo e o estado da checagem visual exigida pelo
pré-requisito 2 da [viabilidade do GMDCSA24](gmdcsa24-feasibility.md#pre-requisitos-sem-treino)
e pela condição V1 do [protocolo proposto](gmdcsa24-pilot-protocol.md). Ela
compara os limites das anotações OmniFall pinadas com os MP4 da tag `v2.1`,
quadro a quadro e pelo PTS real.

**Estado em 2026-10-19: PENDENTE.** Os 52 casos estão `pending`. Nenhuma
anotação foi confirmada por revisor humano.

## Evidência automática e confirmação humana

A planilha `review.csv` separa duas coisas:

- **Evidência automática** (`review-plan` e `review-render`): fronteira anotada,
  rótulos esperados antes e depois, `source_sha256` do MP4, PTS real dos
  quadros, quadro que a grade de taxa constante associaria à fronteira e uma
  contact sheet por caso. Ela mostra o vídeo. Não confirma a anotação.
- **Confirmação humana** (colunas `review_outcome`, `observed_boundary_s`,
  `reviewer`, `reviewed_at`, `notes`): o julgamento de quem olhou os quadros.
  Só essas colunas mudam o estado de um caso.

`review-status` não confia na planilha para saber o que revisar. Ele exige
`verify` OK, decodifica os 160 vídeos e reconstrói o conjunto de casos
exigidos pelas regras abaixo, com SHA-256, fronteiras, rótulos, índices de
quadro, desvios e marcas. A planilha precisa bater exatamente com esse
conjunto:

| Problema na planilha | Efeito |
| --- | --- |
| Caso exigido ausente (inclusive pendente ou com `desvio_cfr` apagado da planilha) | Bloqueia |
| `case_id` repetido | Bloqueia |
| Caso fora da amostra determinística | Bloqueia |
| Qualquer coluna automática diferente da reconstruída (`annotated_s`, `boundary`, `label_before`, `label_after`, `source_sha256`, índices e PTS dos quadros, `cfr_offset_s`, `auto_flag` etc.) | Bloqueia o caso, seja qual for o julgamento |
| Coluna faltando ou coluna nova | A planilha é recusada |

Só as colunas humanas podem ser editadas. Números são comparados pelo valor
lido, para aceitar `1.4` e `1.40`; texto, byte a byte.

## Seleção determinística

A amostra não usa RNG. A ordem é o SHA-256 de `gmdcsa24-provenance-v1:<video_id>`,
e as regras são recalculadas a cada execução:

1. **Inclusão obrigatória, com todas as fronteiras do vídeo:**
    - sobreposição OmniFall >= 0,1 s;
    - lacuna OmniFall >= 0,5 s sobre a duração medida;
    - vídeo `Fall` sem segmento `fall`;
    - `fall` sem intervalo `Falling` no CSV do autor;
    - início do `fall` a 1,5 s ou mais do `Falling` do autor;
    - desvio de PTS acima de 0,1 s.
2. **Estratos sujeito × categoria:** dois vídeos a mais por estrato (oito
   estratos), fora as inclusões obrigatórias. Num vídeo `Fall`, entram o início
   e o fim do `fall`; num vídeo `ADL`, a primeira transição sem `fall`. Um
   vídeo sem essa fronteira não é elegível para o estrato.
3. **Reposição depois de exclusão:** um vídeo de estrato com qualquer caso
   `excluded` continua exigido, mas não conta para a cota. O próximo
   candidato do estrato, na mesma ordem, passa a ser exigido, com seleção
   `estrato S<n>/<categoria>, reposição`, até haver dois vídeos não excluídos.
   Se os candidatos acabarem, o estrato bloqueia a aprovação. Vídeos de
   inclusão obrigatória não puxam reposição.

Uma fronteira é o início de um segmento ou o fim seguido de lacuna. O rótulo
anterior ignora o segmento que começa, para que a sobreposição de
arredondamento (< 0,1 s) não vire duas fronteiras. Um início em 0 s não é
fronteira; por isso `Subject_3/Fall/08`, cuja queda começa no primeiro
quadro, só tem o fim do `fall`.

`Subject_2/ADL/04` (`Length` de 70 s no CSV para 7,84 s medidos) não entra:
é metadado do autor, não fronteira OmniFall.

## Tolerância e bloqueio

A tolerância é um passo da grade de 10 fps: 0,1 s.

| Julgamento | Exige | Estado |
| --- | --- | --- |
| `pending` | — | Pendente |
| `confirmed` | `observed_boundary_s` numérico, `reviewer`, `reviewed_at` em ISO 8601; \|observado − anotado\| <= 0,1 s | Confirmado |
| `confirmed` fora da tolerância | — | Bloqueado |
| `discrepancy` | `reviewer`, `reviewed_at` | Bloqueado até investigação |
| `excluded` | `reviewer`, `reviewed_at`, justificativa em `notes` | Excluído do piloto; em vídeo de estrato, exige reposição |

A marca automática `desvio_cfr` indica que o quadro da grade de taxa
constante fica a mais de 0,1 s da fronteira anotada. A anotação pode estar
certa, mas a grade associaria o rótulo ao quadro errado. Esse caso só deixa de
bloquear como `excluded`.

A proveniência é **APROVADA** só se a planilha bater com os casos
reconstruídos, nenhum caso estiver pendente ou bloqueado, todo estrato tiver
dois vídeos com todas as fronteiras confirmadas e houver ao menos um caso
confirmado. Uma revisão toda excluída nunca aprova. `review-status` sai com
código 1 em qualquer outro estado.

## Como executar

Com o clone pinado em `data/raw/gmdcsa24/GMDCSA24` (ver
[viabilidade](gmdcsa24-feasibility.md#fonte-licenca-e-proveniencia-repositorio-publico)):

```bash
uv run python -m gatefall.data.gmdcsa24.audit review-plan
uv run python -m gatefall.data.gmdcsa24.audit review-render
uv run python -m gatefall.data.gmdcsa24.audit review-status
```

Todos aceitam `--root DIR` e `--review-dir DIR`. O padrão de `--review-dir` é
`data/scratch/gmdcsa24/provenance/`.

- `review-plan` exige `verify` OK, decodifica os 160 vídeos (cerca de seis
  minutos), sorteia os casos e grava `review.csv` com todos os casos
  `pending`. Não sobrescreve uma planilha existente sem `--force`, que apaga os
  julgamentos. Com `--extend`, lê a planilha, reconstrói os casos com as
  exclusões registradas e acrescenta só os que faltam, como reposições de
  estrato, como `pending`. Os julgamentos ficam como estão. Recusa estender uma
  planilha com caso repetido, extra ou com evidência alterada.
- `review-render` grava `contact_sheets/<case_id>.png`: os quadros com PTS a
  até ±0,5 s da fronteira, decodificados sem duplicar nem descartar quadros e
  sem áudio. Cada quadro mostra índice, PTS, distância à fronteira e o rótulo
  OmniFall naquele PTS. A borda vermelha marca o primeiro quadro com PTS >=
  fronteira; a azul, o quadro da grade de taxa constante.
- `review-status` exige `verify` OK e reconstrói os casos exigidos, o que
  também leva cerca de seis minutos. Confere a planilha contra eles, avalia
  cada caso intacto e imprime a contagem de casos exigidos, linhas da planilha,
  evidência renderizada, confirmados, excluídos, pendentes e bloqueados.

`review-plan` e `review-render` recusam um `--review-dir` que o Git
versionaria. As contact
sheets mostram rostos e ficam só na máquina local. `review.csv` não tem
quadros; ao fim da revisão, os julgamentos são transcritos para a tabela de
casos abaixo.

## Vídeos selecionados

30 vídeos em oito estratos: 14 por inclusão obrigatória e 16 por estrato.

| Vídeo | Motivo da seleção | `source_sha256` |
| --- | --- | --- |
| `Subject_1/ADL/04` | desvio de PTS de 0,108 s | `c336f1f81cfa647a77ae351ef02fa4e1716639d60d599c7ddec98aac2e3b969a` |
| `Subject_1/ADL/05` | lacuna de 1,00 s | `2b963e9a1be9d1b1c9ae9807f07d8c65b6760f24b0006c7d5538c0e5574e242a` |
| `Subject_1/ADL/12` | estrato S1/ADL | `ba4b775bbccd031d588f065dba1812a3befd46f6eccbf1a9f6a130a3c50a6a94` |
| `Subject_1/ADL/13` | estrato S1/ADL | `406dd142be0bf973302caabf38f1109d939b5f0154ef88158b3adc5a50859d51` |
| `Subject_1/Fall/03` | início do `fall` −1,70 s do autor | `b17c5217c0050c959c45f7ead8ce80fa1c459d62109f25ea72461ac729d5307c` |
| `Subject_1/Fall/04` | início do `fall` −1,58 s do autor | `a5563045241b462989867ea7973cf08cf5a4684ed43d47e104de1f4964794d5d` |
| `Subject_1/Fall/07` | estrato S1/Fall | `e584c98e13100befb24fa1854987d8adb50324e0f73dd6278661a8887b428b86` |
| `Subject_1/Fall/09` | desvio de PTS de 0,101 s | `9bae374ba45c8dc7e30e93445d5d8d130479f16708875b39c5994fe0a984923c` |
| `Subject_1/Fall/11` | estrato S1/Fall | `c25e4fc9b669cb588b6145d2cfd94f2d403083293c08bc85fa863111372bfea7` |
| `Subject_1/Fall/12` | desvio de PTS de 0,131 s | `2354ba6441e0fc2173e283ed34cb2a75a08a45b01144882f8db5291cfb113851` |
| `Subject_1/Fall/15` | início do `fall` −1,54 s do autor | `c428eb70c50a1ba819ee074add3b5389f6494273eec8228c697a807fc48445bb` |
| `Subject_2/ADL/12` | estrato S2/ADL | `24559694cceadbcf1bb34f217967c233e412b97e25f9dd9be3f733f4e16067df` |
| `Subject_2/ADL/13` | estrato S2/ADL | `d231d21a135524718e384bd6a148e4f6f3ad7dbe3e4ced1dbb8f0cdd548e979d` |
| `Subject_2/ADL/21` | lacuna de 1,26 s | `508789b90c215b33d0b103aa13ae6a10e086953de1373caa9c541a09b0e19f82` |
| `Subject_2/Fall/03` | vídeo Fall sem `fall` no OmniFall | `c73bb07da0e58e21153f19ec13cd8edd403736f529066ddbdff3d90be32f8fd9` |
| `Subject_2/Fall/15` | início do `fall` −1,56 s do autor | `4f288257071fb98f363a771c40a24a0088100c4b9b15b770d95b4a0ef599fa63` |
| `Subject_2/Fall/21` | estrato S2/Fall | `e71633817f5f578b928496f6247262b431a1e68ac8bad423f03362f46071cb0b` |
| `Subject_2/Fall/23` | estrato S2/Fall | `7c9d0973f9a1561e93ec4c8b29208a9884a0dd09033ddb412fbac56d0ed409d1` |
| `Subject_3/ADL/09` | sobreposição de 4,30 s | `5100230c1f7d22a400ca810b9eef7162853cb556484648bf5c9fbf1844df618d` |
| `Subject_3/ADL/10` | estrato S3/ADL | `145f29516fa6c06ecec6f47d024b86898e930fe95286d136a607084d90b64a1d` |
| `Subject_3/ADL/14` | estrato S3/ADL | `4f8d1eb75dc0286a6b99687e25b9adb00f847c9c5b2480d64850dd865b6f2e29` |
| `Subject_3/Fall/02` | estrato S3/Fall | `6d9539cbb137c9de305f05d2d4b49dc45217aaae711ee26ca72da58718eeaf38` |
| `Subject_3/Fall/08` | desvio de PTS de 0,104 s | `39c2bcd61f24ebf38e05fc00cd680cad60688c5a5ad43f642e2ea5833b83da32` |
| `Subject_3/Fall/09` | sobreposição de 0,10 s | `a3149bb1c897d47f7354e33db312ff6ca8e51bfbd702191c03a7b08ac82143e8` |
| `Subject_3/Fall/18` | estrato S3/Fall | `abaced84cd40b09fa0a31dddad8d947247bf3a717878fb423c1cc4de782a6a86` |
| `Subject_4/ADL/14` | estrato S4/ADL | `e337e1248f012e5db82c5e92879add0fc5bb747b65ae31762b38cca6909f90d4` |
| `Subject_4/ADL/16` | estrato S4/ADL | `20895c4a7ce5ff072158cc9c16675e2cbd39dc7c472ad2e511f097123918a4db` |
| `Subject_4/Fall/06` | estrato S4/Fall | `aa074bdcf93026e60ea764752763e836c4c0b1b15eaf70b39cc7e1708421fefb` |
| `Subject_4/Fall/12` | estrato S4/Fall | `dbea18d773a46e9af2911787bc909d9eca24cebc9a45329563ed884ff68222b4` |
| `Subject_4/Fall/15` | `fall` sem intervalo Falling no CSV original | `4a1867bbff15a80893f081419b2953bb55b0fccb29e66c80c809dd8e0c02144f` |

## Casos

52 fronteiras: 16 inícios de `fall`, 17 fins de `fall`, 16 transições sem
`fall`, 1 início e 2 fins de lacuna. O quadro CFR é o índice que
`build_time_grid` usaria; o desvio é o PTS real desse quadro menos o instante
anotado.

| Vídeo | Fronteira | Anotado (s) | Esperado | Quadro CFR (desvio) | Marca | Revisão |
| --- | --- | ---: | --- | --- | --- | --- |
| `Subject_1/ADL/04` | `transition` | 1,140 | `lying` → `stand_up` | #33 (+0,076 s) | — | `pending` |
| `Subject_1/ADL/04` | `transition` | 3,580 | `stand_up` → `sitting` | #105 (+0,052 s) | — | `pending` |
| `Subject_1/ADL/05` | `gap_end` | 1,000 | `sem_rotulo` → `walk` | #30 (+0,024 s) | — | `pending` |
| `Subject_1/ADL/05` | `transition` | 2,080 | `walk` → `sit_down` | #62 (+0,016 s) | — | `pending` |
| `Subject_1/ADL/05` | `transition` | 3,780 | `sit_down` → `other` | #113 (+0,028 s) | — | `pending` |
| `Subject_1/ADL/12` | `transition` | 2,580 | `other` → `stand_up` | #77 (+0,028 s) | — | `pending` |
| `Subject_1/ADL/13` | `transition` | 0,640 | `standing` → `walk` | #19 (+0,032 s) | — | `pending` |
| `Subject_1/Fall/03` | `fall_onset` | 0,800 | `sitting` → `fall` | #24 (+0,032 s) | — | `pending` |
| `Subject_1/Fall/03` | `fall_end` | 3,280 | `fall` → `fallen` | #97 (+0,000 s) | — | `pending` |
| `Subject_1/Fall/04` | `fall_onset` | 0,720 | `sitting` → `fall` | #21 (+0,048 s) | — | `pending` |
| `Subject_1/Fall/04` | `fall_end` | 3,060 | `fall` → `fallen` | #90 (+0,012 s) | — | `pending` |
| `Subject_1/Fall/07` | `fall_onset` | 1,000 | `walk` → `fall` | #30 (+0,040 s) | — | `pending` |
| `Subject_1/Fall/07` | `fall_end` | 2,740 | `fall` → `fallen` | #81 (+0,012 s) | — | `pending` |
| `Subject_1/Fall/09` | `fall_onset` | 0,640 | `standing` → `fall` | #19 (+0,096 s) | — | `pending` |
| `Subject_1/Fall/09` | `fall_end` | 2,160 | `fall` → `fallen` | #63 (+0,048 s) | — | `pending` |
| `Subject_1/Fall/11` | `fall_onset` | 0,520 | `standing` → `fall` | #15 (+0,008 s) | — | `pending` |
| `Subject_1/Fall/11` | `fall_end` | 2,700 | `fall` → `fallen` | #80 (+0,004 s) | — | `pending` |
| `Subject_1/Fall/12` | `fall_onset` | 1,660 | `standing` → `fall` | #49 (+0,116 s) | `desvio_cfr` | `pending` |
| `Subject_1/Fall/12` | `fall_end` | 3,780 | `fall` → `fallen` | #111 (+0,060 s) | — | `pending` |
| `Subject_1/Fall/15` | `fall_onset` | 0,260 | `sitting` → `fall` | #8 (+0,044 s) | — | `pending` |
| `Subject_1/Fall/15` | `fall_end` | 3,620 | `fall` → `fallen` | #107 (−0,004 s) | — | `pending` |
| `Subject_2/ADL/12` | `transition` | 5,520 | `lie_down` → `lying` | #165 (+0,001 s) | — | `pending` |
| `Subject_2/ADL/13` | `transition` | 2,600 | `sit_down` → `sitting` | #78 (+0,008 s) | — | `pending` |
| `Subject_2/ADL/21` | `transition` | 0,640 | `standing` → `other` | #10 (+0,016 s) | — | `pending` |
| `Subject_2/ADL/21` | `gap_start` | 5,340 | `other` → `sem_rotulo` | #80 (+0,004 s) | — | `pending` |
| `Subject_2/ADL/21` | `gap_end` | 6,600 | `sem_rotulo` → `walk` | #99 (+0,008 s) | — | `pending` |
| `Subject_2/Fall/03` | `transition` | 1,760 | `lie_down` → `lying` | #52 (−0,016 s) | — | `pending` |
| `Subject_2/Fall/15` | `fall_onset` | 6,140 | `sitting` → `fall` | #183 (−0,012 s) | — | `pending` |
| `Subject_2/Fall/15` | `fall_end` | 9,540 | `fall` → `fallen` | #285 (+0,012 s) | — | `pending` |
| `Subject_2/Fall/21` | `fall_onset` | 6,200 | `sem_rotulo` → `fall` | #185 (+0,008 s) | — | `pending` |
| `Subject_2/Fall/21` | `fall_end` | 7,400 | `fall` → `fallen` | #221 (+0,008 s) | — | `pending` |
| `Subject_2/Fall/23` | `fall_onset` | 3,280 | `other` → `fall` | #98 (−0,000 s) | — | `pending` |
| `Subject_2/Fall/23` | `fall_end` | 4,900 | `fall` → `fallen` | #146 (−0,004 s) | — | `pending` |
| `Subject_3/ADL/09` | `transition` | 2,500 | `stand_up` → `sitting` | #75 (+0,012 s) | — | `pending` |
| `Subject_3/ADL/09` | `transition` | 6,200 | `stand_up` → `standing` | #185 (+0,008 s) | — | `pending` |
| `Subject_3/ADL/10` | `transition` | 2,341 | `walk` → `standing` | #70 (+0,011 s) | — | `pending` |
| `Subject_3/ADL/14` | `transition` | 2,487 | `sit_down` → `sitting` | #37 (−0,007 s) | — | `pending` |
| `Subject_3/Fall/02` | `fall_onset` | 2,600 | `sem_rotulo` → `fall` | #78 (+0,024 s) | — | `pending` |
| `Subject_3/Fall/02` | `fall_end` | 4,362 | `fall` → `fallen` | #130 (−0,010 s) | — | `pending` |
| `Subject_3/Fall/08` | `fall_end` | 2,612 | `fall` → `fallen` | #77 (+0,060 s) | — | `pending` |
| `Subject_3/Fall/09` | `fall_onset` | 3,300 | `sitting` → `fall` | #99 (+0,012 s) | — | `pending` |
| `Subject_3/Fall/09` | `fall_end` | 4,883 | `fall` → `fallen` | #146 (−0,003 s) | — | `pending` |
| `Subject_3/Fall/18` | `fall_onset` | 4,070 | `stand_up` → `fall` | #61 (+0,010 s) | — | `pending` |
| `Subject_3/Fall/18` | `fall_end` | 6,841 | `fall` → `fallen` | #103 (+0,055 s) | — | `pending` |
| `Subject_4/ADL/14` | `transition` | 0,758 | `stand_up` → `walk` | #23 (+0,010 s) | — | `pending` |
| `Subject_4/ADL/16` | `transition` | 1,049 | `sit_down` → `sitting` | #31 (−0,009 s) | — | `pending` |
| `Subject_4/Fall/06` | `fall_onset` | 0,660 | `sitting` → `fall` | #20 (+0,012 s) | — | `pending` |
| `Subject_4/Fall/06` | `fall_end` | 3,740 | `fall` → `fallen` | #112 (+0,005 s) | — | `pending` |
| `Subject_4/Fall/12` | `fall_onset` | 0,920 | `sitting` → `fall` | #27 (−0,008 s) | — | `pending` |
| `Subject_4/Fall/12` | `fall_end` | 2,280 | `fall` → `fallen` | #68 (+0,008 s) | — | `pending` |
| `Subject_4/Fall/15` | `fall_onset` | 1,180 | `sitting` → `fall` | #35 (−0,012 s) | — | `pending` |
| `Subject_4/Fall/15` | `fall_end` | 2,580 | `fall` → `fallen` | #77 (−0,004 s) | — | `pending` |

## Riscos não resolvidos

- **Nenhuma anotação confirmada.** A evidência automática só mostra que os
  arquivos são os da `v2.1` e onde estão os quadros. A coincidência entre ação
  e rótulo ainda depende de revisão humana.
- **`Subject_1/Fall/12`.** O início do `fall` fica a +0,116 s do quadro da grade
  de taxa constante. O caso bloqueia até ser `excluded`.
- **Inícios divergentes do autor.** Em quatro vídeos, o `fall` do OmniFall
  começa 1,5 s ou mais antes do `Falling` do autor. O revisor decide se isso é
  a definição do OmniFall (primeiro quadro em que o movimento diverge) ou um
  deslocamento.
- **Decodificação de taxa variável.** O pipe de `gatefall.data.video_io` duplica
  quadros nos MP4 do GMDCSA24 (206 para 200 em `Subject_1/Fall/12`). A revisão
  usa `-fps_mode passthrough`; um adapter futuro precisa fazer o mesmo (V5 do
  protocolo).
- **Amostra.** 52 fronteiras em 30 de 160 vídeos. Uma revisão aprovada vale
  para a amostra, não prova os 128 vídeos restantes.

## Verificação manual pendente

1. Abrir cada contact sheet e, se preciso, o MP4 no instante anotado.
2. Para cada caso, registrar `review_outcome`, `observed_boundary_s` (instante
   do primeiro quadro com o rótulo de depois, em PTS), `reviewer`,
   `reviewed_at` e `notes`.
3. Marcar `Subject_1/Fall/12` como `excluded` ou registrar a divergência, e
   investigar toda `discrepancy` antes de aprovar.
4. Se algum vídeo de estrato for excluído, rodar `review-plan --extend`,
   gerar as contact sheets das reposições com `review-render` e revisá-las.
5. Rodar `review-status` até `APROVADA` e transcrever os julgamentos para esta
   página num PR.
