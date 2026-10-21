# Análise — Viabilidade do CMDFall pelos metadados do OmniFall

!!! warning "Auditoria exploratória, não canônica e somente de metadados"
    Esta página usa só os CSVs públicos do CMDFall no OmniFall. Nenhum vídeo
    foi baixado, nenhuma pose foi extraída e nada foi treinado. Nenhum
    baseline, artefato canônico, registro de dataset ou contrato experimental
    do GateFall foi alterado, e nenhum número do `test` oficial do Le2i é
    critério de escolha. As saídas ficam em `data/scratch/cmdfall/`, ignorado
    pelo Git.

`gatefall.data.cmdfall.audit` mede o que os CSVs do CMDFall permitem
verificar antes de qualquer pedido de acesso aos vídeos: contagens e durações
por rótulo, sujeitos, sessões e câmeras, duplicação entre vistas
sincronizadas, cobertura de classes nos splits, gaps e sobreposições entre
segmentos e quantas janelas causais de 24 quadros a 10 fps a anotação
comportaria. A página separa três tipos de informação:

- **fato do CSV**: calculado pelo comando `report` sobre os arquivos fixados;
- **afirmação publicada**: texto do OmniFall ou do artigo original, que o CSV
  confirma, contradiz ou não alcança;
- **bloqueado até o vídeo**: depende de duração, FPS, imagem ou pessoa
  visível e não pode ser decidido por metadados.

## Motivação

Os diagnósticos da branch `analysis/gatefall-improvements` (não integrada a
`main`; páginas `docs/analysis/cross-arm-diagnostic-synthesis.md`,
`intervention-decision.md` e `other-walk-train-composition.md`) apontam que
o limite atual do GateFall está no suporte de avaliação do Le2i, não na fusão:

- `other` participa de 49% a 57% dos erros dos cinco braços, e `other→walk`
  é a confusão dominante;
- `standing` de `test` do Le2i-CS são 43 janelas de 3 trechos, todos de um
  sujeito; `standing` de `val` são 96 janelas de 4 vídeos de um sujeito;
- `other` de `val` do Le2i-CS são 34 janelas de 3 trechos, contra 860 em
  `test`, de modo que o split de desenvolvimento quase não mede o erro
  dominante;
- a decisão registrada é não mudar modelo sem sinal em `train`/`val`.

Esses números entram aqui só como contexto. A pergunta desta auditoria é se
um dataset encenado com mais sujeitos daria a `other`, `walk`, `standing`,
`fall` e `fallen` suporte por sujeito suficiente para um desenvolvimento
menos dependente de um único sujeito, sem mudar a taxonomia nem o contrato
temporal.

## Fonte e reprodução

| Campo | Valor |
| --- | --- |
| Repositório | `simplexsigil2/omnifall` |
| Revisão | `68e5cee56a4bad38cca4aea791cac248f96e79a0`, a mesma das anotações Le2i |
| Segmentos | `labels/cmdfall.csv` |
| Taxonomia | `labels/label2id.csv` |
| Splits cross-subject | `splits/cs/cmdfall/{train,val,test}.csv` |
| Splits cross-view | `splits/cv/cmdfall/{train,val,test}.csv` |

Os oito arquivos têm SHA-256 fixado em `gatefall.data.cmdfall.source`. Na
head `83572a37b9e3081df8c06a56874b1d1f2a19386c` (2026-08-27), os mesmos
oito arquivos têm SHA-256 idêntico; a diferença entre as revisões, para o
CMDFall, é só o `KNOWN_PITFALLS.md` citado adiante.

```bash
uv run python -m gatefall.data.cmdfall.audit fetch
uv run python -m gatefall.data.cmdfall.audit report
uv run python -m gatefall.data.cmdfall.audit selftest
```

`fetch` baixa apenas os CSVs, em `data/scratch/cmdfall/omnifall/`, e confere
o SHA-256. `report` recusa fontes ausentes ou alteradas, valida o schema, a
taxonomia, a identidade das vistas sincronizadas e os splits cross-subject,
imprime as tabelas e grava `data/scratch/cmdfall/audit.json` (sobrescrito a
cada execução, sem valor canônico). O relatório só é gravado quando todas
essas validações passam; se alguma falha, o comando termina com erro e o
destino, inclusive um relatório anterior, fica intacto. O `selftest` não
acessa a rede: usa CSVs sintéticos em memória e num repositório temporário.

As escritas ficam confinadas a `data/scratch/cmdfall/`, que precisa ser o
diretório físico `<raiz>/data/scratch/cmdfall`, sem link simbólico no
caminho. Cada destino (`--output` e cada CSV que o `fetch` grava sob
`--source-dir`) é resolvido antes de qualquer escrita, seguindo `..` e links
simbólicos, e é recusado se sair dessa área ou cair em `runs/reference/`,
`src/`, `scripts/`, `docs/`, `.git/`, `.github/` ou nos dados canônicos
(`data/raw/`, `data/labels/`, `data/processed/`, `data/features/`).
`--output` também precisa terminar em `.json`, não pode ser um diretório e
não pode ficar dentro do diretório das fontes. Os caminhos relativos são
lidos a partir do diretório atual, que deve ser a raiz do repositório.

O `fetch` baixa num diretório temporário fora do repositório, confere o
SHA-256 de todos os CSVs e só então revalida os destinos e publica os
arquivos; uma divergência não publica nenhum. Toda gravação, do relatório e
dos CSVs, usa um temporário de nome imprevisível criado de forma exclusiva no
diretório do destino, seguido de troca atômica: um link simbólico deixado no
caminho nunca é seguido, e, se a gravação falhar, o temporário é removido e
o arquivo anterior fica intacto.

## Contrato preservado

- **Taxonomia congelada.** O `label2id.csv` do OmniFall tem os ids 0–9 na
  mesma ordem de `LE2I_LABEL_NAMES` (fato do CSV; o `report` falha se
  divergir). O CMDFall usa os dez rótulos. `lie_down` e `lying` existem no
  CMDFall, mas continuam fora de `RESTRICTED_CLASSES`; esta auditoria não
  propõe mudar a métrica.
- **Contrato temporal.** A grade usa `TARGET_FPS = 10`, `WINDOW_FRAMES = 24`,
  `EVAL_STRIDE = 1`, `TRAIN_STRIDE = 4` e as mesmas funções do Le2i
  (`labels_for_grid`, `window_end_indices`, `window_frame_indices`):
  intervalo semiaberto, vitória do menor `start` em sobreposição, rótulo do
  quadro final e replicação de borda no início.
- **RGB monocular.** Todos os paths do CSV começam em `colors/`. O CMDFall é
  publicado como multimodal (título do artigo original, Tran et al., ICPR
  2018); profundidade e acelerômetro não entram em nenhuma etapa.
- **Split por sujeito.** Só o protocolo `cs` é compatível com o invariante 2;
  o `cv` não é ([seção própria](#splits-cross-view)).

## Fatos verificáveis pelo CSV

### Inventário

| Medida | Valor |
| --- | ---: |
| Segmentos (todas as vistas) | 42.143 |
| Vídeos | 384 |
| Sujeitos | 50 |
| Sessões (`S` do path) | 55 |
| Câmeras | 1–7 |
| Sessões com 7 vistas / 6 vistas | 54 / 1 (`S39` sem a câmera 1) |
| Sujeitos com duas sessões | 1, 2, 3, 4 e 24 |

O path segue `colors/S<sessão>P<sujeito>K<câmera>`. `P` coincide com a
coluna `subject` e `K` com `cam` em todas as linhas; cada `S` tem um único
sujeito. Chamar `S` de sessão é uma leitura do layout, não um campo
documentado.

### Duplicação por vistas sincronizadas

Nas 55 sessões, todas as vistas têm a mesma sequência de `(label, start,
end)`, com os mesmos tempos: o deslocamento entre câmeras no CSV é zero. A
anotação de uma sessão é, portanto, uma única linha do tempo copiada para até
sete vídeos. Por isso as contagens abaixo usam **uma vista por sessão** (a
de menor câmera); multiplicar por sete conta a mesma atividade sete vezes,
e as vistas de uma sessão nunca podem ficar em splits diferentes.

| Medida | CSV, uma vista (SV) | CSV, todas as vistas (MV) | Publicado pelo OmniFall |
| --- | ---: | ---: | ---: |
| Segmentos | 6.026 | 42.143 | 6.026 / 42.143 |
| Duração | 7,12 h | 49,80 h | 7,12 h / 49,80 h |
| Vídeos | — | 384 | 384 |

As cinco estatísticas publicadas em `statistics.md` coincidem com o CSV.

### Rótulos

Uma vista por sessão. "≥ 2,4 s" conta os segmentos com duração de pelo menos
uma janela.

| Rótulo | Segmentos | Horas | Duração p25 / mediana / p75 (s) | ≥ 2,4 s | Sessões | Sujeitos |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **`walk`** | 1149 | 2,07 | 1,76 / 2,80 / 6,19 | 664 | 55 | 50 |
| **`fall`** | 828 | 0,60 | 1,98 / 2,46 / 3,11 | 446 | 55 | 50 |
| **`fallen`** | 690 | 0,76 | 2,07 / 3,51 / 5,08 | 482 | 55 | 50 |
| `sit_down` | 446 | 0,23 | 1,27 / 1,71 / 2,17 | 79 | 55 | 50 |
| `sitting` | 250 | 0,15 | 0,99 / 1,42 / 2,24 | 58 | 52 | 49 |
| `lie_down` | 290 | 0,27 | 2,44 / 3,18 / 3,85 | 224 | 50 | 50 |
| `lying` | 151 | 0,08 | 1,00 / 1,47 / 2,11 | 32 | 47 | 47 |
| `stand_up` | 1253 | 1,15 | 2,23 / 3,21 / 4,15 | 910 | 55 | 50 |
| **`standing`** | 327 | 0,34 | 1,20 / 2,04 / 4,18 | 150 | 53 | 49 |
| **`other`** | 642 | 1,47 | 2,23 / 3,50 / 9,72 | 433 | 51 | 50 |

Os cinco rótulos prioritários aparecem em 49 ou 50 dos 50 sujeitos. `fall`
tem 828 segmentos com mediana de 2,46 s, próxima dos 2,4 s da janela.
`standing` é curto: menos da metade dos segmentos chega a 2,4 s.

### O que `other` não informa

O CSV não traz o identificador da ação original do CMDFall. Esta auditoria
não deduz ações originais a partir de `other`, nem da posição, da duração ou
dos vizinhos de um segmento `other`; só conta segmentos, durações, sessões e
sujeitos. A definição publicada no `LABELS.md` do OmniFall é "qualquer outra
atividade, incluindo, por exemplo, andar empurrando um objeto como uma
cadeira". Se o `other` do CMDFall se parece com o do Le2i, só o vídeo pode
dizer.

### Fronteiras: gaps e sobreposições

Uma vista por sessão, segmentos consecutivos ordenados por início.
Diferenças de até 1 ms entre fim e início vêm da formatação decimal dos CSVs
(por exemplo, 0,00014 s) e contam como contíguas.

| Fronteira | Quantidade |
| --- | ---: |
| Pares consecutivos | 5971 |
| Contíguos (≤ 1 ms) | 5563 (93,2%) |
| Gap menor que um passo da grade (0,1 s) | 143 |
| Gap de 0,1 s ou mais | 170 |
| Sobreposição menor que 0,1 s | 70 |
| Sobreposição de 0,1 s ou mais | 25 |
| Segmentos contidos no anterior | 8 |

- Gaps interiores somam 238,5 s. Sobreposições somam 93,9 s e aparecem em
  26 das 55 sessões.
- Em 30 sessões, o primeiro segmento começa depois de `t = 0`, somando
  532,0 s não anotados no início. O fim do vídeo não está no CSV, então não
  há gap final mensurável.
- Entre as 25 sobreposições de 0,1 s ou mais, os pares mais frequentes são
  `other→walk` (4), `walk→fall` (4) e `walk→other` (3). Algumas são longas;
  por exemplo, em `S55P18`, um `fallen` de 235,98 s a 259,54 s cobre um
  `stand_up` de 241,5 s a 243,5 s. A grade resolve como no Le2i, mas a sobreposição
  longa é provável erro de anotação e só o vídeo decide.

Na grade de 10 fps sobre a extensão anotada `[0, maior end)`, 7709 de
263.019 quadros (2,9%) ficam `IGNORE_LABEL` e 944 são resolvidos por
sobreposição; nenhum segmento fica sem quadro.

### Splits cross-subject

| Split | Vídeos | Sessões | Sujeitos |
| --- | ---: | ---: | ---: |
| `train` | 188 | 27 | 25 |
| `val` | 42 | 6 | 5 |
| `test` | 154 | 22 | 20 |

Os três arquivos são disjuntos e cobrem os 384 vídeos; nenhum sujeito ou
sessão aparece em mais de um split, e as sete vistas de cada sessão ficam no
mesmo split. O nome "cross-subject" é, portanto, confirmado pelo CSV.

Cobertura dos rótulos prioritários, uma vista por sessão
(segmentos / sessões / sujeitos):

| Rótulo | `train` | `val` | `test` |
| --- | --- | --- | --- |
| `other` | 298 / 25 / 25 | 65 / 6 / 5 | 279 / 20 / 20 |
| `walk` | 544 / 27 / 25 | 90 / 6 / 5 | 515 / 22 / 20 |
| `standing` | 169 / 27 / 25 | 34 / 5 / 5 | 124 / 21 / 19 |
| `fall` | 432 / 27 / 25 | 65 / 6 / 5 | 331 / 22 / 20 |
| `fallen` | 347 / 27 / 25 | 63 / 6 / 5 | 280 / 22 / 20 |

`val` tem só 5 sujeitos. Ainda assim, todos os rótulos prioritários de `val`
aparecem em 5 sujeitos, contra um sujeito para `standing` no Le2i-CS.

### Janelas causais potenciais

Janelas de 24 quadros a 10 fps que terminam em `k_end`, com rótulo do quadro
final diferente de `IGNORE_LABEL`, sobre a grade da extensão anotada e uma
vista por sessão. "Puras" têm os 24 quadros com o rótulo final; "com borda"
usam replicação do quadro 0; "com IGNORE" têm algum quadro não anotado.

Stride 1 (janelas / puras / sujeitos):

| Rótulo | `train` | `val` | `test` |
| --- | --- | --- | --- |
| `other` | 25.404 / 19.724 / 25 | 7239 / 6111 / 5 | 20.030 / 14.712 / 20 |
| `walk` | 34.903 / 24.488 / 25 | 6478 / 4773 / 5 | 32.572 / 22.545 / 20 |
| `standing` | 6281 / 3499 / 25 | 1127 / 550 / 5 | 4886 / 2782 / 19 |
| `fall` | 11.178 / 2191 / 25 | 1585 / 266 / 5 | 8889 / 1781 / 20 |
| `fallen` | 12.962 / 5807 / 25 | 4265 / 2883 / 5 | 9947 / 4322 / 20 |

Em stride 4, `train` teria 6356 janelas `other`, 8730 `walk`, 1565
`standing`, 2795 `fall` e 3246 `fallen`. Só `walk` e `standing` têm janelas
com replicação de borda (sessões cuja anotação começa em `t = 0` com esses
rótulos): `walk` tem 184 em `train` e 130 em `test`; `standing`, 156 e 190;
nenhuma em `val`.
O relatório traz as dez classes e as colunas de borda e `IGNORE`.

Esses números são um teto da anotação, não uma contagem de janelas
utilizáveis. Eles assumem que cada vídeo dura até o maior `end`, que todos
os quadros existem e que a pessoa é visível; cada uma dessas condições está
na [lista de bloqueios](#bloqueado-ate-o-acesso-aos-videos). Somar as sete
vistas multiplicaria as contagens sem acrescentar atividade nova.

### Splits cross-view

| Split | Vídeos | Sessões | Sujeitos | Câmeras |
| --- | ---: | ---: | ---: | --- |
| `train` | 54 | 54 | 50 | 1 |
| `val` | 55 | 55 | 50 | 5 |
| `test` | 220 | 55 | 50 | 2, 3, 4, 6 |
| fora dos splits | 55 | 55 | 50 | 7 |

O `cv` separa por câmera: as 55 sessões e os 50 sujeitos aparecem em mais de
um split, com a mesma atividade vista de outra câmera. Isso viola o
invariante 2 (sem vídeo nem sujeito dos dois lados) e exclui o `cv` de
qualquer uso no GateFall.

## Afirmações publicadas

| Afirmação | Fonte | Situação |
| --- | --- | --- |
| 384 vídeos, 6.026 segmentos SV, 42.143 MV, 7,12 h SV, 49,80 h MV, 7 vistas | `statistics.md`, `README.md` do OmniFall | Confirmada pelo CSV |
| Os splits `cs` são cross-subject | `CONFIGS.md` | Confirmada pelo CSV |
| Os vídeos foram reanotados com as definições do OmniFall | `LABELS.md` | Não verificável pelo CSV |
| 76 de 42.143 segmentos, em 75 de 384 vídeos, terminam depois do vídeo; maior excesso de 3,80 s em `colors/S22P36K6` | `KNOWN_PITFALLS.md` (head `83572a37`) | Bloqueada até o vídeo |
| As vistas secundárias são a vista de referência deslocada por um offset por câmera | `KNOWN_PITFALLS.md` | Não observável: no CSV, todas as vistas têm os mesmos tempos |
| Anotações e splits em CC BY-NC-SA 4.0; vídeos obtidos nas fontes originais | `README.md` do OmniFall | Termos dos vídeos do CMDFall não verificados |
| Dataset multimodal e multivista | Tran et al., ICPR 2018 | Só a parte multivista é visível no CSV |

A divergência de licença no card do OmniFall já está registrada em
[OmniFall](../data/omnifall.md#termos-publicados-pelo-upstream).

## Bloqueado até o acesso aos vídeos

- Termos e forma de acesso aos vídeos do CMDFall na página do MICA.
- FPS, resolução, codec, duração e número de quadros de cada vídeo; logo, o
  número real de quadros da grade e as janelas no fim de cada vídeo.
- Os 76 segmentos publicados como posteriores ao fim do vídeo e o efeito de
  descartá-los ou truncá-los.
- A sincronização real entre câmeras, já que o CSV não traz offsets.
- Se as sobreposições longas e os 532 s iniciais sem anotação são erro de
  anotação, sala vazia ou preparação.
- Visibilidade da pessoa, oclusão, distância e iluminação em RGB monocular,
  e a cobertura de detecção do YOLO-Pose, inclusive o equivalente ao
  `standing` sem pose do prefixo de `home_01/video_15` do Le2i.
- Se `other`, `walk` e `standing` do CMDFall têm o mesmo conteúdo visual que
  no Le2i.

## Preflight condicionado ao acesso

!!! warning "Plano, não execução"
    Esta seção só planeja o que inspecionar se o acesso aos vídeos for
    concedido. Ela não autoriza pedido de acesso, download, extração nem
    treino, e foi derivada apenas dos oito CSVs fixados acima.

### Sessões, vistas e sujeitos

- **Sessão** (`S` do path): uma linha do tempo anotada, de um único sujeito.
  É a unidade da pré-seleção, do split e da exclusão de sessão.
- **Vista** (`K` do path): o vídeo de uma câmera daquela sessão,
  `colors/S<sessão>P<sujeito>K<câmera>`. O texto da anotação é o mesmo em
  todas as vistas, mas sua correspondência com o vídeo não é: o CSV não traz
  offsets e o `KNOWN_PITFALLS.md` publica um deslocamento por câmera. Tempo
  de quadro, sincronização, alinhamento da anotação, visibilidade da pessoa
  e cobertura de pose são medidos em cada vista e não se transferem de uma
  câmera para outra.
- **Sujeito** (`P`): pode ter duas sessões. No `train` do `cs`, o sujeito 1
  tem `S7` e `S39`, e o sujeito 3 tem `S19` e `S32`. Toda separação por
  sujeito leva as duas sessões juntas.

### Pré-seleção para inspeção

Regras, aplicadas aos CSVs fixados:

1. **Universo:** as 27 sessões (25 sujeitos) do `train` do `cs`. `val` e
   `test` ficam fora, para que a inspeção nunca veja gravações de avaliação.
2. **Medidas:** sobre a vista de referência (menor câmera), com segmentos
   ordenados por início e a tolerância de 1 ms das
   [fronteiras](#fronteiras-gaps-e-sobreposicoes).
3. **Estratos, nesta ordem:** cada estrato escolhe a sessão de maior valor
   do critério entre as elegíveis cujo sujeito ainda não foi escolhido; o
   empate vai para o menor número de sessão.

| Estrato | Critério | Elegíveis | Sessão | Valor |
| --- | --- | ---: | --- | --- |
| Vista ausente | Menos de 7 vistas | 1 | `S39P1` | sem a câmera 1 |
| Sobreposição | Maior sobreposição entre segmentos consecutivos | 8 | `S50P47` | 15,85 s |
| Gap interior | Maior gap interior | 16 | `S9P33` | 27,15 s |
| Início sem anotação | Maior início do primeiro segmento | 14 | `S24P37` | 82,07 s |
| `other` e `standing` | Maior mínimo entre segmentos `other` e `standing` | 27 | `S5P13` | 16 (empate com `S34P27`) |
| Referência limpa | Os cinco rótulos prioritários, início em `t = 0` e todas as fronteiras contíguas | 3 | `S15P31` | — (`S15`, `S21`, `S40`) |

O estrato `other`/`standing` vem dos dois rótulos com menor suporte de
desenvolvimento no Le2i-CS ([Motivação](#motivacao)). As irregularidades
escolhidas são as que o CSV permite medir:

- em `S50`, um `other` de 320,80 s a 338,20 s contém um `walk` de 322,35 s a
  336,45 s, a mesma dupla da confusão `other→walk`;
- `S9` começa em 30,12 s e tem 25 gaps de 0,1 s ou mais, o maior entre
  299,99 s e 327,14 s;
- em `S24`, o primeiro segmento é um `stand_up` em 82,07 s;
- `S39` é a única sessão sem a câmera 1 e não tem `other`.

Os 76 segmentos que terminam depois do vídeo não entram como critério: a
lista está só no `KNOWN_PITFALLS.md` da head, não nos arquivos fixados, e o
exemplo publicado (`colors/S22P36K6`) é de `test`.

| Sessão | Sujeito | Vistas | Segmentos | Extensão anotada (s) | `other` | `walk` | `standing` | `fall` | `fallen` |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S39` | 1 | K2–K7 | 39 | 109,20 | 0 | 6 | 1 | 4 | 4 |
| `S50` | 47 | K1–K7 | 109 | 519,14 | 17 | 28 | 9 | 12 | 12 |
| `S9` | 33 | K1–K7 | 138 | 559,92 | 18 | 14 | 2 | 22 | 21 |
| `S24` | 37 | K1–K7 | 115 | 582,10 | 13 | 15 | 3 | 18 | 11 |
| `S5` | 13 | K1–K7 | 163 | 508,00 | 17 | 28 | 16 | 20 | 21 |
| `S15` | 31 | K1–K7 | 88 | 498,90 | 7 | 10 | 9 | 15 | 10 |

São 6 sessões de 6 sujeitos e 41 vídeos. `walk`, `standing`, `fall` e
`fallen` aparecem nas seis sessões; `other`, em cinco. A amostra cobre
casos, não estima taxas do CMDFall. Paths exatos:

```text
colors/S39P1K2  colors/S39P1K3  colors/S39P1K4  colors/S39P1K5  colors/S39P1K6  colors/S39P1K7
colors/S50P47K1 colors/S50P47K2 colors/S50P47K3 colors/S50P47K4 colors/S50P47K5 colors/S50P47K6 colors/S50P47K7
colors/S9P33K1  colors/S9P33K2  colors/S9P33K3  colors/S9P33K4  colors/S9P33K5  colors/S9P33K6  colors/S9P33K7
colors/S24P37K1 colors/S24P37K2 colors/S24P37K3 colors/S24P37K4 colors/S24P37K5 colors/S24P37K6 colors/S24P37K7
colors/S5P13K1  colors/S5P13K2  colors/S5P13K3  colors/S5P13K4  colors/S5P13K5  colors/S5P13K6  colors/S5P13K7
colors/S15P31K1 colors/S15P31K2 colors/S15P31K3 colors/S15P31K4 colors/S15P31K5 colors/S15P31K6 colors/S15P31K7
```

### Checklist GO / NO-GO

As condições M1–M6 terminam em GO, NO-GO ou INCONCLUSIVO. As checagens
V1–V8 são feitas por vista ou por sessão e podem, além disso, tirar de uso
uma vista ou excluir uma sessão sem decidir nada sobre o dataset
([desfechos](#desfechos-de-v4-v5-e-v8)). INCONCLUSIVO bloqueia o preflight
até ser resolvido, e nenhuma resolução usa métrica de modelo nem o `test`
de qualquer dataset. O preflight é GO quando M1–M6 estão em GO, nenhuma
checagem terminou em NO-GO do dataset, nenhum INCONCLUSIVO está aberto e as exclusões
estão registradas.

**Condições obrigatórias, decidíveis sem vídeo RGB:**

| # | Condição | GO quando |
| --- | --- | --- |
| M1 | Acesso | Concedido pelo canal oficial dos autores do CMDFall, com o registro guardado; nenhum espelho ou terceiro |
| M2 | Licença | Os termos dos vídeos, registrados com data, permitem uso em pesquisa, cópia local, extração de features derivadas e publicação de métricas agregadas; termo ausente ou ambíguo é INCONCLUSIVO |
| M3 | Anotações | Os oito CSVs conferem o SHA-256 fixado e o `report` passa (já atendida) |
| M4 | Split | Só o `cs`, com sessões e sujeitos de um lado só; o `cv` não é usado (já atendida pelo CSV) |
| M5 | RGB monocular | Só os vídeos `colors/` entram; profundidade, esqueleto do sensor e acelerômetro, se vierem no pacote, não são lidos |
| M6 | Escopo | Decisão escrita do responsável dizendo o que o trabalho desloca |

**Checagens que só o vídeo RGB resolve**, nos 41 vídeos pré-selecionados e,
para inclusão no piloto, nas vistas candidatas das outras sessões.
Nenhuma etapa corrige tempos: o CSV é usado como publicado, e uma vista
cujos tempos não batem com ele fica registrada como deslocada e deixa de
ser usável.

| # | Checagem | Unidade | GO quando | Fora do GO |
| --- | --- | --- | --- | --- |
| V1 | Integridade e tempo de quadro | Vista | `ffprobe -count_frames` é igual à contagem do pipe de decodificação, como no [relatório do Le2i](../data/temporal-contract.md#relatorio-e-checagens); FPS, resolução e duração ficam registrados | A vista deixa de ser usável. NO-GO do dataset só quando a causa identificada é do formato publicado (contêiner, codec ou carimbos de tempo comuns a todos os vídeos); sem causa identificada, a falha fica na vista |
| V2 | Grade de 10 fps | Vista | O quadro de origem escolhido pela grade fica a menos de um passo da grade (0,1 s) do seu tempo nominal | A vista deixa de ser usável, porque uma grade por PTS mudaria o contrato temporal. NO-GO do dataset só quando a taxa variável vem do formato publicado, comum a todos os vídeos |
| V3 | Fim do vídeo | Vista | A grade usa a duração decodificada e os segmentos que passam do fim ficam listados por vídeo | — (o excesso é registrado, não corrigido) |
| V4 | Deslocamento temporal | Vista | O início do primeiro e do último `fall` da sessão aparece na vista a menos de um passo da grade do tempo do CSV. O deslocamento medido, com sinal, fica registrado por evento e por vista; vistas da mesma sessão podem diferir entre si, e essa diferença é registrada, não exigida igual a zero | [Desfechos](#desfechos-de-v4-v5-e-v8) |
| V5 | Alinhamento da anotação | Vista | Numa vista alinhada em V4, as transições de V4, a sobreposição de `S50`, o maior gap de `S9` e os inícios sem anotação de `S9` e `S24` concordam com o que a vista mostra, a até um passo da grade | [Desfechos](#desfechos-de-v4-v5-e-v8) |
| V6 | Visibilidade da pessoa | Vista | A pessoa aparece na vista durante todos os segmentos inspecionados em V4, V5 e V8 | A vista deixa de ser usável |
| V7 | Cobertura de pose | Vista | O YOLO-Pose congelado, com os mesmos pesos e a mesma [seleção de pessoa](../data/temporal-contract.md#selecao-de-pessoa-na-extracao-de-pose) do Le2i, roda nos 41 vídeos com saída em `data/scratch/cmdfall/`; `person_found` e [`q_pose`](../data/pose-quality.md) ficam registrados por rótulo e vista, ao lado do treino Le2i-CS, e um critério de aceite é fixado por escrito depois dessa comparação | Sem critério escrito, INCONCLUSIVO |
| V8 | Semântica de classe | Sessão | O primeiro e o último segmento de cada rótulo prioritário da sessão (57 segmentos nas seis), julgados na vista escolhida para ela, concordam com a definição do `LABELS.md`; para `other`, só se verifica que não é outra classe da taxonomia, sem identificar a ação original | [Desfechos](#desfechos-de-v4-v5-e-v8) |

Uma vista é **usável** quando passa em V1, V2, V4, V5 e V6 e, depois de
fixado, no critério de V7. Uma vista não inspecionada não é usável, e uma
sessão sem vista usável não entra no piloto. O passo da grade é o único
limiar numérico:
abaixo dele, o contrato de `TARGET_FPS = 10` não distingue os tempos
comparados.

#### Escolha determinística da vista

O piloto usa uma vista por sessão, então só a vista usada precisa estar
alinhada com o CSV. A escolha é fixada uma vez, antes de qualquer treino:

1. Para cada câmera, de K1 a K7, conta-se em quantas das seis sessões
   pré-selecionadas a vista daquela câmera é usável.
2. As câmeras são ordenadas por essa contagem, da maior para a menor; o
   empate vai para o menor número de câmera.
3. Em cada sessão pré-selecionada, a vista escolhida é a da primeira câmera
   da ordem que existe na sessão e é usável nela. Sessão sem vista usável é
   excluída.
4. Nas 21 sessões do `train` do `cs` fora da pré-seleção, vale a mesma
   ordem, e a sessão só entra no piloto depois de verificada. A vista da
   primeira câmera da ordem que existe na sessão passa por V1–V7, com o
   critério de V7 já fixado; em V4 e V5 entram o primeiro e o último `fall`
   da sessão e, quando o CSV os mostra, a maior sobreposição, o maior gap
   interior e o início sem anotação, com a tolerância de 1 ms. Se a vista
   não for usável, passa-se à câmera seguinte da ordem; com uma vista usável,
   V8 é julgado nela. Os desfechos são os mesmos das seis sessões. Sessão
   sem vista usável, ou ainda não verificada, fica fora do piloto.

A verificação do passo 4 decide a inclusão de cada sessão, uma a uma. Ela
não muda a ordem das câmeras, que vem só das seis sessões diagnósticas, e
seus resultados não são lidos como estimativa de taxa no CMDFall.

#### Desfechos de V4, V5 e V8

As seis sessões são diagnósticas: foram escolhidas por casos extremos e não
estimam frequências no CMDFall. Por isso, a repetição de uma anomalia entre
elas não é, sozinha, evidência de falha do dataset; ela abre uma
investigação.

| Desfecho | V4 — deslocamento | V5 — alinhamento | V8 — semântica |
| --- | --- | --- | --- |
| **Exclusão** | A vista deslocada deixa de ser usável; a sessão sai se não restar vista usável | Discordância numa só vista deixa essa vista inutilizável; discordância em todas as vistas alinhadas é erro da anotação da sessão e exclui a sessão | Segmento que contradiz a definição exclui a sessão inteira |
| **INCONCLUSIVO** | Início do `fall` não identificável na vista, ou deslocamento que muda entre eventos da mesma vista (volta a V1 e V2) | Transição ambígua, irregularidade sem explicação visível, ou a mesma discordância em sessões diferentes | Julgamento indeterminado, ou a mesma discordância de um rótulo em sessões diferentes |
| **NO-GO do dataset** | A fonte oficial ou o `KNOWN_PITFALLS.md` estabelece que nenhuma câmera corresponde aos tempos do CSV sem correção | A discordância decorre de uma regra de anotação publicada pelo OmniFall ou pelos autores, que vale para todas as sessões por construção | A definição publicada de um rótulo prioritário, ou a regra documentada que leva o CMDFall a ela, é incompatível com o rótulo do GateFall |

Um INCONCLUSIVO é investigado, nesta ordem, só com material de `train`:

1. a documentação do OmniFall e dos autores do CMDFall, para saber se a
   anomalia é uma regra publicada (NO-GO do dataset) ou não;
2. o resultado da mesma checagem nas 21 sessões do `train` do `cs` fora da
   pré-seleção, obtido na verificação do passo 4 da
   [escolha da vista](#escolha-deterministica-da-vista), nunca em `val` ou
   `test`.

Cada sessão afetada termina excluída ou com a causa documentada. Enquanto
alguma não termina, o INCONCLUSIVO continua aberto. Esta página não fixa
quantas sessões precisam restar: o protocolo do piloto registra as
exclusões e decide, antes do treino e sem métrica de modelo, se o conjunto
restante o sustenta. Toda exclusão é fixada antes de qualquer treino, e a de
sessão vale para todas as vistas.

### Piloto possível: braço A com dados externos

!!! note "Proposta condicional"
    Só pode ser proposto formalmente depois do preflight em GO, por um
    protocolo próprio aprovado antes de qualquer treino. Não é canônico e não
    substitui nenhuma referência.

| | Controle | Tratamento |
| --- | --- | --- |
| Braço | A (TCN sobre as 134 features de pose) | A, sem mudança de arquitetura |
| Treino | `train` do Le2i-CS | `train` do Le2i-CS mais as sessões do `train` do `cs` do CMDFall com vista usável verificada, uma vista por sessão; sessão excluída ou não verificada fica fora |
| Seleção | `val` do Le2i-CS | O mesmo `val`, byte a byte |
| Receita | [Receita congelada](../train/baseline-a.md#receita-de-treino-congelada) | A mesma, com as mesmas seeds |

- **Separação.** Nenhuma janela do CMDFall entra em `val` ou `test` do
  Le2i. Do CMDFall, só o `train` do `cs` é usado; `val` e `test` do `cs`, o
  `cv` e as outras vistas de cada sessão ficam fora. A vista de cada sessão
  vem da [escolha determinística](#escolha-deterministica-da-vista).
- **Contrato.** `TARGET_FPS`, `WINDOW_FRAMES`, `TRAIN_STRIDE`, a taxonomia e
  `RESTRICTED_CLASSES` não mudam. O protocolo fixa antes do treino o que
  acontece com janelas de classes fora de `RESTRICTED_CLASSES`. A
  padronização e os pesos da perda do tratamento são calculados só sobre o
  treino do tratamento.
- **Baseline congelado.** O controle são os runs do
  [sumário multi-seed](multiseed-summary.md). `runs/reference/le2i/baseline_a`
  e as demais referências não são retreinadas nem sobrescritas, e o
  tratamento grava só sob `runs/local/`.
- **Invariante 1.** Só A muda de dados, então o piloto não compara A, B e
  C. Levar CMDFall a B ou C exigiria a mudança nos três braços e um novo
  conjunto de runs.
- **Desloca:** a extração offline de pose das sessões aceitas e os treinos
  do braço A ocupam o lugar de qualquer outro dataset externo, braço,
  backbone ou métrica até a decisão do piloto. B0, B1, C0 e C1 não recebem
  CMDFall.

Proibido, em qualquer etapa:

- usar o `test` oficial do Le2i para escolher modelo, época, seed, limiar,
  exclusão ou critério;
- treinar antes do GO do preflight e da aprovação do protocolo;
- baixar vídeos de acesso restrito sem M1 e M2 em GO, ou de fonte que não
  seja a oficial;
- alterar `runs/reference/`, o registro de datasets, o contrato temporal, a
  taxonomia, `RESTRICTED_CLASSES`, os splits do Le2i ou o protocolo de
  alarme.

## Escopo e o que deslocaria

Esta auditoria não acrescenta configuração, dataset ao registro, backbone
nem métrica, e não altera A, B0, B1, C0 nem C1. Usar o CMDFall seria
acrescentar um dataset, que o `CLAUDE.md` trata como o risco dominante do
projeto. Qualquer próximo passo, a começar pelo pedido de acesso e por uma
sondagem dos vídeos, precisa declarar antes o que desloca no cronograma;
sem esse trade-off, a resposta padrão continua sendo não.

## Conclusão

**Os metadados não bloqueiam o CMDFall, mas também não estabelecem sua
viabilidade.** Pelo CSV, o protocolo `cs` é disjunto por sujeito e por
sessão, mantém as vistas de uma sessão juntas, usa a taxonomia de dez
classes do GateFall e dá aos cinco rótulos prioritários 5 sujeitos em `val`,
19 ou 20 em `test` e 25 em `train`, contra o suporte de um sujeito que
limita `standing` no Le2i-CS. O `cv` é incompatível com o invariante 2. A
viabilidade experimental depende dos itens bloqueados até o acesso aos
vídeos e de uma decisão explícita sobre o que esse trabalho deslocaria. O
[preflight](#preflight-condicionado-ao-acesso) fixa quais gravações
inspecionar e o critério GO / NO-GO, sem antecipar o resultado.
