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
vídeos e de uma decisão explícita sobre o que esse trabalho deslocaria.
