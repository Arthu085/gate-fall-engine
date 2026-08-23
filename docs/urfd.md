# Dataset URFD

Referência da organização em disco, do schema dos CSVs de labels e do que a
validação encontrou no UR Fall Detection Dataset (URFD), câmera 0. Para os
comandos de download e validação em forma resumida, ver a seção "URFD:
download e validação" no README do repositório.

## Como baixar

```bash
scripts/download_urfd.sh
```

Baixa os 70 vídeos câmera 0 do URFD (30 quedas + 40 ADLs) para
`data/urfd/videos/{fall,adl}/`. É idempotente: reexecutar pula os arquivos já
presentes e válidos (checagem de MIME/assinatura MP4 e rejeição de páginas de
erro HTML), sem nova requisição de rede.

Os CSVs de labels (`urfall-cam0-falls.csv` e `urfall-cam0-adls.csv`) **não são
baixados por este script** — precisam ser obtidos manualmente da fonte
original (<http://fenix.ur.edu.pl/~mkepski/ds/uf.html>) e colocados em
`data/urfd/labels/` antes de rodar a validação.

## Layout em disco

```
data/urfd/
├── videos/
│   ├── fall/
│   │   └── fall-<NN>-cam0.mp4   (30 arquivos, NN = 01..30)
│   └── adl/
│       └── adl-<NN>-cam0.mp4    (40 arquivos, NN = 01..40)
├── labels/
│   ├── urfall-cam0-falls.csv    (sem header)
│   └── urfall-cam0-adls.csv     (sem header)
├── inspect/
│   └── <sequência>/             (frames extraídos para inspeção visual, ver seção abaixo)
└── validation_report.csv        (saída de scripts/validate_urfd.py)
```

`data/` está completamente no `.gitignore` — nada dentro dela é versionado.

## Schema dos CSVs de labels

Cada CSV tem 11 colunas, sem header, uma linha por frame anotado:

| Coluna | Nome | Conteúdo |
| --- | --- | --- |
| A | `sequence` | Nome da sequência (ex.: `fall-01`, `adl-01`) |
| B | `frame_idx` | Índice do frame — **1-based** (confirmado pela validação, ver abaixo) |
| C | `label` | Rótulo do estado (ver "Convenção de rótulo confirmada") |
| D–K | `feat_hw_ratio`, `feat_maj_min_ratio`, `feat_bbox_occ`, `feat_max_std_xz`, `feat_hh_max_ratio`, `feat_h`, `feat_d`, `feat_p40` | 8 descritores derivados do sensor de profundidade Kinect do URFD |

As colunas D–K são **explicitamente não utilizadas**: o pipeline do GateFall é
RGB monocular (invariante do projeto, ver `CLAUDE.md`). `scripts/validate_urfd.py`
carrega essas colunas apenas para registrar presença/exclusão — nunca em
valor.

## Conclusões da validação

Resultado de `uv run scripts/validate_urfd.py` sobre os 70 vídeos + CSVs de
labels já baixados neste repositório.

### Convenção de rótulo confirmada

`-1` = não caído, `0` = pose transitória/caindo, `1` = deitado no chão.

O validador encontrou **16 sequências `adl-*`** com algum valor de `label`
diferente de -1 (esperado, dado que ADL cobre atividades que incluem sentar,
deitar na cama etc. — não é um bug do dataset nem da validação):
`adl-10, adl-11, adl-21, adl-22, adl-23, adl-30, adl-31, adl-32, adl-33,
adl-34, adl-35, adl-36, adl-37, adl-38, adl-39, adl-40`.

### Base de indexação do frame

Confirmada como **1-based**. O método: para cada sequência com vídeo abrível,
compara-se `csv_last_idx` (maior `frame_idx` do CSV) contra a contagem de
frames do MP4 (`cv2.CAP_PROP_FRAME_COUNT`). A hipótese 1-based
(`csv_last_idx == mp4_frame_count`) bateu em 69/70 sequências (98.6%); a
hipótese 0-based (`csv_last_idx == mp4_frame_count - 1`) não bateu em nenhuma.

A única sequência divergente é `adl-17`: `csv_last_idx = 183` contra
`mp4_frame_count = 230` (diferença de 47 frames, não explicável por um mero
deslocamento de base 0/1).

### Natureza das lacunas de frame

Todas as lacunas encontradas estão em sequências `adl-*`; nenhuma sequência
`fall-*` tem frame ausente.

- **18 sequências com lacuna inicial** (frames ausentes antes do primeiro
  `frame_idx` do CSV): tamanho médio ~19 frames, mínimo 4, máximo 44.
- **3 sequências com lacuna interna** (`adl-01`, `adl-06`, `adl-17`): tamanho
  de 1 frame cada.

Para inspecionar visualmente, `scripts/validate_urfd.py` extrai, para cada
sequência com lacuna, os frames `first_missing`, `last_missing` e
`first_present` como JPEGs em `data/urfd/inspect/<sequência>/`. A hipótese em
teste é que as lacunas iniciais em sequências ADL correspondem a trechos em
que ainda não há ninguém em quadro. Uma inspeção visual pontual (`adl-02` e
`adl-40`) não contradiz essa hipótese — os três frames amostrados de cada
sequência mostram o cômodo vazio, sem pessoa claramente visível mesmo no
primeiro frame presente no CSV — mas não é confirmação definitiva para as 18
sequências: a verificação completa exige percorrer o `data/urfd/inspect/` de
cada uma.

### Decisão de alinhamento

Regra de alinhamento que qualquer código a jusante deve usar: o `frame_idx`
do CSV (1-based) corresponde ao frame do MP4 no índice `frame_idx - 1`
(0-based, o que o OpenCV/`cv2.VideoCapture` espera em
`CAP_PROP_POS_FRAMES`).
