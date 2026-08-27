# Organização dos dados

O pipeline separa fontes originais, anotações, features derivadas e o índice
que relaciona esses artefatos. Todo o conteúdo de `data/` é obtido ou gerado
localmente e não é versionado, com exceção dos arquivos `.gitkeep` que preservam
os diretórios vazios.

## Layout local

```text
data/
├── raw/
│   └── le2i/
│       ├── FallDataset.zip
│       └── <ambientes extraídos>/
├── labels/
│   ├── omnifall/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── le2i.csv
│   │   └── PROVENANCE.json
│   └── le2i/
│       └── frames.parquet
├── features/
└── manifest.parquet
```

| Artefato | Responsabilidade | Versionado? |
| --- | --- | --- |
| `data/raw/` | Distribuições e vídeos originais dos datasets | Não |
| `data/labels/` | Anotações obtidas de fontes externas, sua proveniência, e artefatos tabulares derivados delas (ex.: `le2i/frames.parquet`, ver [Contrato temporal](temporal-contract.md)) | Não |
| `data/features/` | Features pré-computadas pelos backbones | Não |
| `data/manifest.parquet` | Relação entre vídeo, anotação, metadados e estado das features | Não |

Os arquivos não são incorporados ao repositório porque pertencem a fontes com
termos próprios ou podem ser reconstruídos. O código do GateFall continua sob a
licença indicada no repositório; os termos de cada dataset devem ser consultados
em sua fonte.

## Fluxo de preparação

1. Baixe as anotações fixadas do OmniFall com
   `uv run python scripts/fetch_labels.py`.
2. Obtenha manualmente o pacote do Le2i e extraia-o com
   `uv run python scripts/extract_le2i.py`.
3. Relacione os vídeos e as anotações e gere o manifesto com
   `uv run python -m gatefall.data.ingest ingest`.
4. Verifique os arquivos e as propriedades do conjunto com
   `uv run python -m gatefall.data.ingest verify`.
5. Audite a cobertura dos segmentos anotados sobre a duração dos vídeos com
   `uv run python -m gatefall.data.coverage audit`.
6. Use o manifesto como índice para as etapas posteriores de features. Essas
   etapas ainda não estão implementadas.

Consulte [OmniFall](omnifall.md), [Le2i](le2i.md) e [Manifesto e
verificação](manifest-verification.md) antes de executar o fluxo completo.

## Responsabilidades no código

Os módulos genéricos não conhecem detalhes do Le2i:

- `gatefall.data.video_metadata` executa o `ffprobe`, interpreta metadados e
  resolve a taxa de quadros;
- `gatefall.data.manifest` define o schema tabular e a persistência atômica do
  manifesto Parquet;
- `gatefall.data.omnifall.annotations` carrega configurações do OmniFall e
  persiste CSVs;
- `gatefall.data.omnifall.provenance` grava e verifica o `PROVENANCE.json`.

As decisões específicas do Le2i ficam em `gatefall.data.le2i`:

- `annotations` escolhe a revisão e as configurações do OmniFall, prepara os
  CSVs e indexa as anotações por vídeo;
- `archive` extrai a distribuição original;
- `path_matching` normaliza e casa os caminhos do OmniFall com os vídeos
  extraídos;
- `manifest` constrói o manifesto do Le2i usando os componentes genéricos;
- `verification` reúne as verificações críticas e os relatórios informativos;
- `coverage` audita o quanto os segmentos anotados cobrem a duração de cada
  vídeo, a partir do manifesto e das anotações já preparados.

Os arquivos `scripts/fetch_labels.py`, `scripts/extract_le2i.py`,
`gatefall.data.ingest` e `gatefall.data.coverage` são apenas pontos de
entrada. A análise histórica fica em `scripts/exploratory/explore_le2i.py`;
ela pode consumir módulos de produção, mas nenhuma etapa de produção depende
dela.
