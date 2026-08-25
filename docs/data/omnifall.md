# Anotações e proveniência do OmniFall

O GateFall usa anotações do Le2i publicadas pelo OmniFall. Os CSVs são obtidos
do snapshot fixado no Hugging Face; os vídeos originais não são fornecidos por
esse comando.

## Fontes oficiais

- Artigo: [*OmniFall: From Staged Through Synthetic to Wild, A Unified
  Multi-Domain Dataset for Robust Fall Detection*](https://arxiv.org/abs/2505.19889).
- Projeto: [simplexsigil.github.io/omnifall](https://simplexsigil.github.io/omnifall/).
- Dataset: [`simplexsigil2/omnifall`](https://huggingface.co/datasets/simplexsigil2/omnifall).

## Snapshot consumido

O código fixa uma revisão em vez de acompanhar a branch móvel do dataset:

| Campo | Valor |
| --- | --- |
| Repositório | `simplexsigil2/omnifall` |
| Revisão | `68e5cee56a4bad38cca4aea791cac248f96e79a0` |
| Configurações | `le2i-cs`, `labels` |

A escolha da revisão e das configurações é específica da integração Le2i e
fica em `gatefall.data.le2i.annotations`. O carregamento de uma configuração,
a escrita de CSVs e a proveniência são reutilizáveis e ficam em
`gatefall.data.omnifall`.

## Download e integridade

Baixe as anotações:

```bash
uv run python scripts/fetch_labels.py
```

Arquivos já existentes são preservados. Para sobrescrevê-los:

```bash
uv run python scripts/fetch_labels.py --force
```

O comando grava os quatro CSVs e o
`data/labels/omnifall/PROVENANCE.json`. Esse JSON registra o repositório, a
revisão, as configurações, o instante UTC da preparação e, para cada CSV, nome,
número de linhas e SHA-256.

Para verificar os arquivos presentes sem acessar novamente a fonte:

```bash
uv run python scripts/fetch_labels.py --verify
```

A verificação exige o `PROVENANCE.json`, confirma a presença dos arquivos
listados e compara seus hashes. Arquivos ausentes ou hashes divergentes fazem o
comando terminar com erro.

## Arquivos e contagens

Cada linha dos CSVs representa um **segmento anotado**, não um vídeo. Um mesmo
path pode aparecer em mais de um segmento.

| Arquivo | Origem | Segmentos | Paths de vídeo únicos |
| --- | --- | ---: | ---: |
| `train.csv` | `le2i-cs`, split `train` | 670 | 133 |
| `val.csv` | `le2i-cs`, split `validation` | 94 | 19 |
| `test.csv` | `le2i-cs`, split `test` | 203 | 38 |
| **Total dos splits** |  | **967** | **190** |
| `le2i.csv` | `labels`, filtrado por `dataset == "le2i"` | 967 | 190 |

O `le2i.csv` permite conferir a união das anotações do Le2i; ele não constitui
um quarto split. O nome upstream `le2i-cs` é mantido como identificador da
configuração, mas não é usado como prova de disjunção por sujeito. O comando de
verificação do manifesto relata essa propriedade separadamente e de forma
informativa.

## Termos publicados pelo upstream

O card do snapshot fixado contém uma divergência interna: o frontmatter declara
`cc-by-nc-4.0`, enquanto o badge e a seção *License* indicam `CC BY-NC-SA 4.0`.
Consulte o [dataset na revisão fixada](https://huggingface.co/datasets/simplexsigil2/omnifall/tree/68e5cee56a4bad38cca4aea791cac248f96e79a0)
e as fontes do projeto para determinar os termos aplicáveis ao uso e à
redistribuição das anotações. Essa divergência é também uma razão para manter os
CSVs fora do repositório, sem tentar reinterpretar seus termos como os da licença
do código.
