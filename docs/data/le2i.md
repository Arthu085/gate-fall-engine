# Preparação do Le2i

Os vídeos do Le2i são obtidos manualmente na fonte original. O GateFall não
automatiza o download e não distribui esses arquivos; o script do projeto apenas
extrai o pacote já obtido.

## Fontes oficiais

- Dataset: [Fall Detection Dataset — Université de
  Franche-Comté](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)
  ([DOI `10.25666/DATAUBFC-2024-04-09`](https://doi.org/10.25666/DATAUBFC-2024-04-09)).
- Artigo original: [*Optimized spatio-temporal descriptors for real-time fall
  detection: comparison of support vector machine and Adaboost-based
  classification*](https://doi.org/10.1117/1.JEI.22.4.041106), de Charfi et al.

Consulte a página oficial e os arquivos que acompanham a distribuição para os
termos aplicáveis aos vídeos.

## Download e extração

1. Obtenha `FallDataset.zip` na página oficial.
2. Coloque o arquivo no caminho padrão `data/raw/le2i/FallDataset.zip`.
3. Extraia os arquivos preservando a estrutura original:

```bash
uv run python scripts/extract_le2i.py
```

Para usar outro caminho:

```bash
uv run python scripts/extract_le2i.py --zip PATH
```

O pacote externo contém arquivos ZIP por ambiente. O extrator abre cada pacote
aninhado, mantém seus diretórios e copia o `README.txt` da distribuição quando
presente. Antes da extração, imprime o SHA-256 de `FallDataset.zip`; ao final,
mostra a árvore resumida e o tamanho total.

Diretórios já extraídos são preservados. Para removê-los e extraí-los de novo:

```bash
uv run python scripts/extract_le2i.py --force
```

`--force` atua apenas nos diretórios identificados dentro dos pacotes aninhados
e no `README.txt`; não baixa novamente o arquivo externo.

## Ferramentas de vídeo

A extração do ZIP usa apenas a biblioteca padrão do Python. A ingestão posterior
requer `ffprobe`, distribuído com o FFmpeg, para ler metadados e contar quadros.
Confirme a instalação antes de construir o manifesto:

```bash
ffprobe -version
ffmpeg -version
```

Em Debian/Ubuntu, o pacote pode ser instalado com `sudo apt install ffmpeg`. No
macOS com Homebrew, use `brew install ffmpeg`.

## Layout e correspondência de caminhos

A estrutura extraída não coincide literalmente com os paths publicados pelo
OmniFall:

| Path na anotação | Arquivo extraído |
| --- | --- |
| `Coffee_room_01/video_1` | `Coffee_room_01/Videos/video (1).avi` |
| `Lecture_room/video_1` | `Lecture room/video (1).avi` |
| `Office/video_1` | `Office/video (1).avi` |

O casamento percorre recursivamente os arquivos `.avi` e normaliza somente as
diferenças conhecidas da distribuição:

- desconsidera componentes de diretório chamados `Videos`;
- converte `video (N).avi` em `video_N`;
- trata espaços e underscores como equivalentes no nome do ambiente;
- compara os nomes em minúsculas.

Se dois paths distintos gerarem a mesma chave normalizada, o processo falha em
vez de escolher um deles. Depois da normalização, a ingestão exige uma bijeção
entre todos os vídeos locais e todos os paths anotados. No snapshot atual, são
190 vídeos e 190 paths únicos.

## Exploração histórica

O script `scripts/exploratory/explore_le2i.py` preserva as análises usadas para
entender a distribuição original e pode ser executado com:

```bash
uv run python scripts/exploratory/explore_le2i.py
```

Ele não é uma etapa obrigatória da preparação e nenhum módulo de produção
depende desse script.
