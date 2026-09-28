# Preparação do Le2i

Os vídeos do Le2i são obtidos manualmente na fonte original. O GateFall não
automatiza o download e não distribui esses arquivos; o script do projeto apenas
extrai o pacote já obtido.

## Fontes oficiais

- Dataset: [Fall Detection Dataset — Université de
  Franche-Comté](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)
  ([DOI `10.25666/DATAUBFC-2024-04-09`](https://doi.org/10.25666/DATAUBFC-2024-04-09)).

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

Diretórios já extraídos são preservados. Se `FallDataset.zip` não for encontrado
mas os seis diretórios esperados (`Coffee_room_01`, `Coffee_room_02`, `Home_01`,
`Home_02`, `Lecture room`, `Office`) já estiverem extraídos em `data/raw/le2i/`,
o script imprime um aviso e encerra sem erro; nesse caso o ZIP não é necessário.
Sem `--force`, a ausência do arquivo só é um erro quando a extração está
incompleta.

Para removê-los e extraí-los de novo:

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

| Path na anotação         | Arquivo extraído                      |
| ------------------------ | ------------------------------------- |
| `Coffee_room_01/video_1` | `Coffee_room_01/Videos/video (1).avi` |
| `Lecture_room/video_1`   | `Lecture room/video (1).avi`          |
| `Office/video_1`         | `Office/video (1).avi`                |

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

O adapter Le2i resolve a identidade portátil `relative_path` contra
`data/raw/le2i/` e rejeita caminhos absolutos, componentes `..` ou qualquer
escape da raiz. O manifesto e a grade gerados ficam em
`data/processed/le2i/`; consulte a [organização dos dados](organization.md)
para a migração dos caminhos legados.

## Extração remota de features

Depois de construir `manifest.parquet` e `frames.parquet` no checkout local,
prepare um pacote portátil do protocolo Le2i CS na raiz do repositório:

```bash
uv run python -m gatefall.data.le2i.bundle prepare --output /caminho/le2i-cs-bundle
uv run python -m gatefall.data.le2i.bundle verify --bundle /caminho/le2i-cs-bundle
```

O pacote contém `data/processed/le2i/manifest.parquet` e
`data/processed/le2i/frames.parquet` byte-idênticos aos originais, além dos
vídeos referenciados pelo `relative_path` do manifesto sob `data/raw/le2i/`.
`bundle-sha256.json` guarda apenas os hashes dos dois parquets para a
verificação no destino. A preparação recusa um destino existente e publica o
diretório somente depois de copiar e verificar todos os arquivos. Não inclui
labels, repositórios de terceiros, pesos ou checkpoints.

Transfira o conteúdo do pacote para a raiz de um checkout remoto do mesmo
commit e verifique ali antes de extrair:

```bash
rsync -a /caminho/le2i-cs-bundle/ usuario@host:/caminho/gate-fall-engine/
ssh usuario@host 'cd /caminho/gate-fall-engine && uv run python -m gatefall.data.le2i.bundle verify --bundle .'
```

No checkout remoto, instale as dependências com `uv sync --locked`. Para
DINOv3, disponibilize separadamente o repositório DINOv3 e seus pesos; para
SAM 3, sincronize o runtime isolado com `uv sync --project sam3_runtime --locked`
e disponibilize o checkpoint separadamente. Então execute os extratores já
existentes, conforme o modelo desejado:

```bash
uv run python -m gatefall.dinov3.extract extract-all --repo-dir /caminho/dinov3 --weights /caminho/pesos-dinov3 --dataset le2i
uv run python -m gatefall.dinov3.extract report --dataset le2i
uv run python -m gatefall.sam3.extract extract-all --runtime-dir sam3_runtime --checkpoint /caminho/sam3.pt --dataset le2i
uv run python -m gatefall.sam3.extract report --dataset le2i
```

Copie de volta apenas as árvores produzidas pelos extratores executados:

```bash
mkdir -p data/features/le2i
rsync -a usuario@host:/caminho/gate-fall-engine/data/features/le2i/dinov3/ data/features/le2i/dinov3/
rsync -a usuario@host:/caminho/gate-fall-engine/data/features/le2i/sam3/ data/features/le2i/sam3/
```

Esses comandos devem rodar na raiz do checkout local. Use somente a linha de
`rsync` correspondente ao extrator executado. Os caminhos de saída existentes
são `data/features/le2i/dinov3/` e `data/features/le2i/sam3/`.

## Exploração histórica

O script `scripts/exploratory/explore_le2i.py` preserva as análises usadas para
entender a distribuição original e pode ser executado com:

```bash
uv run python scripts/exploratory/explore_le2i.py
```

Ele não é uma etapa obrigatória da preparação e nenhum módulo de produção
depende desse script.
