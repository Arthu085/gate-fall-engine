# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de
Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso (TCC) em Visão Computacional.
**Autor:** Arthur Ghizi · **Orientador:** Rodrigo Ramos Silva.

## Pergunta de pesquisa

> Em que medida acrescentar informação visual de modelos de fundação congelados
> (DINOv3, SAM 3) a um detector de quedas baseado em pose (YOLO-Pose) aumenta a
> robustez da detecção em vídeo RGB monocular — e como a fusão deve ponderar
> pose e informação visual em função da confiança de cada fonte, nos regimes em
> que a estimativa de pose se degrada (oclusão, truncamento, poses atípicas do
> corpo caído)?

## Configurações experimentais

| Config | Conteúdo de cada linha da janela    | Codificador temporal |
| ------ | ----------------------------------- | -------------------- |
| **A**  | Pose (YOLO-Pose)                    | TCN                  |
| **B**  | Pose + embedding visual (DINOv3)    | TCN                  |
| **C**  | Pose + descritor de máscara (SAM 3) | TCN                  |

**As três configurações são mantidas rigorosamente idênticas em tudo exceto no
conteúdo de cada linha da janela.** Tamanho de janela, split treino/teste, seed,
codificador temporal e número de épocas são os mesmos em A, B e C; a única
variável entre elas é o vetor de features por timestep. Qualquer alteração que
afete apenas uma das configurações invalida a comparação.

## Instalação

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12 (o uv baixa a versão
automaticamente a partir do `.python-version`).

```bash
git clone https://github.com/Arthu085/gate-fall-engine.git
cd gate-fall-engine
uv sync
```

Documentação local:

```bash
uv run mkdocs serve
```

### Docker Compose (alternativa)

Para quem prefere não instalar `uv` e Python diretamente no host, o
repositório inclui um ambiente Docker Compose:

```bash
docker compose run --rm dev
```

O comando builda a imagem (se necessário), sincroniza as dependências via
`uv sync --locked` e abre um shell interativo dentro do container. Por
padrão, o usuário do container usa UID/GID 1000, o padrão do primeiro usuário
em distribuições Linux/WSL — o que já casa com a maioria dos hosts. Se o seu
usuário tiver UID/GID diferente, builde passando os valores explicitamente:

```bash
UID=$(id -u) GID=$(id -g) docker compose build
```

Isso evita que arquivos criados em bind mounts — como `data/` — fiquem com
dono `root`.

Para servir a documentação:

```bash
docker compose up docs
```

Acessível em <http://localhost:8000>, com recarregamento automático a partir
de alterações em `docs/` e `mkdocs.yml`. Use `docker compose up -d docs` para
rodar em segundo plano.

Como o repositório inteiro é montado como bind mount, os datasets baixados em
`data/` (ver seção [Dados](#dados)) ficam visíveis automaticamente dentro do
container, sem necessidade de mount adicional.

## Dados

O diretório `data/` tem três subpastas, nenhuma versionada — todo o
conteúdo de dados é gerado localmente ou baixado, nunca commitado:

| Pasta            | Conteúdo                                   | Versionado? |
| ---------------- | ------------------------------------------- | ----------- |
| `data/raw/`      | Vídeos originais dos datasets (ex.: Le2i)   | Não         |
| `data/labels/`   | CSVs de anotação (pequenos, texto puro)     | Não         |
| `data/features/` | Features pré-computadas pelos backbones     | Não         |

### Anotações (`data/labels/`)

As anotações do Le2i são obtidas a partir do dataset agregado
[OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall) no
HuggingFace, que já resolve a correspondência entre vídeos, splits e rótulos
de queda entre os datasets originais. Para baixá-las:

```bash
uv run python scripts/fetch_labels.py
```

O script grava em `data/labels/omnifall/`:

- `train.csv`, `val.csv` e `test.csv` — split oficial `le2i-cs` do OmniFall
  (670 / 94 / 203 vídeos, cerca de 967 no total), com colunas
  `path, label, start, end, subject, cam, dataset`.
- `le2i.csv` — mesmas 967 linhas, extraídas do config `labels` (pool de todos
  os datasets de origem) e filtradas para `dataset == "le2i"`; útil para
  conferência cruzada com os splits acima.
- `PROVENANCE.json` — revisão do dataset fixada, checksums SHA-256 e
  contagem de linhas de cada CSV baixado, usado por `--verify`.

O download é idempotente por arquivo: rodar o comando de novo pula os CSVs já
existentes. Use `--force` para re-baixar e sobrescrever:

```bash
uv run python scripts/fetch_labels.py --force
```

Para conferir a integridade dos arquivos já baixados contra o
`PROVENANCE.json` gravado (sem baixar nada de novo):

```bash
uv run python scripts/fetch_labels.py --verify
```

As anotações do OmniFall são distribuídas sob a licença
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) —
uso não comercial, com atribuição e compartilhamento pela mesma licença — por
isso os CSVs não são versionados neste repositório, cujo código é MIT. Para a
revisão fixada do dataset, checksums e citações completas, ver
`docs/data-provenance.md`.

### Vídeos brutos (`data/raw/`)

O script acima baixa **apenas as anotações**, não os vídeos. Os vídeos do
Le2i devem ser obtidos manualmente na fonte original e colocados em
`data/raw/le2i/`, seguindo os caminhos referenciados pela coluna `path` dos
CSVs de anotação.

Antes de trabalhar com vídeo (extração de frames, features, etc.), confirme
que `ffprobe` e `ffmpeg` estão instalados:

```bash
ffprobe -version && ffmpeg -version
```

Caso não estejam:

```bash
sudo apt install ffmpeg   # Debian/Ubuntu
brew install ffmpeg       # macOS
```

## Licença

O **código** deste repositório é distribuído sob a licença [MIT](LICENSE).
