# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso em Visão Computacional de **Arthur Ghizi**, orientado por **Rodrigo Ramos Silva**.

O **GateFall** investiga a detecção de quedas humanas em vídeo RGB monocular utilizando backbones congelados e features pré-computadas.

O projeto é organizado em três braços experimentais:

- **Braço A — YOLO-Pose + TCN:** implementado.
- **Braço B — YOLO-Pose + DINOv3 + TCN:** planejado.
- **Braço C — YOLO-Pose + SAM 3 + TCN:** planejado.

A arquitetura foi projetada para manter o pipeline temporal e o classificador consistentes entre os braços. A principal diferença entre eles é o **vetor de features produzido para cada timestep**.

---

## Requisitos

O GateFall foi desenvolvido para **Linux ou WSL**.

Para executar o projeto são necessários:

- Git
- `curl`
- FFmpeg / ffprobe
- [uv](https://docs.astral.sh/uv/)
- Python 3.12 ou superior

O `uv` é utilizado para gerenciar a versão do Python, o ambiente virtual e todas as dependências do projeto.

### 1. Instalar dependências do sistema

Em Ubuntu ou WSL com Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl ffmpeg
```

Verifique a instalação:

```bash
git --version
ffmpeg -version
ffprobe -version
```

### 2. Instalar o uv

Instale o `uv` utilizando o instalador oficial:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reabra o terminal ou recarregue a configuração do shell:

```bash
source ~/.bashrc
```

Verifique:

```bash
uv --version
```

### 3. Instalar o Python

O próprio `uv` pode instalar e gerenciar a versão do Python utilizada pelo projeto:

```bash
uv python install 3.12
```

Verifique as versões disponíveis:

```bash
uv python list
```

Não é necessário criar manualmente um ambiente virtual com `venv`. O `uv` cuida dessa etapa durante a sincronização do projeto.

---

## Instalação do GateFall

### 1. Clonar o repositório

```bash
git clone https://github.com/Arthu085/gate-fall-engine.git
cd gate-fall-engine
```

### 2. Instalar as dependências

Sincronize o ambiente a partir da configuração e do lockfile do projeto:

```bash
uv sync --python 3.12
```

Esse comando cria o ambiente virtual do projeto e instala as dependências necessárias.

Para confirmar que o ambiente está funcionando:

```bash
uv run python --version
```

---

## Preparação dos dados

Os datasets utilizados pelo GateFall possuem licenças próprias e não são distribuídos diretamente pelo repositório.

Para reproduzir o **braço A**, prepare o dataset **Le2i Fall Detection Dataset** conforme as instruções completas da documentação:

- [Organização e preparação dos dados](docs/data/organization.md)

---

## Executando o pipeline

Após instalar as dependências e preparar os dados, execute o pipeline completo do braço A a partir da raiz do repositório:

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm A
```

O comando executa automaticamente as etapas de preparação, validação, processamento, extração de features, treinamento e avaliação.

Para detalhes sobre pré-requisitos, etapas executadas, artefatos gerados, `--dry-run`, `--force` e diagnóstico de falhas, consulte:

- [Runbook completo do pipeline A](docs/runbooks/pipeline-a.md)
- [Referência de comandos](docs/reference/commands.md)

---

## Validação do ambiente

O projeto possui selftests sintéticos que podem ser executados sem o dataset real.

Para validar o orquestrador do pipeline:

```bash
uv run python -m gatefall.pipeline selftest
```

Para executar as verificações de desenvolvimento:

```bash
uv run pyright
uv run mkdocs build --strict
```

A CI do projeto também executa os selftests sintéticos, Pyright e a validação da documentação sem exigir dataset real ou GPU.

---

## Documentação

A documentação detalhada do projeto está disponível em:

- [Arquitetura](docs/architecture/overview.md)
- [Tecnologias](docs/architecture/technology-stack.md)
- [Organização e preparação dos dados](docs/data/organization.md)
- [Referência de comandos](docs/reference/commands.md)
- [Runbook completo do pipeline A](docs/runbooks/pipeline-a.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos](docs/eval/baseline-a-events.md)

Para abrir a documentação localmente:

```bash
uv run mkdocs serve
```

Depois, acesse o endereço exibido pelo MkDocs no terminal.

---

## Licença

O **código do GateFall** é distribuído sob a licença [MIT](LICENSE).

Datasets, pesos pré-treinados e dependências de modelos possuem seus próprios termos e licenças. A licença MIT deste repositório **não se estende automaticamente a esses materiais externos**.
