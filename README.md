# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso em Visão Computacional de **Arthur Ghizi**, orientado por **Rodrigo Ramos Silva**.

O **GateFall** investiga a detecção de quedas humanas em vídeo RGB monocular utilizando backbones congelados e features pré-computadas.

O projeto é organizado em três braços experimentais:

- **Braço A — YOLO-Pose + TCN:** implementado.
- **Braço B — YOLO-Pose + DINOv3 + TCN:** planejado.
- **Braço C — YOLO-Pose + SAM 3 + TCN:** planejado.

Os três braços compartilham a mesma base de dados, protocolo temporal e estrutura de avaliação. A principal diferença entre eles é o **vetor de features produzido para cada timestep**.

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

A preparação dos dados constitui a base compartilhada pelos três braços experimentais do GateFall.

Ela é responsável por organizar e validar os datasets, construir a grade temporal utilizada pelo projeto e produzir os artefatos comuns que serão consumidos posteriormente pelos pipelines dos braços A, B e C.

Os datasets utilizados pelo GateFall possuem licenças próprias e não são distribuídos diretamente pelo repositório.

Atualmente, o dataset utilizado nos experimentos é o **Le2i Fall Detection Dataset**.

A documentação completa da preparação dos dados descreve:

- organização dos diretórios;
- obtenção e posicionamento dos dados brutos;
- ingestão do dataset;
- validação de manifesto e vídeos;
- geração da grade temporal;
- convenções de splits e labels;
- localização dos artefatos processados;
- migração dos caminhos legados;
- comandos de verificação e diagnóstico.

Consulte:

- [Instalação do Dataset Le2i](docs/data/le2i.md) — **etapa obrigatória antes de executar as pipelines**. Siga as instruções desta página para obter e posicionar corretamente o dataset.
- [Organização e preparação dos dados](docs/data/organization.md) — descreve a estrutura dos dados e os artefatos processados utilizados pelo projeto.
- [Referência de comandos de dados](docs/reference/commands.md) — reúne os comandos disponíveis para executar ou inspecionar manualmente cada etapa.

> Após a instalação do dataset Le2i, as pipelines executam automaticamente as etapas de preparação e processamento necessárias para seus respectivos experimentos.

A preparação deve ser concluída antes da execução dos experimentos que dependem desses artefatos.

---

## Pipelines experimentais

Após a preparação dos dados, cada braço experimental possui seu próprio pipeline de extração de features, treinamento e avaliação.

Os braços compartilham o mesmo protocolo temporal e a mesma base experimental, mas utilizam representações distintas por timestep.

### Braço A — YOLO-Pose + TCN

O braço A utiliza features cinemáticas derivadas das poses estimadas pelo YOLO-Pose e um classificador temporal TCN.

Para executar o pipeline completo:

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm A
```

O pipeline executa e valida as etapas necessárias para reproduzir o braço A, incluindo geração dos artefatos compartilhados quando necessário, extração de pose e features cinemáticas, padronização, treinamento do TCN e avaliação.

Documentação:

- [Runbook completo do pipeline A](docs/runbooks/pipeline-a.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos do braço A](docs/eval/baseline-a-events.md)
- [Referência de comandos](docs/reference/commands.md)

### Braço B — YOLO-Pose + DINOv3 + TCN

O braço B combinará as informações de pose utilizadas pelo baseline com features visuais extraídas pelo **DINOv3**, mantendo o protocolo temporal e o classificador TCN compatíveis com o braço A.

O pipeline do braço B ainda está em desenvolvimento.

Quando implementado, esta seção concentrará:

- comando principal de execução;
- requisitos específicos do DINOv3;
- preparação e armazenamento das features visuais;
- composição do vetor de features por timestep;
- treinamento;
- avaliação;
- artefatos produzidos;
- runbook específico do braço B.

Documentação prevista:

- Runbook do pipeline B
- Extração de features DINOv3
- Treino do braço B
- Avaliação do braço B

### Braço C — YOLO-Pose + SAM 3 + TCN

O braço C utilizará informações de pose combinadas com representações derivadas do **SAM 3**, preservando o mesmo protocolo temporal e estrutura de classificação utilizados nos demais braços.

O pipeline do braço C ainda está em desenvolvimento.

Quando implementado, esta seção concentrará:

- comando principal de execução;
- requisitos específicos do SAM 3;
- geração e armazenamento das representações utilizadas;
- composição do vetor de features por timestep;
- treinamento;
- avaliação;
- artefatos produzidos;
- runbook específico do braço C.

Documentação prevista:

- Runbook do pipeline C
- Extração de features SAM 3
- Treino do braço C
- Avaliação do braço C

---

## Validação do ambiente

O projeto possui selftests sintéticos que podem ser executados sem o dataset real.

Para validar o orquestrador dos pipelines:

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

A documentação detalhada está organizada por responsabilidade.

### Arquitetura

- [Visão geral da arquitetura](docs/architecture/overview.md)
- [Tecnologias](docs/architecture/technology-stack.md)

### Dados

- [Organização e preparação dos dados](docs/data/organization.md)
- [Referência de comandos](docs/reference/commands.md)

### Braço A

- [Runbook completo do pipeline A](docs/runbooks/pipeline-a.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos](docs/eval/baseline-a-events.md)

### Braço B

Documentação específica será adicionada durante a implementação do braço B.

### Braço C

Documentação específica será adicionada durante a implementação do braço C.

Para abrir a documentação localmente:

```bash
uv run mkdocs serve
```

Depois, acesse o endereço exibido pelo MkDocs no terminal.

---

## Licença

O **código do GateFall** é distribuído sob a licença [MIT](LICENSE).

Datasets, pesos pré-treinados e dependências de modelos possuem seus próprios termos e licenças. A licença MIT deste repositório **não se estende automaticamente a esses materiais externos**.
