# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso em Visão Computacional de **Arthur Ghizi**, orientado por **Rodrigo Ramos Silva**.

O **GateFall** investiga a detecção de quedas humanas em vídeo RGB monocular utilizando backbones congelados e features pré-computadas.

O projeto é organizado em três braços experimentais:

- **Braço A — YOLO-Pose + TCN:** implementado.
- **Braço B — YOLO-Pose + DINOv3 + TCN:** extração offline de features
  DINOv3, o braço B0 (fusão por concatenação simples com pose) e o braço B1
  (fusão adaptativa por gate escalar) implementados.
- **Braço C — YOLO-Pose + SAM 3 + TCN:** fundação de extração offline do
  descritor de máscara `V_t` do SAM 3, braços C0 (fusão por concatenação simples)
  e C1 (gate adaptativo), e avaliação por eventos e alarmes de ambos implementadas.

Este projeto é **Built with DINOv3**.

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

Execute o pipeline completo do braço desejado com `--dataset le2i`:

```bash
uv run python -m gatefall.pipeline run --dataset le2i --arm A
uv run python -m gatefall.pipeline run --dataset le2i --arm B0
uv run python -m gatefall.pipeline run --dataset le2i --arm B1
uv run python -m gatefall.pipeline run --dataset le2i --arm C0
uv run python -m gatefall.pipeline run --dataset le2i --arm C1
```

Somente o braço A aceita `--dataset le2i-cv`. Consulte o [runbook](docs/runbooks/pipelines.md)
e a [referência de comandos](docs/reference/commands.md) para pré-requisitos e detalhes.

### Braço A — YOLO-Pose + TCN

O braço A utiliza features cinemáticas derivadas das poses estimadas pelo YOLO-Pose e um classificador temporal TCN.

O pipeline executa e valida as etapas necessárias para reproduzir o braço A, incluindo geração dos artefatos compartilhados quando necessário, extração de pose e features cinemáticas, padronização, treinamento do TCN e avaliação.

Documentação:

- [Runbook dos pipelines experimentais](docs/runbooks/pipelines.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos do braço A](docs/eval/baseline-a-events.md)
- [Referência de comandos](docs/reference/commands.md)

### Braço B — YOLO-Pose + DINOv3 + TCN

O braço B combina as informações de pose utilizadas pelo baseline com features visuais extraídas pelo **DINOv3**, mantendo o protocolo temporal e o classificador TCN compatíveis com o braço A.

A extração offline de features DINOv3 está implementada: veja
[Features DINOv3](docs/data/dinov3-features.md) para o schema do HDF5
produzido, a fórmula do descritor e os termos de licença do DINOv3.

O braço B0 (fusão por concatenação simples entre pose e DINOv3 projetados,
seguida da mesma TCN do braço A) também está implementado: veja
[Padronização de features DINOv3](docs/data/dinov3-standardization.md) e
[Treino — Braço B0](docs/train/baseline-b0.md) para a arquitetura, a receita de
treino compartilhada com o braço A e como executar.

O braço B1 acrescenta uma fusão adaptativa: um gate escalar por timestep,
calculado a partir dos proxies de qualidade `q_pose` e `q_visual`, pondera as
duas fontes antes da TCN. Veja [Features de
qualidade](docs/data/quality-features.md) e [Treino — Braço
B1](docs/train/baseline-b1.md).

### Braço C — YOLO-Pose + SAM 3 + TCN

O braço C combina informações de pose com representações derivadas do
**SAM 3**, preservando o mesmo protocolo temporal e estrutura de
classificação utilizados nos demais braços.

A fundação de extração offline do descritor de máscara `V_t` está
implementada: veja [Fundação SAM 3](docs/data/sam3-foundation.md) para o
schema do HDF5 produzido, a fórmula do descritor, a política de seleção de
instância e o isolamento de ambiente do runtime SAM 3.

O braço C0 (fusão por concatenação simples entre pose e `V_t` projetados,
seguida da mesma TCN do braço A) também está implementado: veja
[Padronização do descritor SAM 3](docs/data/sam3-standardization.md) e
[Treino — Braço C0](docs/train/baseline-c0.md). O braço C1 também está implementado:
veja [Treino — Braço C1](docs/train/baseline-c1.md). A avaliação por
eventos e alarmes de C0 e C1 também está implementada e usa o protocolo de alarme
congelado do braço A: veja [Avaliação por eventos de C0](docs/eval/baseline-c0-events.md)
e [Avaliação por eventos de C1](docs/eval/baseline-c1-events.md).

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
As [referências finais do Le2i](docs/reference/le2i-runs.md) reúnem os
resultados textuais dos braços A, B0, B1, C0 e C1.

### Arquitetura

- [Visão geral da arquitetura](docs/architecture/overview.md)
- [Tecnologias](docs/architecture/technology-stack.md)

### Dados

- [Organização e preparação dos dados](docs/data/organization.md)
- [Referência de comandos](docs/reference/commands.md)

### Braço A

- [Runbook dos pipelines experimentais](docs/runbooks/pipelines.md)
- [Treino do braço A](docs/train/baseline-a.md)
- [Avaliação por eventos](docs/eval/baseline-a-events.md)

### Braço B

- [Features DINOv3](docs/data/dinov3-features.md)
- [Padronização de features DINOv3](docs/data/dinov3-standardization.md)
- [Treino do braço B0 (fusão pose + DINOv3)](docs/train/baseline-b0.md)
- [Avaliação por eventos do braço B0](docs/eval/baseline-b0-events.md)
- [Features de qualidade (`q_pose`, `q_visual`)](docs/data/quality-features.md)
- [Treino do braço B1 (fusão adaptativa por gate)](docs/train/baseline-b1.md)
- [Avaliação por eventos do braço B1](docs/eval/baseline-b1-events.md)

### Braço C

- [Fundação SAM 3](docs/data/sam3-foundation.md)
- [Padronização do descritor SAM 3](docs/data/sam3-standardization.md)
- [Treino do braço C0 (fusão pose + SAM 3)](docs/train/baseline-c0.md)
- [Avaliação por eventos de C0](docs/eval/baseline-c0-events.md)
- [Treino do braço C1 (gate adaptativo com SAM 3)](docs/train/baseline-c1.md)
- [Avaliação por eventos de C1](docs/eval/baseline-c1-events.md)

Para abrir a documentação localmente:

```bash
uv run mkdocs serve
```

Depois, acesse o endereço exibido pelo MkDocs no terminal.

---

## Licença

O **código do GateFall** é distribuído sob a licença [MIT](LICENSE).

Datasets, pesos pré-treinados e dependências de modelos possuem seus próprios termos e licenças. A licença MIT deste repositório **não se estende automaticamente a esses materiais externos**.

Em particular, o código e os pesos do **DINOv3** são distribuídos sob a
licença própria do DINOv3 (não MIT, não Apache-2.0) e não são redistribuídos
por este repositório — veja [Features DINOv3](docs/data/dinov3-features.md#licenca-do-dinov3)
para os termos completos.
