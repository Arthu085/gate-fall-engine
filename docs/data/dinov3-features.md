# Features DINOv3

`src/gatefall/dinov3/` implementa a extração offline de features visuais do
braço B a partir do backbone congelado **DINOv3**. A CLI fina
`src/gatefall/dinov3/extract.py` (`extract`, `extract-all`, `report`, `audit`,
`verify-determinism`, `verify-frame-alignment`, `selftest`) opera sobre a
mesma grade temporal
(`frames.parquet`) e o mesmo manifesto usados pelo braço A — nenhuma janela
ou split é recalculado aqui. Fusão com pose, treino e avaliação do braço B
ainda não estão implementados.

## Backbone

O modelo é o `dinov3_vitb16` (ViT-B/16, pré-treinado em LVD-1689M), carregado
congelado via `torch.hub.load(..., source="local")` a partir de um clone
local do repositório de referência do DINOv3 e de um arquivo de pesos local
— nunca baixado em tempo de execução. Nenhum fine-tuning acontece; o backbone
só roda em modo de inferência (`eval()`, `torch.inference_mode()`).

Caminhos padrão:

- Repositório: `data/scratch/dinov3_repo`
- Pesos: `data/scratch/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

Ambos ficam em `data/scratch/`, que é git-ignored — o repositório do DINOv3 é
clonado localmente para uso, nunca vendorizado no GateFall.

Os caminhos podem ser sobrescritos, em ordem de precedência, pelas flags de
CLI `--repo-dir`/`--weights` ou pelas variáveis de ambiente
`GATEFALL_DINOV3_REPO_DIR`/`GATEFALL_DINOV3_WEIGHTS_PATH`.

### Falha rápida sem repositório ou pesos

Se o repositório ou o arquivo de pesos não existirem no caminho resolvido, a
extração falha imediatamente com `FileNotFoundError`, apontando qual flag ou
variável de ambiente usar — não há download automático nem inicialização
aleatória do backbone que mascare a ausência dos artefatos locais.

### Determinismo na inferência

`configure_deterministic_inference()` **não mexe no RNG global** — a
inferência do DINOv3 não consome aleatoriedade, então a função só ativa
`torch.backends.cudnn.deterministic`, desativa
`torch.backends.cudnn.benchmark` e o TF32 (tanto em
`torch.backends.cuda.matmul.allow_tf32` quanto em
`torch.backends.cudnn.allow_tf32`) e chama
`torch.use_deterministic_algorithms(True, warn_only=True)`. Ela é chamada
explicitamente por cada ponto de entrada de extração (`run_dinov3_extract` e
`run_dinov3_extract_all`), imediatamente antes de carregar o backbone —
nunca implicitamente dentro de `load_backbone`. Isso importa porque o treino
do braço A (ver `docs/train/gpu-determinism.md`) já é dono da seed global
(42) e chamar `torch.manual_seed(0)` aqui, como acontecia antes, a
sobrescreveria silenciosamente caso um futuro treinador do braço B reutilize
`load_backbone`.

O `warn_only=True` faz operações sem implementação determinística cair para
um aviso em vez de lançar exceção, e por isso a variável de ambiente
`CUBLAS_WORKSPACE_CONFIG` (necessária apenas para o modo estrito, sem
`warn_only`) não é exigida aqui. Esse contrato é mais permissivo que o do
treino do braço A, que exige `torch.use_deterministic_algorithms(True)`
**sem** `warn_only` e `CUBLAS_WORKSPACE_CONFIG` fixado (documentado em
[Treino — Investigação de determinismo de GPU](../train/gpu-determinism.md)).
A diferença é justificada: o treino atualiza pesos a partir de gradientes
estocásticos e precisa do modo estrito para garantir reprodutibilidade
bit-a-bit ponta a ponta; a inferência do DINOv3 não consome RNG algum, e o
contrato mais frouxo já é suficiente na prática — `verify-determinism`
(abaixo) mede hashes SHA-256 bit-idênticos entre duas extrações do mesmo
vídeo em hardware/driver fixos, validando essa propriedade diretamente em
vez de depender do modo estrito.

## Pré-processamento

Cada quadro RGB decodificado é redimensionado para 224×224 com interpolação
bicúbica e antialiasing, normalizado para `[0, 1]` e depois padronizado com
as constantes de normalização do ImageNet (`mean=(0.485, 0.456, 0.406)`,
`std=(0.229, 0.224, 0.225)`), a mesma convenção usada pelos pesos
pré-treinados do DINOv3.

## Vetor de features por quadro

Para cada quadro, o descritor de 1536 dimensões concatena o token CLS
(`x_norm_clstoken`, 768 dimensões) com a média espacial dos patch tokens
(`x_norm_patchtokens`, 768 dimensões):

```text
feature = concat(cls_token, mean(patch_tokens, eixo espacial))  # [1536]
```

O CLS token carrega contexto global do quadro; a média dos patch tokens
carrega contexto espacial agregado. As features são gravadas em `float16`,
diferente das features cinemáticas do braço A, que são `float32`. No range
de valores observado no dataset (`max_abs≈5.86`, medido pelo `audit`), o
erro relativo de quantização por dimensão é de aproximadamente `0,07%`:
o ULP do `float16` nesse valor é `≈0,0039`
(`np.spacing(np.float16(5.86))`), e `0,0039 / 5,86 ≈ 0,0007`, ou `0,07%`.

## Schema do HDF5

Um arquivo por vídeo, em `data/features/le2i/dinov3/<env>/<video_name>.h5`
(resolvido por `adapter.dinov3_root`), com:

- dataset `features`: shape `[K, 1536]`, `float16`, `K` = número de quadros
  do vídeo na grade temporal;
- atributos: `video_id`, `env`, `split`, `subject`, `K`, `fps`, `width`,
  `height`, `target_fps`, `model_name`, `feature_dim`, `weights_sha256`
  (hash SHA-256 do arquivo de pesos), `dinov3_repo_commit` (commit do clone
  local do repositório), `resize_height`, `resize_width`, `normalize_mean`,
  `normalize_std`, `torch_version`, `torchvision_version`.

A gravação é atômica (escreve em `.tmp` e usa `os.replace`) e, depois do
`replace`, relê o arquivo do disco e compara dataset e atributos com o
conteúdo em memória byte a byte antes de considerar a gravação bem-sucedida.

Nenhuma estatística de padronização (z-score) é calculada nesta etapa, e o
contrato do adapter não reserva caminho algum para ela — a padronização das
features DINOv3, análoga à de pose, entra em uma etapa posterior.

### Somente o protocolo cs

Todos os comandos de `gatefall.dinov3.extract` aceitam apenas `--dataset le2i`,
isto é, o Le2i sob o protocolo `cs`. Um adapter `le2i-cv` passado
programaticamente é rejeitado com `ValueError` pelos seis pontos de entrada
(`extract`, `extract-all`, `report`, `audit`, `verify-determinism` e
`verify-frame-alignment`), antes de qualquer efeito colateral.

O motivo é que `dinov3_root` não é derivado do protocolo: os dois adapters
apontam para o mesmo `data/features/le2i/dinov3/`. Uma extração sob `cv`
sobrescreveria os `.h5` do `cs` gravando `env`, `split` e `subject` vindos do
manifesto `cv`, e o `report` compararia as contagens de quadros do `cv` contra
os totais por split do `cs`. A contaminação seria silenciosa. O braço DINOv3,
portanto, não participa do [relatório de
generalização](../eval/le2i-cv-generalization.md).

## Como executar

```bash
uv run python -m gatefall.dinov3.extract selftest
```

Roda checagens sintéticas de pré-processamento e armazenamento, as fixtures
sintéticas de `verify-frame-alignment` (caminho feliz, deslocamento de quadro,
`K` divergente, manifesto sem o vídeo e `.h5` ausente) e a guarda que restringe
o braço ao protocolo `cs`, sem tocar no repositório do DINOv3, nos pesos ou no
dataset real.

```bash
uv run python -m gatefall.dinov3.extract extract --video-id ID [--repo-dir DIR] [--weights PATH] [--batch-size N] [--force] [--dataset le2i]
```

Extrai features de um único vídeo e grava o `.h5`. Se já existir um `.h5`
para o vídeo, ele só é pulado quando é válido: abre corretamente, tem o `K`
esperado a partir de `frames.parquet`, shape `[K, 1536]` em `float16` e
atributos de proveniência (`model_name`, `feature_dim`, `weights_sha256`,
`dinov3_repo_commit`, `resize_height`, `resize_width`, `normalize_mean`,
`normalize_std`, `target_fps`) idênticos aos da extração atual. Um `.h5`
existente mas inválido faz o comando falhar (código de saída diferente de
zero) sem reextrair, a menos que `--force` seja passado — nesse caso o
motivo da invalidade é impresso e o arquivo é reextraído.

```bash
uv run python -m gatefall.dinov3.extract extract-all [--repo-dir DIR] [--weights PATH] [--batch-size N] [--force] [--dataset le2i]
```

Extrai features de todos os vídeos listados em `frames.parquet`,
reutilizando o mesmo backbone carregado uma única vez. Ao final, imprime
quantos vídeos foram processados, pulados e com falha; falhas não
interrompem os demais vídeos, mas fazem o comando sair com código de erro.

```bash
uv run python -m gatefall.dinov3.extract report [--dataset le2i]
```

Valida a cobertura dos `.h5` já extraídos contra `frames.parquet`: vídeos
ausentes, `K` divergente do número de quadros esperado, contagem de quadros
por split contra os totais esperados do Le2i, e homogeneidade de
proveniência entre todos os `.h5` — os atributos `model_name`,
`feature_dim`, `weights_sha256`, `dinov3_repo_commit`, `resize_height`,
`resize_width`, `normalize_mean`, `normalize_std` e `target_fps` devem
estar presentes e idênticos em todo arquivo extraído, inclusive quando há
um único vídeo no dataset ou quando o atributo está ausente de todos os
arquivos ao mesmo tempo; qualquer divergência ou ausência (por exemplo,
uma extração parcial feita com pesos ou commit diferentes, ou um `.h5`
gravado por uma versão antiga do extrator sem algum atributo) faz o
`report` falhar.

```bash
uv run python -m gatefall.dinov3.extract audit [--dataset le2i]
```

Audita a qualidade das features já extraídas, vídeo a vídeo e no dataset
inteiro: ausência de valores não finitos (NaN/Inf), valor máximo absoluto
frente ao teto do `float16` (65504 — apenas informativo, não falha o
comando), variância por dimensão maior que zero (reporta dimensões mortas
entre as 1536), ausência de linhas consecutivas duplicadas dentro do mesmo
vídeo, contiguidade do `frame_index` (0..K-1) por vídeo contra
`frames.parquet`, e `K` de DINOv3 igual ao `K` do `.h5` de pose
correspondente.

```bash
uv run python -m gatefall.dinov3.extract verify-determinism --video-id ID [--repo-dir DIR] [--weights PATH] [--batch-size N] [--output-dir DIR] [--dataset le2i]
```

Reextrai um único vídeo duas vezes com `--force` e compara o hash SHA-256
dos bytes brutos do array `features` entre as duas extrações. Exige backbone,
pesos e GPU reais — **não faz parte do `selftest` nem de nenhuma checagem de
CI**; é uma validação manual em hardware real de que a configuração de
determinismo descrita acima realmente produz saídas bit-idênticas.

Por padrão (sem `--output-dir`), as duas extrações de verificação são
gravadas em um diretório temporário efêmero, nunca em `data/features/` —
o comando imprime qual modo está rodando. Com `--output-dir`, as extrações
vão para o diretório indicado, que é criado se não existir: apontá-lo para o
caminho canônico do dinov3 (`adapter.dinov3_root`) sobrescreve o dataset
real e reproduz o comportamento antigo — esse é o único modo destrutivo, e é
opt-in explícito; qualquer outro caminho é tratado como diretório de
trabalho comum.

```bash
uv run python -m gatefall.dinov3.extract verify-frame-alignment [--repo-dir DIR] [--weights PATH] [--dataset le2i]
```

Confere, por reamostragem independente, que `decode_frames` retorna
realmente o quadro pedido pelo `src_index` — um erro sistemático de
deslocamento de um quadro não é detectado por nenhuma outra checagem
existente (contagens de `K`, contiguidade de `frame_index`, concordância de
`K` com pose). Para uma amostra fixa de dois vídeos (`coffee_room_01/video_1`,
~25 fps, e `home_01/video_1`, ~24,000384 fps), recomputa as features nas
posições início/meio/fim da grade (`grid_positions`) redecodificando o
quadro e comparando com a linha armazenada no `.h5`:

- **tolerância**: a diferença absoluta máxima contra a linha esperada deve
  ficar dentro de `ULP_TOLERANCE_MULTIPLE = 2.0` vezes o ULP do `float16` no
  valor observado. O vetor recomputado e a linha armazenada já são ambos
  `float16` (`compute_features` sempre converte para `float16` antes de
  retornar), então redecodificar exatamente o mesmo quadro já dá
  `max_abs_diff == 0`; a tolerância baseada em ULP existe como margem para
  não associatividade de ponto flutuante entre a extração em lote original e
  esta reinferência quadro a quadro (tamanho de batch e kernels de GPU
  diferentes), não por diferença de precisão entre os dois lados;
- **discriminação**: o vetor recomputado deve ser estritamente mais próximo
  da linha esperada do que das linhas vizinhas (posições `k-1` e `k+1`) —
  é essa checagem que pegaria um deslocamento sistemático de um quadro, que
  passaria despercebido pelas demais. Um empate exato de distância contra
  uma vizinha é reportado como inconclusivo, não como falha — evita uma
  falha espúria quando duas linhas armazenadas coincidem por acaso.

Exige backbone, pesos e GPU reais, como `verify-determinism`; **não faz
parte do `selftest` nem de nenhuma checagem de CI** e nunca grava em
`data/features/` (é somente leitura).

## Licença do DINOv3

O código e os pesos do DINOv3 utilizados aqui **não são MIT nem
Apache-2.0**: são distribuídos sob a licença própria do DINOv3, definida
pelo repositório de referência da Meta. Este repositório GateFall:

- **não redistribui** o código nem os pesos do DINOv3 — o repositório de
  referência é clonado localmente pelo usuário em `data/scratch/dinov3_repo`
  (git-ignored) e os pesos são baixados separadamente para
  `data/scratch/weights/dinov3/`, nunca vendorizados ou commitados neste
  repositório;
- declara que este projeto é **"Built with DINOv3"**;
- reconhece que qualquer publicação decorrente do uso deste código deve
  reconhecer o uso de **"DINO Materials"**, conforme exigido pela licença do
  DINOv3.

Consulte os termos completos e atualizados diretamente no repositório oficial
do DINOv3 antes de baixar, redistribuir ou publicar resultados obtidos com
ele; a licença MIT deste repositório não se estende a esses materiais.
