# Features DINOv3

`src/gatefall/dinov3/` implementa a extração offline de features visuais do
braço B a partir do backbone congelado **DINOv3**. A CLI fina
`src/gatefall/dinov3/extract.py` (`extract`, `extract-all`, `report`,
`selftest`) opera sobre a mesma grade temporal (`frames.parquet`) e o mesmo
manifesto usados pelo braço A — nenhuma janela ou split é recalculado aqui.
Fusão com pose, treino e avaliação do braço B ainda não estão implementados.

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
carrega contexto espacial agregado. As features são gravadas em `float16`.

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

Nenhuma estatística de padronização (z-score) é calculada nesta etapa;
`dinov3_stats_path` é apenas reservado no contrato do adapter para uma etapa
futura de padronização, análoga à de pose.

## Como executar

```bash
uv run python -m gatefall.dinov3.extract selftest
```

Roda checagens sintéticas de pré-processamento e armazenamento, sem tocar no
repositório do DINOv3, nos pesos ou no dataset real.

```bash
uv run python -m gatefall.dinov3.extract extract --video-id ID [--repo-dir DIR] [--weights PATH] [--batch-size N] [--force] [--dataset le2i]
```

Extrai features de um único vídeo e grava o `.h5`. Pula vídeos já extraídos a
menos que `--force` seja passado.

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
ausentes, `K` divergente do número de quadros esperado, e contagem de
quadros por split contra os totais esperados do Le2i.

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
