# Features de qualidade (`q_pose`, `q_visual`)

`gatefall.features.quality_extract` persiste offline, por vídeo, o par de
proxies operacionais de qualidade por quadro consumido pelo gate adaptativo da
[arma B1](../train/b1-adaptive-gate.md). Nenhuma fórmula nova é introduzida
aqui: `q_pose` vem de [`compute_pose_quality`](pose-quality.md) e `q_visual`
vem de [`compute_visual_quality`](dinov3-quality.md), aplicado exatamente sobre
o mesmo pré-processamento de quadro da extração DINOv3
(`resize_frames(decode_frames(...))`, `224 × 224` em `[0, 1]`).

Os dois valores são **proxies operacionais de qualidade em `[0, 1]`, não
probabilidades calibradas nem escores de confiança**. O mecanismo que os
consome é descrito como fusão adaptativa.

## Schema do HDF5

Um arquivo por vídeo, nunca um por quadro (`CLAUDE.md`, invariante 5):

```
data/features/le2i/quality/<env>/<video>.h5
```

| Elemento | Conteúdo |
| --- | --- |
| dataset `quality` | `[K, 2]` `float32` — coluna 0 = `q_pose`, coluna 1 = `q_visual` |
| attrs de identidade | `video_id`, `env`, `split`, `subject`, `K`, `fps` |
| attrs de proveniência | `channel_names`, `target_fps`, `pose_quality_source`, `visual_quality_source`, `pose_source_sha256`, `resize_height`, `resize_width` |

A ordem de canal (`0 = q_pose`, `1 = q_visual`) é contrato persistido:
`QUALITY_CHANNEL_NAMES` em `gatefall.features.quality_storage` é a única fonte
dessa ordem, e tanto o dataset de janelas quanto o gate a assumem.

`K` é o número de quadros da grade temporal daquele vídeo em `frames.parquet` —
o mesmo `K` das features de pose e DINOv3. A extração falha se `q_pose` e a
sequência decodificada não cobrirem exatamente os mesmos quadros, de modo que o
alinhamento quadro a quadro entre as três fontes é estrutural.

A gravação é atômica (arquivo temporário + `os.replace`) e seguida de uma
releitura que compara valores e atributos com o que foi gravado, como em
[features DINOv3](dinov3-features.md).

`pose_source_sha256` é o SHA-256 do sidecar de pose
(`<pose_root>/<env>/<video>.h5`) do qual `q_pose` foi derivado, pelo mesmo
motivo que a extração DINOv3 registra `weights_sha256`: sem ele, uma reextração
de pose deixaria o sidecar de qualidade intacto e silenciosamente superado, e o
treino do B1 usaria `q_pose` derivado de dados já substituídos sem que nenhum
digest da receita mudasse.

## Causalidade e escopo monocular

`q_visual` depende somente do quadro RGB atual. `q_pose` depende do quadro
atual e da última observação **anterior** de pose; nenhum quadro futuro entra
no cálculo. Nenhum descritor derivado de profundidade participa: o pipeline
permanece monocular RGB (`CLAUDE.md`, invariante 3).

## Idempotência

Um sidecar já existente e válido (shape, dtype e proveniência batendo,
incluindo `pose_source_sha256`) é preservado e o vídeo é pulado; um sidecar
inválido — inclusive um cuja pose de origem foi reextraída — faz a extração
falhar com o motivo, e só `--force` reextrai. Os schemas persistidos de pose e DINOv3 e seus
artefatos de padronização não são tocados por esta etapa.

A detecção de sidecar superado vive nesta CLI, não no treino: `b1_gate train` e
`b1_gate report` hasheiam os arquivos de qualidade, que não mudam quando só a
pose é reextraída. Depois de qualquer reextração de pose, rode
`quality_extract extract-all` (ou ao menos `quality_extract report`) antes do
treino do B1 — é esse passo que compara `pose_source_sha256` e recusa a
qualidade superada.

## Digest do conjunto

Como a qualidade é persistida por vídeo, não existe um único arquivo a hashear.
`quality_set_sha256` calcula o SHA-256 das linhas `<video_id> <sha256 do
arquivo>` ordenadas por `video_id`; é esse digest que a receita de treino do B1
registra em `quality_features_sha256`, tornando o run sensível à reextração de
qualquer vídeo.

## Como executar

```bash
uv run python -m gatefall.features.quality_extract selftest
```

Roda as checagens sintéticas da montagem `[K, 2]` e do armazenamento, sem tocar
no dataset real nem carregar backbone.

```bash
uv run python -m gatefall.features.quality_extract extract-all --dataset le2i
```

Extrai os sidecars de todos os vídeos de `frames.parquet`. Requer as features de
pose já extraídas e os vídeos do Le2i em disco; não requer os pesos do DINOv3,
porque `q_visual` é calculado sobre o quadro pré-processado, não sobre o
descritor.

```bash
uv run python -m gatefall.features.quality_extract report --dataset le2i
```

Relata cobertura e a distribuição de cada canal por split, sem mutar nada. Cada
sidecar passa pela mesma `validate_existing_file` da extração, então o relatório
aponta também divergência de proveniência (canais, `target_fps`, resize ou
`pose_source_sha256`), e não só de shape e dtype.

## Somente Le2i CS

Como as demais etapas que dependem da grade DINOv3, a CLI aceita apenas
`--dataset le2i` (`DINOV3_SUPPORTED_DATASET_IDENTIFIERS`).
