# Modelo de dados

## Dataset encenado (`data/processed/<id>/`)

| Arquivo | Conteúdo |
| --- | --- |
| `manifest.parquet` | Esquema de manifesto compartilhado com o Le2i, um registro por vídeo de câmera: `video_id`, `dataset`, `relative_path`, `absolute_path`, `env`, `subject`, `cam`, `split`, `fps`, `fps_source`, `n_frames_header`, `n_frames_counted`, `duration_s`, `width`, `height`, `codec`, `sha256`. |
| `identities.parquet` | Identidade completa por vídeo, incluindo `recording_id` (vistas sincronizadas compartilham o valor). |
| `sequence_index.parquet` | UP-Fall: ordem e timestamp de cada imagem. |
| `frames.parquet` | Grade de 10 fps: `video_id`, `frame_index`, `src_index`, `label`, `split`, posição (`source_gap` quando desalinhado). |
| `alignment.json` | Deslocamento máximo, passos desalinhados e lacunas por vídeo. |
| `coverage.json` | Por partição: passos por classe, `IGNORE`, `source_gap`, eventos de queda, horas, vazamentos e sincronia entre vistas. |
| `dataset_provenance.json` | Partição e hash, revisão OmniFall, hash do manifesto, `unlabeled_sources`, estado Git. |

## Pacote de features (`runs/local/<proto>/study2026/bundles/<visual>/`)

O `manifest.json` registra:

- schema `pose138-valid-intervals-v1`;
- `visual_kind` e `visual_dim`;
- `label_names`, `evidence_names`, `splits`;
- `sources`, com hash por arquivo de origem, extrator de pose e extrator DINOv3/SAM 3;
- `videos`, com `video_id`, `split`, caminho HDF5, `sha256` e `k`.

Cada `videos/<sha>.h5` contém `labels`, `pose`, `visual` e `evidence`, além das variantes
corrompidas apenas no treino.

## Run (`runs/local/<proto>/study2026/<arm-fusion>/seed-<n>/`)

| Arquivo | Escrita | Conteúdo |
| --- | --- | --- |
| `config.json` | treino | Braço, fusão, semente, épocas, batch, política, otimização, `manifest_sha256`, código. |
| `stats.json` | treino | Média e desvio só do treino (`fit_split = train`). |
| `checkpoint.pt` / `last.pt` | treino | Checkpoint selecionado por validação / última época. |
| `metrics.json` | treino | Histórico de validação, época selecionada, hashes, `timing` (s, s/época, pico de memória). |
| `study-provenance.json` | executor, uma vez | Identidade e `identity_sha256`, `protocol_sha256`, Git, pacote (hash, `split_identity_sha256`, fontes), hashes de artefatos, pesos (SHA-256), ambiente, tempos. |
| `evaluation-test.json` | avaliação, uma vez | Classificação, eventos, alarmes, predições por janela, hashes, `timing`. |
| `evaluation-external-gmdcsa24.json` | transferência, uma vez | Como acima, com `adaptation: none`, datasets de origem/alvo e hash do pacote alvo. |

Os runs de movimento usam o layout importado `runs/local/<proto>/motion-<braço>/seed-<n>/<receita>/`.

## Registro do estudo (`runs/local/study2026/`)

| Caminho | Conteúdo |
| --- | --- |
| `execution/execution-<utc>-<id>.jsonl` | Ledger por execução: início (protocolo, Git, pesos, ambiente, passos), `present`/`reused`/`finished` com código de saída e tempo, `complete`. |
| `weights-sha256.json` | Cache de SHA-256 por (caminho, tamanho, mtime). |
| `profiles/extraction-<dataset>-<utc>.json` | Segundos por vídeo e por quadro (com aquecimento separado) e memória CUDA. |
| `reports/<matriz>-<utc>/` | Tabelas, figuras, linhas do tempo e `report-manifest.json`. |

Nenhum desses artefatos entra no Git.
