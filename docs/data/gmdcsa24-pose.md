# Pose do GMDCSA24 (piloto com rótulos fracos)

`gatefall.data.gmdcsa24.pose_extract` implementa a etapa de extração da
[autorização do passo 4](../analysis/gmdcsa24-weak-label-protocol.md#autorizacao-do-passo-4):
pose offline dos 145 vídeos elegíveis do GMDCSA24 com o backbone congelado do
braço A, numa saída isolada. O módulo não muda o protocolo, não ajusta
padronização, não monta treino de T ou S, não calcula métrica de modelo e não
usa DINOv3 nem SAM 3.

**Estado:** extração real informada pelo responsável no commit integrado
`5fc8204` (ver [Extração informada para o commit
5fc8204](#extracao-informada-para-o-commit-5fc8204)). O PR que introduziu este
módulo não executou extração real.

## Evidência E0 informada para o commit do PR #135

O responsável pelo projeto informou que o portão E0 passou no checkout de
`experiment/external-fall-data` no merge do PR #135
(`2888cefee64ae67d3b0ceb939a76db9548f108c2`). É **evidência informada pelo
responsável**, não reexecutada de forma independente neste PR:

- checkout limpo;
- `code.commit` do `manifest.json` do adapter igual a
  `2888cefee64ae67d3b0ceb939a76db9548f108c2` e `code.dirty = false`;
- as seis condições do `adapter validate` (integridade, proveniência, W1, W2,
  V2 e V5) aprovadas;
- T = 1 973 e S = 1 562;
- os hashes de conteúdo de `windows_t.parquet`, `windows_s.parquet` e
  `window_candidates.parquet` iguais aos de
  [Resultado sobre os dados pinados](../analysis/gmdcsa24-weak-label-protocol.md#resultado-sobre-os-dados-pinados).

Essa evidência vale para o commit `2888cef`. A integração deste módulo cria um
novo commit integrado, e a extração exige um E0 refeito nele.

## Extração informada para o commit 5fc8204

O responsável pelo projeto informou que extraiu e validou de forma
independente a pose dos 145 vídeos elegíveis no commit integrado
`5fc8204ae474d36d849e87000b5d4b52cd3d5177` (merge do PR que introduziu este
módulo), com as seis condições do `pose_extract validate` aprovadas. É
**evidência informada pelo responsável**; a extração não foi refeita depois.

Conferido sem gravar nada no PR do [treino de T e
S](../train/gmdcsa24-weak-label.md#execucao-de-leitura-sobre-os-dados-reais):

- `code.commit` dos manifestos do adapter e da pose igual a
  `5fc8204ae474d36d849e87000b5d4b52cd3d5177`, com `code.dirty = false`;
- resumo do manifesto da pose: 145 vídeos, 11 710 quadros da grade e 11 421
  com pessoa;
- SHA-256 do `manifest.json` do adapter
  `f7c741940bc3cb0ebd40996710a04314083a16794f3889040752aa08b71e5c01` e do da
  pose `7d258616254a9c93500cb3aaaa037f79097510059a44eae191f7cb6e4130263d`;
- as seis condições do `pose_extract validate` aprovadas sobre essa saída.

O treino usa essa extração como está. Um commit novo do código de treino não
exige refazer o adapter nem a extração.

## Contrato congelado do braço A

A extração reutiliza a inferência da extração Le2i (`track_pose` e
`write_pose_file` de `gatefall.pose.extract`): o mesmo YOLO-Pose, o mesmo
ByteTrack com `persist=True`, o reset do tracker no início de cada vídeo, um
`PersonSelector` novo por vídeo, a conversão RGB → BGR e o mesmo esquema
HDF5. A extração recusa qualquer divergência deste contrato:

| Item | Valor exigido |
| --- | --- |
| Modelo | `yolo26n-pose.pt` em `data/scratch/weights/` |
| SHA-256 dos pesos | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| Ultralytics | `8.4.131` |
| Tracker | `bytetrack.yaml` |
| Grade | 10 fps, 17 keypoints COCO |

Os 190 HDF5 de pose do Le2i registram nome do modelo, versão do Ultralytics,
tracker e fps, mas não o hash dos pesos. O SHA-256 acima é o do
`data/scratch/weights/yolo26n-pose.pt` local, o caminho que a extração Le2i
resolve, com data anterior à dos HDF5 do Le2i; é uma inferência, não um hash
registrado na extração Le2i. A extração do GMDCSA24 não baixa pesos: sem o arquivo, ou com outro hash, ela
para. O SHA-256 do `bytetrack.yaml` instalado e a versão do PyTorch ficam só
registrados.

## Decodificação e alinhamento

Os MP4 do GMDCSA24 são decodificados pelo pipe passthrough do adapter
(`ffmpeg -fps_mode passthrough`), nunca pelo decodificador do Le2i, que
duplica quadros nos vídeos de taxa variável. `GridFrameReader` lê o vídeo
inteiro, entrega só os quadros cujo índice de decodificação está em
`src_index` e conta o total decodificado.

Antes da inferência, a grade de cada vídeo em `frames.parquet` precisa ter
`frame_index` de 0 a K−1, K igual a `grid_frames` de `videos.parquet`,
`src_index` igual ao de `build_time_grid` e estritamente crescente, e a
contagem do decodificador do adapter igual à de `ffprobe -count_frames`.
Depois da inferência, o total decodificado precisa ser igual a
`n_frames_decoded`. A linha `k` do HDF5 corresponde à linha `k` da grade.

## Pré-requisitos

`extract` falha fechado, antes de carregar o modelo ou gravar qualquer
arquivo, quando:

- a saída do adapter está ausente ou não passa nas seis condições do
  `adapter validate`;
- os hashes de `windows_t.parquet`, `windows_s.parquet` ou
  `window_candidates.parquet` diferem dos publicados;
- o manifesto do adapter tem `code.dirty` diferente de `false`, ou
  `code.commit` diferente do HEAD atual (E0 não refeito neste commit);
- o checkout do GateFall tem arquivo modificado ou não rastreado;
- `videos.parquet` não tem exatamente 145 vídeos incluídos, ou algum vídeo
  pedido em `--video-id` está excluído ou é desconhecido;
- a grade de algum vídeo elegível está desalinhada;
- os pesos ou o Ultralytics divergem do contrato do braço A;
- a saída se sobrepõe a `data/features/le2i/`, `data/processed/` (Le2i-CS,
  Le2i-CV e a saída do adapter), `data/labels/`, `data/raw/`, `runs/`, às
  estatísticas de padronização ou à revisão de proveniência, ou não é
  ignorada pelo Git.

Por vídeo, a extração também falha quando o MP4 está ausente, quando o SHA-256
do MP4 difere do `source_sha256` do adapter ou quando o total decodificado
difere de `n_frames_decoded`. Nesses casos nenhum HDF5 é gravado. A fonte é
conferida antes de qualquer outra decisão sobre o vídeo, inclusive quando já
existe um HDF5 válido para ele.

## Saída

```text
data/features/gmdcsa24/pose/
├── manifest.json
└── Subject_<n>/<ADL|Fall>/<nn>.h5
```

O caminho segue `gatefall.pose.loading.pose_path`, então `load_pose` lê esses
arquivos como lê os do Le2i. Cada HDF5 tem os datasets do Le2i (`keypoints`
`[K, 17, 3]`, `bbox` `[K, 4]`, `person_found`, `n_detections` e `track_id`) e
os atributos do Le2i, com `env = GMDCSA24` e `split = train`. A proveniência
fica em atributos extras: `dataset`, `relative_path`, `source_sha256`,
`grid_sha256`, `n_frames_decoded`, `weights_sha256`, `tracker_config_sha256`,
`adapter_manifest_sha256`, `code_commit` e `extraction_code_sha256`.

O `manifest.json` é gravado por último e só com os 145 vídeos extraídos e
conferidos. Ele registra o protocolo e a fonte do adapter, o SHA-256 do
manifesto do adapter, os hashes das tabelas, a identidade do modelo, o commit
e o hash do código de extração, e, por vídeo, o SHA-256 do HDF5, K, o total de
quadros com pessoa, `source_sha256`, `grid_sha256` e `n_frames_decoded`. Sem
manifesto, a saída não valida.

Um HDF5 existente que confere com a proveniência atual é pulado, desde que o
MP4 correspondente exista e tenha o `source_sha256` do adapter; um que não
confere é recusado, e só `--force` o refaz. Com o manifesto presente,
`extract` sem `--force` é recusado. `--video-id` extrai só os vídeos pedidos e
não grava manifesto enquanto a cobertura estiver incompleta.

## Validação

`validate` relê a saída do adapter, a grade e cada HDF5, sem gravar nada, e só
passa com as seis condições:

- **adapter:** as seis condições do `adapter validate`, os hashes publicados
  das tabelas e a seção `code` do E0;
- **separação:** saída fora dos artefatos canônicos e ignorada pelo Git;
- **manifesto:** modelo igual ao contrato do braço A, manifesto do adapter com
  o mesmo SHA-256 registrado na extração, extração no commit do E0 e
  `dirty = false`, o mesmo SHA-256 de protocolo do adapter, e `summary`
  igual ao resumo recalculado dos HDF5 conferidos (vídeos, quadros da grade e
  quadros com pessoa), sem campo ausente ou extra;
- **cobertura:** exatamente os 145 HDF5 elegíveis, nenhum vídeo excluído,
  nenhum arquivo extra ou temporário de escrita interrompida, e o manifesto com
  um registro para cada vídeo elegível e nenhum outro;
- **alinhamento:** a grade de cada vídeo elegível, como em
  [Decodificação e alinhamento](#decodificacao-e-alinhamento);
- **HDF5:** shapes e dtypes do esquema, atributos iguais aos esperados,
  `person_found` igual a `n_detections > 0`, `track_id = -1` e coordenadas
  nulas sem pessoa, e valores finitos; e o registro de cada vídeo no
  manifesto igual ao recalculado do plano do adapter e do HDF5 gravado
  (caminho, SHA-256 do arquivo, K, quadros com pessoa, `source_sha256`,
  `grid_sha256` e `n_frames_decoded`), sem campo ausente ou extra.

A validação não confia nos valores do manifesto: cada registro e o resumo são
recalculados e comparados.

A validação não calcula métrica de modelo.

O `selftest` (na CI) usa o clone sintético do selftest do adapter e um modelo
falso que codifica o quadro de origem nos keypoints. Ele cobre decodificação
de taxa variável, recusa de quadro duplicado, isolamento do tracker entre
vídeos, cada pré-requisito acima, HDF5 existente que confere ou não, fonte
alterada ou ausente com HDF5 válido já gravado, saída incompleta, divergências
de proveniência, registros e resumo do manifesto adulterados e separação de
caminhos.

## E0 antes da extração

Depois da integração do PR deste módulo em `experiment/external-fall-data`, e
antes da extração real, o [portão E0](../analysis/gmdcsa24-weak-label-protocol.md#portoes-antes-de-cada-etapa)
é refeito no novo commit integrado, na raiz de um checkout limpo desse commit:

```bash
git rev-parse HEAD
git status --porcelain --untracked-files=all
uv run python -m gatefall.data.gmdcsa24.adapter build --force
uv run python -m gatefall.data.gmdcsa24.adapter validate
uv run python -c "import json; m = json.load(open('data/processed/gmdcsa24/manifest.json')); print(m['code'], m['protocol']['sha256']); print({k: m['files'][k]['content_sha256'] for k in ('windows_t.parquet', 'windows_s.parquet', 'window_candidates.parquet')})"
sha256sum data/processed/gmdcsa24/manifest.json
```

O `git status` não pode listar nada. As seis condições passam, T = 1 973,
S = 1 562, `code.commit` é o HEAD acima, `code.dirty = false` e os três hashes
são os publicados. O registro do E0 traz esses valores, o SHA-256 do
manifesto e o SHA-256 do protocolo.

Só então, no mesmo checkout:

```bash
uv run python -m gatefall.data.gmdcsa24.pose_extract extract
uv run python -m gatefall.data.gmdcsa24.pose_extract validate
```

A extração confere de novo o E0 (commit e checkout limpo) e recusa rodar se
algo mudou desde o build. Nada de `data/features/gmdcsa24/` é sincronizado
com o Google Drive.
