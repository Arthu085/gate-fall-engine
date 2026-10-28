# Contrato: artefatos e proveniência

## Regras

1. **Nada pesado no Git.** Dados, features, pacotes, runs, logs, perfis e relatórios ficam em
   `data/` e `runs/local/`, ambos ignorados pelo Git.
2. **Escrita única.**
   - Pacotes e runs são montados em diretório temporário e publicados com `os.replace`; um
     destino existente é recusado.
   - Avaliações, `study-provenance.json`, ledgers, perfis e relatórios são abertos com o modo `x`.
3. **Identidade de run.** O `identity_sha256` é o JSON canônico de: protocolo, dataset, braço,
   fusão, fonte visual, semente, receita de movimento, épocas, batch, política, `augment` e
   `order_weight`. O executor confere cada campo contra `config.json` antes de registrar.
4. **Reuso verificado.** O executor nunca pula um passo só porque suas saídas existem. Cada
   passo tem um verificador que devolve um destes estados:
   - `pending`: nada foi produzido, ou falta trabalho retomável. O passo roda;
   - `done`: as saídas conferem. O passo é pulado;
   - `invalid`: há uma saída existente que não confere. A execução para e nada é sobrescrito.

   Depois de rodar, o passo é verificado de novo. As conferências por tipo de passo são:

   | Passo | Conferência |
   | --- | --- |
   | `data:upfall:stage` | Cada ZIP de `data/scratch/upfall_archives` contra seu destino, pelos nomes e tamanhos do diretório central. Um ZIP sem destino deixa o passo `pending`, mesmo com `data/raw/upfall` existente; um destino divergente ou um `.partial` restante o deixa `invalid`. |
   | `fetch-labels` | Hashes de `PROVENANCE.json` e revisão pinada. |
   | `manifest` | Anotações verificadas e iguais ao SHA-256 registrado; schema, papel, partição e revisão; SHA-256 dos três parquets; vídeos iguais aos anotados; impressão digital das fontes por `stat`. |
   | `timegrid` | Contrato causal; SHA-256 da proveniência do manifesto, de `labels.csv` e de `frames.parquet`; vídeos iguais aos do manifesto. `timegrid` só reconstrói depois de conferir o manifesto e as anotações. |
   | `audit` | Grade conferida; `coverage.json` com os SHA-256 atuais (senão é refeito) e `passed = true`. |
   | `bundle` | Schema; todo arquivo listado existe com o tamanho registrado e não há arquivos extras; `sources.frames_sha256` igual ao `frames.parquet` atual. |
   | `targets` | Schema; `bundle_sha256` igual ao pacote atual; alvos listados presentes. |
   | `train` | Identidade planejada; `manifest_sha256` (e `targets_sha256` no movimento) iguais aos atuais ou, só no movimento, aos originais de um atestado de equivalência válido que os liga aos atuais; hashes de `config.json`, `stats.json`, `checkpoint.pt` e `last.pt`; `study-provenance.json`, se existir, inclusive o SHA-256 do atestado. Sem registro, o executor grava `origin = reused-verified`. |
   | `evaluate`, `transfer` | Run conferido; `split = test`, `selection_performed = false`, checkpoint, configuração e pacote iguais aos do run; na transferência, `adaptation = none` e o hash do pacote externo atual; em run atestado, `equivalence` igual ao atestado e ao pacote atuais. |

   O hash completo dos HDF5 de um pacote é conferido por `load_bundle` ao treinar e avaliar.
5. **Pesos.** O SHA-256 dos pesos YOLO-Pose, DINOv3 e SAM 3 entra no ledger e em cada
   proveniência de run. Um cache por (caminho, tamanho, mtime) evita reler gigabytes.
6. **Partição.** O `split_identity_sha256`, calculado sobre os pares (vídeo, partição) do pacote,
   entra na proveniência. O relatório exige rótulos e janelas de teste idênticos entre os métodos
   de um mesmo dataset.
7. **Agregação.** `report` reaplica o verificador de avaliação da regra 4 e recusa:
   - avaliação ausente ou sementes incompletas;
   - avaliação que não confere com o run, o pacote atual ou, na transferência, o pacote externo
     atual (`target_manifest_sha256`);
   - protocolo diferente, checkpoint diferente do registrado ou artefato alterado;
   - `selection_performed` diferente de `false`;
   - transferência com adaptação.
8. **Evidência histórica.** Os runs de `runs/reference` são somente leitura, e os runs externos
   incompatíveis ficam fora dos relatórios do estudo. Um run de movimento histórico entra só com
   um atestado de equivalência explícito (`equivalence/motion-<visual>.json`, escrito uma vez),
   que liga pacote e alvos originais aos atuais sem reescrever manifestos ou hashes; regras e
   limites em `docs/experiments/comparative-study.md`.

## Conteúdo mínimo por run

| Campo | Onde |
| --- | --- |
| Identidade, semente, política de seleção, configuração | `config.json`, `study-provenance.json` |
| Revisão Git e alterações locais | `config.json.code_provenance`, `study-provenance.json.code` |
| Pesos e extratores | `study-provenance.json.weights`, `bundle.sources` |
| Schema de features e estatísticas | `config.json`, `stats.json` (só treino) |
| Hash da partição e do pacote | `study-provenance.json.bundle` |
| Predições, eventos, alarmes | `evaluation-*.json` |
| Tempo e memória | `metrics.json.timing`, `evaluation-*.json.timing`, ledger, perfil de extração |
