# Investigação não canônica de RGB temporal congelado

Esta investigação substitui novas tentativas de perda, amostragem e dropout
de pose depois dos resultados negativos da
[investigação de desbalanceamento](imbalance-investigation.md) e das
[intervenções anteriores](intervention-decision.md). A
[síntese entre braços](cross-arm-diagnostic-synthesis.md) motiva examinar
informação temporal RGB para as confusões persistentes. Há um único backbone,
uma receita e um controle estático. Não se cria um braço canônico nem se
combinam DINOv3 e SAM 3.

## Contrato registrado antes dos treinos

`gatefall.train.temporal_rgb` usa o R3D-18 de torchvision com os pesos
`R3D_18_Weights.KINETICS400_V1`, congelado em `eval()` e sob
`inference_mode()`. Remove-se só a classificação Kinetics; o descritor após
pooling tem 512 canais. Usa-se diretamente o
[preprocessing oficial](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html):
resize bilinear para 128×171, recorte central 112×112, conversão para
float32 e normalização pelas médias e desvios oficiais. Não há recorte de
pessoa, aumento, ajuste do backbone, autocast nem TF32.

Cada janela existente de 24 posições termina em `k_end`. Os 16 offsets são
`floor(linspace(0,23,16))`: `[0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,23]`.
Reutilizam-se `window_frame_indices`, a replicação da borda esquerda e o
rótulo do último quadro; endpoints `IGNORE_LABEL` são excluídos. O treino
usa stride 4, e o retido, stride 1.

A grade existente usa arredondamento para `src_index`, que pode apontar
alguns milissegundos além do endpoint. Para manter causalidade estrita, cada
índice selecionado é limitado a `floor(time_s[k_end] * fps_src)`. A grade
canônica permanece intacta. O cache registra a posição na janela, o índice
original e o índice causal efetivamente decodificado, com seu timestamp.
Nenhum quadro posterior ao endpoint chega ao backbone. O controle `static`
repete 16 vezes o último quadro causal, usando o mesmo backbone e
preprocessing de `dynamic`.

As features congeladas são extraídas uma vez por vídeo do `train` oficial.
Para cada modo e dobra, média e desvio dos 512 canais são ajustados só nas
janelas de desenvolvimento em stride 4; desvios têm piso `1e-6`. Treina-se
somente uma camada linear 512→10, com 5.130 parâmetros. A receita fixa usa
seed 42, 30 épocas, batch 64, AdamW (`lr=0.001`, `weight_decay=0.01`), cosine,
clipping 1 e entropia cruzada com frequência inversa calculada só nos
rótulos de desenvolvimento. O checkpoint é sempre o da última época. A
Macro-F1 restrita mantém exatamente `[0,1,2,3,4,7,8,9]`, incluindo zeros.

As dobras são as existentes:

- CS: `retain_s0`, `retain_s1`, `retain_s3-4-6-8`, com sujeitos disjuntos.
- CV: `retain_Coffee_room_01`, `retain_Coffee_room_02`,
  `retain_Lecture_room`, com ambientes e vídeos disjuntos.

Só linhas `split == train` são carregadas, com filtro no Parquet. As
partições são reconciliadas com os `folds.json` existentes e as entradas
atuais. As features podem ser compartilhadas entre dobras porque não são
ajustadas nos vídeos Le2i; a padronização e a cabeça são exclusivas da dobra.
`val` e `test` oficiais não selecionam receita, seed, limiar ou checkpoint.

O controle A é o run `baseline` da investigação de desbalanceamento, na
mesma dobra e seed. Ele usa a receita A original. A leitura verifica sua
configuração, checkpoint, métricas, entradas, estatística de pose e
proveniência atuais. Os hashes de pertencimento das janelas, os rótulos, os
strides e os suportes devem coincidir. Um controle ausente ou divergente
interrompe o piloto; ele nunca é retreinado automaticamente.

## Critério do piloto CS

As três dobras e os dois modos são obrigatórios. O piloto só indica
progressão se **todas** as condições forem satisfeitas:

1. Média das diferenças pareadas `dynamic − A` de pelo menos **+0,030**.
2. Ganho estritamente positivo contra A em pelo menos duas dobras.
3. Média `dynamic − static` de pelo menos **+0,010**, positiva em pelo menos
   duas dobras, como sinal predefinido de contribuição temporal.
4. Queda de revocação de `fall` ou `fallen` contra A de no máximo 0,050 em
   cada dobra.
5. Nenhum aumento dos FP combinados de `other` e `walk` em qualquer dobra.

O relatório também apresenta `static − A`. A indicação de progressão serve
para uma avaliação posterior com múltiplas seeds; esta implementação fica
limitada ao piloto seed 42 e não promove candidatos. CV é relatado
separadamente, com as mesmas guardas descritivas. O alvo 0,800 não participa
da decisão. Métricas de eventos, FA/h e latência ficam explicitamente
indisponíveis (`null`) até existir avaliador validado por dobra. Revocação
por janela não prova preservação da sensibilidade de eventos.

## Execução no checkout persistente

Execute a partir do repositório persistente, na branch
`analysis/gatefall-improvements`, depois de preparar as dobras existentes
de [CS](../train/subject-folds.md) e [CV](../train/internal-folds.md).
Os controles esperados já produzidos pela investigação anterior ficam em
`runs/local/imbalance_experiment/{cs,cv}/<dobra>/baseline/seed_42/`.
Não execute os pilotos de desbalanceamento novamente para esta etapa.

```bash
cd /home/arthu/projects/gate-fall/gate-fall-engine
git branch --show-current
uv sync --locked --dev
uv run python -m gatefall.train.temporal_rgb selftest
uv run python -m gatefall.train.temporal_rgb smoke --device cuda --download-weights

uv run python -m gatefall.train.temporal_rgb plan --protocol cs --device cuda
uv run python -m gatefall.train.temporal_rgb extract --protocol cs --device cuda
uv run python -m gatefall.train.temporal_rgb pilot --protocol cs --device cuda
uv run python -m gatefall.train.temporal_rgb report --protocol cs --device cuda
```

Após examinar o relatório CS, a execução equivalente para CV é:

```bash
uv run python -m gatefall.train.temporal_rgb plan --protocol cv --device cuda
uv run python -m gatefall.train.temporal_rgb extract --protocol cv --device cuda
uv run python -m gatefall.train.temporal_rgb pilot --protocol cv --device cuda
uv run python -m gatefall.train.temporal_rgb report --protocol cv --device cuda
```

`--fold-root` aceita o diretório das dobras existentes; os padrões são
`runs/local/le2i/subject_folds` e `runs/local/le2i_cv_internal_folds`.
Para CPU, troque **todas** as ocorrências de `--device cuda` por
`--device cpu`, inclusive no smoke. `extract` e `pilot` recusam worktrees
vinculados. Os comandos não criam PR nem promovem artefatos.

Toda saída fica em `runs/local/temporal_rgb/`, ignorado pelo Git:

```text
weights/r3d_18-b3b3357e.pth
smoke_<dispositivo>.json
<cs|cv>/feasibility.json
<cs|cv>/features/<sha256_do_video_id>.h5
<cs|cv>/runs/<dobra>/<dynamic|static>/seed_42/
    checkpoint.pt, metrics.json, predictions.npz, run.json
<cs|cv>/comparison.json
```

O download é explícito, valida o prefixo SHA-256 oficial e registra o hash
completo dos pesos. Cada HDF5 agrupa todas as janelas rotuladas de um vídeo,
com descritores dinâmicos e estáticos, rótulos, endpoints, pertencimento à
janela de 24 posições e alinhamento dos 16 quadros. Há hashes dos vídeos
RGB, entradas, dobras, arrays, features, pesos, lockfile e implementação.
Registram-se versões, dispositivo, cuDNN/CUDA, seed, threads e opções
determinísticas. Extração com código ou lockfile diferente é recusada.

O smoke usa apenas pixels sintéticos com os pesos reais. Mede uma inferência
aquecida, compara duas saídas byte a byte, confere a distinção do controle
estático e confirma que os parâmetros e buffers do backbone não mudam.
`plan` estima RAM por vídeo, armazenamento dos caches e duração de
inferência; recusa mais de 768 MiB de RGB decodificado por vídeo, RAM ou
disco insuficientes, VRAM sem margem de 50% e estimativa acima de 24 horas
por protocolo. O tempo estimado não inclui decodificação e I/O.

Os caches e os runs são publicados só depois de verificados. Uma falha não
substitui uma saída concluída. Repetir os comandos reutiliza apenas saídas
com proveniência e hashes íntegros; artefatos parciais, antigos ou
corrompidos são recusados, sem sobrescrita automática. Para refazer uma
rodada, arquive manualmente apenas o diretório experimental; nunca remova
controles existentes. Links que desviem a saída para outro destino são
recusados.

`report` verifica hashes, checkpoint final, padronização ajustada só no
desenvolvimento e reprodução exata das predições a partir das features
antes de relatar resultados. Registra Macro-F1, precisão, revocação, F1,
matriz de confusão e TP/TN/FP/FN de todas as classes, diferenças pareadas,
combinações ausentes e a decisão do piloto. Protocolos nunca são agregados.

## Validação e situação experimental

Os selftests sintéticos cobrem causalidade e prefixos, padding, exclusão de
rótulos ignorados, dobras/strides, vazamento, cache interrompido ou
adulterado, normalização sem retido, repetição do head, checkpoint final,
reprodução de métricas, relatório incompleto e guardas de recursos.

Validação de implementação em 17/10/2026: o selftest RGB temporal e o
selftest da investigação de desbalanceamento passaram; `uv run pyright`
terminou sem erros ou avisos, e `uv run mkdocs build --strict` passou. O
smoke com pesos oficiais passou no ambiente pinado, com PyTorch
`2.13.0+cu130`, torchvision `0.28.0+cu130` e uma GTX 1650. Produziu 512
canais, repetição byte a byte e descritores distintos para o clipe
dinâmico e o controle estático; os 33.166.272 parâmetros congelados e os
buffers permaneceram idênticos. Uma inferência aquecida levou cerca de
0,057 s, com pico de 203.022.848 bytes alocados em CUDA. Essa medição
sintética não estabelece o tempo ou a viabilidade de uma extração Le2i
completa. O registro integral fica em
`runs/local/temporal_rgb/smoke_cuda.json`, junto dos pesos locais cujo
SHA-256 é `b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d`.

**Nenhum treino ou extração real Le2i foi executado nesta implementação.**
Não há resultado real de Macro-F1, ganho contra A, preservação de eventos
ou observação do alvo 0,800 para esta proposta. O piloto continua
**não validado**, sem PR para `main` ou candidato promovido.

Mesmo com ganho futuro, três grupos e uma seed fornecem apenas evidência
exploratória. As janelas se sobrepõem, o `test` já foi conhecido por estudos
anteriores, e a comparação RGB versus A muda representação e arquitetura.
O controle estático isola a contribuição do histórico dentro da sonda RGB.
Os 16 quadros cobrem a janela Le2i de 2,4 s, com padding no início, uma
escala temporal diferente dos clipes de 16 quadros a 15 fps descritos na
avaliação Kinetics do fornecedor. O preprocessing oficial não elimina essa
diferença de domínio.
