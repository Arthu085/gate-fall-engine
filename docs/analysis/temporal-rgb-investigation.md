# Investigação não canônica de RGB temporal congelado

Esta investigação substitui novas tentativas de perda, amostragem e dropout
de pose depois dos resultados negativos da
[investigação de desbalanceamento](imbalance-investigation.md) e das
[intervenções anteriores](intervention-decision.md). A
[síntese entre braços](cross-arm-diagnostic-synthesis.md) motiva examinar
informação temporal RGB para as confusões persistentes. Há um único backbone,
uma receita e um controle estático. Não se cria um braço canônico nem se
combinam DINOv3 e SAM 3.

**Investigação encerrada após o piloto CS, seed 42.** O RGB dinâmico
superou o estático nas três dobras de desenvolvimento, mas ficou abaixo
de A em todas e falhou nos critérios predeclarados de progressão. Os
resultados e a proveniência local estão registrados abaixo. O experimento
canônico A/B/C e os resultados do manuscrito permanecem inalterados.

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

O controle A preferencial é o run nativo `baseline_a/seed_42` da mesma
dobra, nas raízes existentes de CS e CV. A leitura reutiliza os validadores
de cada protocolo: integridade de treino, `fold_run.json`, composição,
hashes das entradas, estatística ajustada só no desenvolvimento, receita A
original, seed 42, checkpoint da última época e mapeamento de avaliação.
Confere também os suportes das dez classes nos dois lados e os hashes de
pertencimento das janelas, rótulos e strides contra a entrada RGB.

Se o controle nativo não existir, o run `baseline` da investigação de
desbalanceamento continua aceito após validação própria. Se ambos existirem,
os dois precisam ser válidos e cientificamente equivalentes: mesma receita
(exceto `run_name`), mesmas janelas, métricas finais idênticas e tensores do
checkpoint exatamente iguais. Bytes de serialização diferentes são
aceitáveis. Qualquer discordância ou controle presente mas inválido
interrompe a verificação, sem fallback silencioso. As métricas completas,
configuração, proveniência e hashes dos artefatos originais dos controles
usados são preservados na proveniência RGB. Nenhum controle é retreinado,
copiado ou modificado automaticamente.

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

## Registro da execução no checkout persistente

Os comandos abaixo documentam o procedimento do piloto concluído no
checkout persistente, na branch `analysis/gatefall-improvements`, com as
dobras existentes de [CS](../train/subject-folds.md) e
[CV](../train/internal-folds.md). Não constituem recomendação de nova
execução após o encerramento.
Os seis controles persistentes ficam em:

- CS: `runs/local/le2i/subject_folds/<dobra>/runs/baseline_a/seed_42/`.
- CV: `runs/local/le2i_cv_internal_folds/<dobra>/runs/baseline_a/seed_42/`.

Cada diretório precisa conter `config.yaml`, `metrics.json`, `checkpoint.pt`
e `fold_run.json`; os `folds.json` e as estatísticas por dobra também são
obrigatórios. A ausência de `runs/local/imbalance_experiment/` não impede a
investigação. Não execute os pilotos de desbalanceamento novamente para
esta etapa.

```bash
cd /home/arthu/projects/gate-fall/gate-fall-engine
git branch --show-current
uv sync --locked --dev
uv run python -m gatefall.train.temporal_rgb selftest
uv run python -m gatefall.train.temporal_rgb verify-controls --protocol cs
uv run python -m gatefall.train.temporal_rgb verify-controls --protocol cv
uv run python -m gatefall.train.temporal_rgb smoke --device cuda --download-weights

uv run python -m gatefall.train.temporal_rgb plan --protocol cs --device cuda
```

`verify-controls` é somente leitura e emite JSON apenas após validar os
três controles do protocolo. As duas verificações validaram os seis
controles reais. A mudança de código da PR #128 invalidou o registro
anterior do smoke pelo hash de implementação; o registro persistente atual
corresponde ao código após essa correção. Extração e treino reais exigem
o checkout persistente.

O procedimento CS registrado foi:

```bash
uv run python -m gatefall.train.temporal_rgb extract --protocol cs --device cuda
uv run python -m gatefall.train.temporal_rgb pilot --protocol cs --device cuda
uv run python -m gatefall.train.temporal_rgb report --protocol cs --device cuda
```

Não há diretório `runs/local/temporal_rgb/cv/` no checkout validado;
portanto, não há preflight, extração ou piloto RGB CV local a relatar.
O resultado CS não justifica executar essas etapas.

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
As provas dos controles nativos exercitam os validadores reais em três
dobras CS e três CV, incluindo artefatos ausentes ou corrompidos,
proveniência divergente, dobra/seed incorretas, época final, estatísticas
com retido, suporte por classe, janelas diferentes e ausência de retreino.
Também verificam o fallback e a equivalência entre controles nativos e de
desbalanceamento, incluindo serializações diferentes e discordância de
métricas ou tensores. Esses artefatos são sintéticos e temporários; não
constituem, por si só, validação dos seis controles reais do checkout
persistente.

### Evidência local validada em 17/10/2026

A fonte primária dos resultados é
`runs/local/temporal_rgb/cs/comparison.json`, cujo SHA-256 é
`176500f0a36d878a100775e0f5194657982f583c7e2ce1e1eef6b46a9e8a1eec`.
O relatório registra `missing: []`, seed 42, os três pares
`static`/`dynamic` e os controles A nativos. A conferência usa os artefatos
persistentes, sem refazer extração, treino ou sobrescrever relatórios.

- **Smoke pré-treinado:** `runs/local/temporal_rgb/smoke_cuda.json`
  registra `synthetic_pretrained_smoke_passed`, PyTorch `2.13.0+cu130`,
  torchvision `0.28.0+cu130` e NVIDIA GeForce GTX 1650. Produziu 512 canais,
  repetição byte a byte e descritores dinâmico/estático distintos. Os
  33.166.272 parâmetros congelados e os buffers permaneceram idênticos.
  A inferência aquecida levou `0.053282754000065324` s, com pico de
  `203022848` bytes alocados em CUDA. O campo `real_data_trained: false`
  descreve apenas esse smoke sintético, não o piloto subsequente.
- **Preflight CS:** `cs/feasibility.json` registra `feasible: true` e
  `problems: []`, com 20.740 janelas e 41.480 inferências estimadas.
  A estimativa foi `0.6139357322007527` h de inferência, excluindo
  decodificação e I/O. O cache estimado sem compressão foi `105525120`
  bytes; o maior vídeo decodificado, `224409600` bytes, com
  `4940705792` bytes de RAM disponíveis. O disco exigido foi
  `1284792064` bytes, frente a `939535351808` bytes livres. Esses são
  recursos registrados no preflight, não medições atuais nem duração
  observada da extração completa.
- **Extração real:** os 133 vídeos do `train` oficial usados em CS possuem
  um HDF5 íntegro por vídeo em `cs/features/`, cobrindo as 20.740 janelas
  rotuladas em stride 1, com features `static` e `dynamic` de 512 canais.
  Foram conferidos os hashes dos vídeos RGB, arrays e features, a
  proveniência e o alinhamento causal contra as entradas atuais. Não há
  leitura de `val` ou `test` oficiais nesta investigação.
- **Seis controles A originais:** os validadores reais confirmaram os
  três runs CS (`retain_s0`, `retain_s1`, `retain_s3-4-6-8`) e os três CV
  (`retain_Coffee_room_01`, `retain_Coffee_room_02`, `retain_Lecture_room`),
  todos em `baseline_a/seed_42/` nas raízes indicadas acima. Foram
  conferidos receita, composição, hashes, estatísticas de desenvolvimento,
  checkpoint final, suportes por classe e pertencimento das janelas.
  Os três controles CS coincidem com os incorporados em `comparison.json`;
  a validação dos controles CV não constitui execução RGB CV.
- **Piloto CS concluído:** seis runs, três dobras × dois modos, seed 42,
  30 épocas e checkpoint da última época. A reconstrução do relatório
  em memória, com `build_report`, reproduziu exatamente
  `comparison.json`, incluindo diferenças e decisão. A verificação
  reproduziu as predições a partir dos seis checkpoints e features e
  confirmou a padronização ajustada só no desenvolvimento.

Os pesos locais têm SHA-256
`b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d`.
O smoke e os runs registram a implementação
`225866d46397415db1752d85441ccd5b80cac5ebd1ee8d739a0930c7378daa70`
e o lockfile
`859e22caf510c5836a756f05967ee368751e4d7c7780b99e3ab4512e72702004`,
conferidos contra o checkout após a PR #128. Os hashes de entradas,
composição e features permanecem nos JSONs e HDF5 locais. Não faltou nem
falhou artefato necessário à validação CS e dos seis controles A.

### Macro-F1 pareada nas dobras de desenvolvimento CS

Os valores abaixo mantêm a representação decimal dos números locais, sem
arredondamento. A métrica é a Macro-F1 restrita às oito classes do
contrato, avaliada no grupo retido de cada dobra do `train` oficial.
O mapeamento interno `val`/`test` dos controles A significa `retained`,
conforme `fold_run.json`; não representa os splits oficiais homônimos.

| Dobra | Baseline A | RGB estático | RGB dinâmico |
| --- | --- | --- | --- |
| `retain_s0` | 0.5584449623146064 | 0.3239285310919959 | 0.39249832731572265 |
| `retain_s1` | 0.6575916704711838 | 0.3115730369800097 | 0.40901715788166837 |
| `retain_s3-4-6-8` | 0.6156845792679513 | 0.3429590312215618 | 0.45721239046798223 |

| Dobra | `dynamic − A` | `static − A` | `dynamic − static` |
| --- | --- | --- | --- |
| `retain_s0` | -0.16594663499888374 | -0.2345164312226105 | 0.06856979622372678 |
| `retain_s1` | -0.24857451258951546 | -0.3460186334911741 | 0.09744412090165866 |
| `retain_s3-4-6-8` | -0.15847218879996905 | -0.2727255480463895 | 0.11425335924642044 |
| Média das diferenças pareadas | -0.19099777879612276 | -0.2844202042533914 | 0.09342242545726863 |

As médias são aritméticas entre dobras, sem agregar janelas ou protocolos.
As médias `dynamic − A` e `dynamic − static` constam em `pilot_decision`;
a média `static − A` foi calculada diretamente das três diferenças
registradas.

### Resultado exato das verificações predeclaradas

| Verificação em `pilot_decision.checks` | Exigência | Observado | Resultado |
| --- | --- | --- | --- |
| `mean_gain_vs_a` | Média `dynamic − A` ≥ +0.030 | -0.19099777879612276 | `false` |
| `positive_folds_vs_a` | Pelo menos 2 dobras com `dynamic − A` > 0 | 0 de 3 | `false` |
| `temporal_gain` | Média `dynamic − static` ≥ +0.010 | 0.09342242545726863 | `true` |
| `positive_folds_vs_static` | Pelo menos 2 dobras com `dynamic − static` > 0 | 3 de 3 | `true` |
| `fall_fallen_recall` | Cada diferença de revocação contra A ≥ -0.050 | Falha em `fallen` de `retain_s0` e `fall` de `retain_s1` | `false` |
| `other_walk_fp` | Diferença dos FP combinados contra A ≤ 0 em cada dobra | +380 em `retain_s1` | `false` |

As salvaguardas por dobra, sempre `dynamic − A`, foram:

| Dobra | Δ revocação `fall` | Δ revocação `fallen` | FP `other` + `walk`: A | FP `other` + `walk`: dinâmico | Δ FP |
| --- | --- | --- | --- | --- | --- |
| `retain_s0` | 0.018027571580063628 | -0.06727272727272726 | 1829 | 1533 | -296 |
| `retain_s1` | -0.37744034707158347 | -0.040752351097178785 | 355 | 735 | 380 |
| `retain_s3-4-6-8` | 0.02037351443123936 | 0.09530201342281885 | 990 | 836 | -154 |

O relatório registra `status: complete`, `window_criteria_met: false`,
`advance: false` e `promotion: false`. Também registra
`candidate_promoted: false`, `independent_validation: false` e
`observed_at_or_above_0_800: []`. Sensibilidade de eventos de queda, FA/h e
latência permanecem `null`, com status
`unavailable_without_validated_fold_event_evaluator`; não há prova de
preservação de eventos.

### Conclusão e encerramento

O RGB dinâmico superou o RGB estático nas três dobras CS de desenvolvimento,
com sinal temporal segundo os dois critérios correspondentes. Entretanto,
ficou abaixo do baseline A em todas as dobras e falhou no ganho médio,
no número de ganhos contra A e nas salvaguardas de revocação de
`fall`/`fallen` e de falsos positivos de `other`/`walk`. Portanto, o piloto
**não justifica execução CV, seeds adicionais, ajuste de hiperparâmetros,
promoção ou avaliação no teste oficial**. A investigação fica encerrada
na branch `analysis/gatefall-improvements`, sem proposta de incorporar a
implementação experimental em `main`.

O experimento canônico A/B/C, suas referências e os resultados do
manuscrito permanecem inalterados. Caches, pesos, checkpoints, controles
e relatórios experimentais locais são preservados. Este encerramento
altera apenas esta documentação; não altera treino, contratos de
avaliação ou documentos no Google Drive.

Três grupos e uma seed fornecem apenas evidência
exploratória. As janelas se sobrepõem, o `test` já foi conhecido por estudos
anteriores, e a comparação RGB versus A muda representação e arquitetura.
O controle estático isola a contribuição do histórico dentro da sonda RGB.
Os 16 quadros cobrem a janela Le2i de 2,4 s, com padding no início, uma
escala temporal diferente dos clipes de 16 quadros a 15 fps descritos na
avaliação Kinetics do fornecedor. O preprocessing oficial não elimina essa
diferença de domínio.
