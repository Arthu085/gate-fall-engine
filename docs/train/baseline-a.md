# Treino — Braço A (TCN sobre pose)

`src/gatefall/train/` implementa o treino do braço A: uma TCN causal dilatada
consumindo o vetor de 134 features de pose descrito em [Contrato
temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose), já
padronizado por [`apply_standardization`](../data/pose-standardization.md)
fora de `PoseWindowDataset` — nem `kinematics.py`, nem `pose_dataset.py`, nem
o JSON de estatísticas fazem parte deste módulo.

## Como executar

```bash
uv run python -m gatefall.train.baseline_a selftest
```

Roda checagens sintéticas da arquitetura da TCN (`tcn_selftest.py`), das
métricas restritas (`metrics_selftest.py`) e das guardas de determinismo de
GPU (`engine_selftest.py`), sem treinar nem tocar no dataset real.

```bash
uv run python -m gatefall.train.baseline_a train --dataset le2i \
  --run-dir runs/local/le2i/baseline_a
```

Treina o braço A sobre o Le2i real. O destino padrão é
`runs/local/le2i/baseline_a/`; `--run-dir` permite explicitar outro diretório
local. Destinos dentro de `runs/reference/` são rejeitados, inclusive quando o
comando é chamado fora da raiz do repositório.

Um run completo exige `config.yaml`, `metrics.json` e `checkpoint.pt` válidos e
coerentes. O treino escreve os três artefatos em um diretório de staging irmão
do destino, registra hashes de configuração e checkpoint nas métricas, valida
o conjunto e só então o publica com `os.replace`. Sem `--force`, um run completo
é preservado e um run parcial ou inconsistente falha informando o artefato
inválido ou ausente. Com `--force`, o run local anterior é movido para um
backup irmão; se a promoção do staging falhar, o backup é restaurado. Depois de
uma promoção bem-sucedida, o backup é removido. Esse lifecycle do treino não
usa lock nem journal e `--force` não torna a referência gravável.

## Arquitetura

TCN causal com convoluções dilatadas, `kernel_size=3`, dilatações
`[1, 2, 4]` e três blocos de canais `[32, 32, 32]`, `dropout=0.3`. O campo
receptivo resultante é 29 quadros, maior que `WINDOW_FRAMES=24` — a rede
enxerga a janela inteira em pelo menos um caminho de convolução. A cabeça é
many-to-one: classifica sobre `NUM_CLASSES=10`, tomando apenas a saída do
último timestep da janela.

## Receita de treino congelada

A receita abaixo é compartilhada, sem alteração, pelos braços B e C
(`CLAUDE.md`, invariante 1 — só o vetor de feature por passo muda entre A,
B e C):

- Seed 42.
- Otimizador AdamW, `lr=1e-3`, `weight_decay=1e-2`.
- Agendamento de learning rate cosseno.
- `batch_size=64`, 30 épocas, sem early stopping.
- `CrossEntropyLoss` ponderada por frequência inversa das classes.
- Clipping de gradiente por norma, limite 1.0.
- Determinismo de GPU: `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `torch.use_deterministic_algorithms(True)` e `CUBLAS_WORKSPACE_CONFIG=:4096:8` —
  retreinos com a mesma seed produzem checkpoint idêntico na mesma
  máquina/GPU/driver/cuDNN. `runs/reference/le2i/baseline_a` já foi
  regenerado sob esse regime determinístico (ver "Migração de referência:
  determinismo de GPU" abaixo). Ver [investigação de determinismo de
  GPU](gpu-determinism.md) para o diagnóstico completo.

As 30 épocas são um orçamento fixo pré-registrado, definido antes de rodar
o treino, não um resultado de monitorar `val_macro_f1_restricted` e parar
quando ela parecesse boa. A ausência de early stopping é intencional: fixar
o orçamento de antemão evita escolher a época com base em uma métrica de
validação pequena e enviesada (ver "Seleção de checkpoint" abaixo), o que
inflaria artificialmente o desempenho reportado.

## Seleção de checkpoint: sempre a última época

O checkpoint salvo é sempre o peso da última época, nunca o de melhor
`val_macro_f1_restricted`. O split `val` do Le2i cobre 19 vídeos,
concentrados em apenas três ambientes (`Coffee_room_01`, `Home_01` e
`Home_02`) — então escolher checkpoint por essa métrica otimizaria para o
ruído desse subconjunto pequeno em vez de generalização real.

## Métrica: macro-F1 restrita

Implementada em NumPy puro em `gatefall/train/metrics.py`, sem adicionar
scikit-learn como dependência. A macro-F1 é calculada apenas sobre as
classes `{0, 1, 2, 3, 4, 7, 8, 9}`, excluindo:

- `5` (`lie_down`): suporte quase nulo no Le2i.
- `6` (`lying`): nunca ocorre no Le2i.

Incluir essas duas classes no macro-F1 faria o denominador da média ser
dominado por F1 indefinido ou instável sobre poucas ou nenhuma amostra,
distorcendo a métrica agregada sem refletir desempenho real do modelo.

## Artefatos locais e referência histórica

`runs/reference/le2i/baseline_a/config.yaml` e `metrics.json` são evidência
histórica versionada. Desde a migração de referência para o regime
determinístico de GPU (ver "Migração de referência: determinismo de GPU"
abaixo), o conteúdo desses arquivos é a saída real e regenerada do run
determinístico, não apenas conteúdo movido de um caminho legado. O
checkpoint não é versionado. Uma reprodução grava os três artefatos em
`runs/local/le2i/baseline_a/`, ignorado pelo Git.

`config.yaml` grava a configuração completa da execução: identificação do
run e do braço, `seed`, dimensão de entrada, `window_frames`, stride de
treino e de avaliação, número de classes, hiperparâmetros da TCN
(`kernel_size`, `dilations`, `channels`, `dropout`, `receptive_field`),
hiperparâmetros de otimização (`optimizer_name`, `lr`, `weight_decay`,
`grad_clip_norm`, `lr_schedule_name`, `batch_size`, `epochs`), a loss
(`loss_name`, `class_weighted`) e a proveniência das estatísticas de
padronização usadas (`standardization_stats_path` e
`standardization_stats_sha256`, o hash do JSON no momento do treino).

`metrics.json` grava `epochs_trained`, `device`, `torch_version`, o
histórico por época (`train_loss` e `val_macro_f1_restricted`), o bloco
`final` com `macro_f1_restricted`, `f1_by_class` e `support` por split
(`train`, `val`, `test`), e as listas `restricted_classes` /
`excluded_classes`.

## Resultado da execução real

Execução registrada em `runs/reference/le2i/baseline_a/metrics.json`, checkpoint na
última época (30):

| Split | Macro-F1 restrita |
| ----- | ------------------ |
| Treino | 0,8565 |
| Validação | 0,6656 |
| Teste | 0,6201 |

A queda de treino para validação/teste é esperada: o split de validação é
pequeno e enviesado (19 vídeos concentrados em três ambientes, ver acima) e
o split le2i-cs é cross-subject — treino e teste cobrem os mesmos seis
ambientes do Le2i, mudam apenas os subjects — então a queda para teste
reflete subjects não vistos no treino, não ambientes novos. O split é por
vídeo/subject justamente para evitar vazamento entre treino e teste
(`CLAUDE.md`, invariante 2).

## Migração de referência: determinismo de GPU

Antes da correção de determinismo de GPU (PR #35), a referência mantinha um
run treinado sem as guardas corrigidas de `CUBLAS_WORKSPACE_CONFIG` e
determinismo.

Referência antiga (mantida apenas como registro histórico, não reprodutível
sob as guardas atuais): macro-F1 de teste 0,6212 (0.6211639593563492); 12/13
eventos de queda detectados em validação; 21/22 em teste; 10 falsos alarmes
em teste.

Referência nova (esta migração): macro-F1 de teste 0,6201 (0.6201067209256219);
13/13 em validação; 20/22 em teste; 12 falsos alarmes em teste. Checkpoint
sha256 `264f4997f0875881f35e20f64a370c955b3488759d5f3714816c6771f12f0ff7`,
reproduzido de forma idêntica em retreinos independentes sob as guardas de
determinismo corrigidas (ver [determinismo de GPU](gpu-determinism.md)).

Esta é uma migração metodológica de referência, para reprodutibilidade sob
o regime determinístico congelado e compartilhado com os braços B e C
(`CLAUDE.md`, invariante 1) — não uma seleção do checkpoint de melhor
desempenho. O novo macro-F1 de teste (0,6201) não é maior que o antigo
(0,6212), o que é evidência contra cherry-picking.

## Limitações

Nesta execução de seed único, `val_macro_f1_restricted` não é monotônica
ao longo do treino: pelo histórico em `runs/reference/le2i/baseline_a/metrics.json`, ela
atinge um pico de 0,6686 na época 27 e termina em 0,6656 na época 30 (o
checkpoint salvo, ver "Seleção de checkpoint" acima). Como o orçamento de
30 épocas é fixo e pré-registrado (ver "Receita de treino congelada"), o
checkpoint final não é o de melhor macro-F1 de validação observada — a
diferença entre pico e final (~0,0030) é bem menor que na execução anterior
(~0,0249), mas segue sendo um lembrete de que essa métrica de validação é
ruidosa (19 vídeos, três ambientes) e não deve ser lida como uma curva
estável.
