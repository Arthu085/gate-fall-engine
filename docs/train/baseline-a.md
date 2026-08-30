# Treino — Arma A (TCN sobre pose)

`src/gatefall/train/` implementa o treino da arma A: uma TCN causal dilatada
consumindo o vetor de 134 features de pose descrito em [Contrato
temporal](../data/temporal-contract.md#dataset-de-janelas-de-pose), já
padronizado por [`apply_standardization`](../data/pose-standardization.md)
fora de `PoseWindowDataset` — nem `kinematics.py`, nem `pose_dataset.py`, nem
o JSON de estatísticas fazem parte deste módulo.

## Como executar

```bash
uv run python -m gatefall.train.baseline_a selftest
```

Roda checagens sintéticas da arquitetura da TCN (`tcn_selftest.py`) e das
métricas restritas (`metrics_selftest.py`), sem treinar nem tocar no dataset
real.

```bash
uv run python -m gatefall.train.baseline_a train [--force]
```

Treina a arma A sobre o Le2i real e grava `config.yaml`, `metrics.json` e
`checkpoint.pt` em `runs/baseline_a/`. Sem `--force`, uma segunda execução
não sobrescreve um `run_dir` já existente.

## Arquitetura

TCN causal com convoluções dilatadas, `kernel_size=3`, dilatações
`[1, 2, 4]` e três blocos de canais `[32, 32, 32]`, `dropout=0.3`. O campo
receptivo resultante é 29 quadros, maior que `WINDOW_FRAMES=24` — a rede
enxerga a janela inteira em pelo menos um caminho de convolução. A cabeça é
many-to-one: classifica sobre `NUM_CLASSES=10`, tomando apenas a saída do
último timestep da janela.

## Receita de treino congelada

A receita abaixo é compartilhada, sem alteração, pelas armas B e C
(`CLAUDE.md`, invariante 1 — só o vetor de feature por passo muda entre A,
B e C):

- Seed 42.
- Otimizador AdamW, `lr=1e-3`, `weight_decay=1e-2`.
- Agendamento de learning rate cosseno.
- `batch_size=64`, 30 épocas, sem early stopping.
- `CrossEntropyLoss` ponderada por frequência inversa das classes.
- Clipping de gradiente por norma, limite 1.0.

## Seleção de checkpoint: sempre a última época

O checkpoint salvo é sempre o peso da última época, nunca o de melhor
`val_macro_f1_restricted`. O split `val` do Le2i cobre apenas
`Coffee_room_01`, `Home_01` e `Home_02` — poucos vídeos e enviesados —
então escolher checkpoint por essa métrica otimizaria para o ruído desse
subconjunto pequeno em vez de generalização real.

## Métrica: macro-F1 restrita

Implementada em NumPy puro em `gatefall/train/metrics.py`, sem adicionar
scikit-learn como dependência. A macro-F1 é calculada apenas sobre as
classes `{0, 1, 2, 3, 4, 7, 8, 9}`, excluindo:

- `5` (`lie_down`): suporte quase nulo no Le2i.
- `6` (`lying`): nunca ocorre no Le2i.

Incluir essas duas classes no macro-F1 faria o denominador da média ser
dominado por F1 indefinido ou instável sobre poucas ou nenhuma amostra,
distorcendo a métrica agregada sem refletir desempenho real do modelo.

## Schema de `runs/baseline_a/config.yaml` e `metrics.json`

`runs/baseline_a/config.yaml` e `runs/baseline_a/metrics.json` são
versionados no Git como resultado reproduzível do treino (ver `.gitignore`);
`checkpoint.pt` permanece fora do versionamento pela regra global de
`*.pt`.

`config.yaml` grava a configuração completa da execução: identificação do
run e da arma, `seed`, dimensão de entrada, `window_frames`, stride de
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

Execução registrada em `runs/baseline_a/metrics.json`, checkpoint na
última época (30):

| Split | Macro-F1 restrita |
| ----- | ------------------ |
| Treino | 0,8589 |
| Validação | 0,6558 |
| Teste | 0,6212 |

A queda de treino para validação/teste é esperada: o split de validação é
pequeno e enviesado (três vídeos, ver acima) e o de teste cobre ambientes
não vistos no treino, testando a generalização real do split por vídeo
(`CLAUDE.md`, invariante 2).
