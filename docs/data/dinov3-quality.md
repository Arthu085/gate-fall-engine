# Qualidade visual DINOv3

`gatefall.dinov3.quality` calcula `q_visual`, uma proxy operacional de qualidade
visual por quadro. O índice é causal, determinístico e limitado a `[0, 1]`, mas
**não é uma confiança nem uma probabilidade calibrada**. Seu uso previsto é
avaliar posteriormente a fusão adaptativa; esta etapa não implementa B1, gating
nem SAM 3 e não fecha o PEND-013.

## Entrada e fórmula

O cálculo recebe somente o quadro RGB atual, redimensionado para `224 × 224` e
convertido para `[0, 1]`, antes da normalização do DINOv3. Não usa profundidade,
outros quadros, rótulos, features do backbone ou estatísticas do batch.

Para a luminância Rec. 709

\[
Y = 0{,}2126R + 0{,}7152G + 0{,}0722B,
\]

os componentes são:

- `q_exposure`: produto da fração não clipada por `4m(1-m)`, em que pixels
  escuros satisfazem `Y <= 1/255`, pixels claros satisfazem `Y >= 254/255` e
  `m` é a mediana da luminância;
- `q_contrast`: diferença `p95(Y) - p5(Y)`;
- `q_sharpness`: média do valor absoluto do Laplaciano de quatro vizinhos,
  dividida por `4` e normalizada pelo contraste, com proteção numérica.

Todos os componentes são limitados a `[0, 1]`. O índice final é a média
geométrica:

\[
q_{visual} = \sqrt[3]{q_{exposure}q_{contrast}q_{sharpness}}.
\]

## Validação controlada

O comando `validate` mede a resposta de `q_visual` e a similaridade cosseno do
descritor DINOv3 entre o quadro limpo e degradações controladas. Ele usa somente
`train` e `val`; `test` nunca participa. Para cada classe e split, seleciona
deterministicamente até oito eventos, toma o quadro central de cada sequência
contígua de rótulo e ordena candidatos por SHA-256. O rótulo `IGNORE` participa
da distribuição limpa, mas é excluído da seleção para os sweeps.

Os sweeps têm cinco níveis:

| Degradação | Severidades |
| --- | --- |
| Box blur | raios `0, 2, 3, 6, 12` (kernels `1, 5, 7, 13, 25`) |
| Subexposição | ganhos `1, 0.75, 0.5, 0.25, 0.125` |
| Sobre-exposição | ganhos `1, 0.75, 0.5, 0.25, 0.125` em relação ao branco |
| Contraste | ganhos `1, 0.75, 0.5, 0.25, 0.125` em torno da mediana |

Execução de referência:

```bash
uv run python -m gatefall.dinov3.quality validate \
  --dataset le2i \
  --events-per-class 8 \
  --batch-size 8 \
  --repo-dir data/scratch/dinov3_repo \
  --weights data/scratch/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

O relatório JSON enviado para stdout inclui o hash e as contagens da seleção,
distribuições limpas por quadro e evento, distribuições por severidade,
monotonicidade e correlações entre `q_visual` e a similaridade do descritor.

### Resultado de referência no Le2i

A execução em `train` e `val` selecionou 120 eventos, com hash
`c75fdd70087975cc7c3739c15a9c2a5a00439aadcc765ef6e2009e8d7016c9de`.
Algumas classes têm menos de oito eventos elegíveis; em `val`, por exemplo, o
rótulo `5` tem um evento (cinco quadros), enquanto `-1` tem um evento na
distribuição limpa, mas é excluído do sweep.

| Degradação | Fração não crescente | Pearson intraevento mediana [Q25, Q75] | Spearman intraevento mediana [Q25, Q75] |
| --- | ---: | ---: | ---: |
| Blur | 1.000000 | 0.8202 [0.7866, 0.8550] | 1.0000 [1.0000, 1.0000] |
| Contraste | 1.000000 | 0.8848 [0.8730, 0.9113] | 1.0000 [1.0000, 1.0000] |
| Sobre-exposição | 0.995833 | 0.9522 [0.9444, 0.9579] | 1.0000 [1.0000, 1.0000] |
| Subexposição | 1.000000 | 0.8859 [0.8732, 0.9066] | 1.0000 [1.0000, 1.0000] |

A fração de correlações positivas foi `1.0` para Pearson e Spearman em todas
as degradações. As medianas limpas de `q_visual` por classe ficaram em cerca de
`0.2180–0.2225` em `train` e `0.2184–0.2256` em `val`. Agregadas no sweep, as
medianas nos extremos foram:

| Condição | `q_visual` mediana | Similaridade mediana |
| --- | ---: | ---: |
| Limpa | 0.2210 | 1.000 |
| Blur, raio 12 | 0.0690 | 0.170 |
| Contraste, ganho 0.125 | 0.1120 | 0.899 |
| Sobre-exposição, ganho 0.125 | 0.0730 | 0.771 |
| Subexposição, ganho 0.125 | 0.0645 | 0.925 |

Esses resultados sustentam `q_visual` como proxy operacional defensável para
uma avaliação posterior do gating. Eles não demonstram calibração probabilística
nem validam uma política de fusão. O selftest sintético da CI verifica fórmula,
limites, causalidade, determinismo, sweeps, seleção e agregações, mas não usa o
dataset real nem os pesos e, portanto, **não fecha o PEND-013**.

## Limitações

- Ruído e artefatos de alta frequência podem parecer nitidez ao Laplaciano.
- Cenas legitimamente escuras ou de baixo contraste podem receber índice baixo.
- Conteúdo semântico pode confundir a relação entre a proxy e a estabilidade do
  descritor.
- Os resultados são evidência diagnóstica no Le2i, não um limiar de gating nem
  uma garantia de generalização.

## Selftest

```bash
uv run python -m gatefall.dinov3.quality selftest
```

O selftest não acessa vídeos, dataset, repositório DINOv3, pesos ou GPU.
