# Índice de qualidade de pose (`q_pose`)

`src/gatefall/pose/quality.py` calcula, por quadro, um índice de qualidade da
pose extraída — `q_pose` — a partir dos dados brutos de
`gatefall.pose.loading.load_pose`. É um diagnóstico de relatório, no mesmo
espírito de `gatefall.eval.qualitative` e `gatefall.pose.smoke`: **não é um
estágio do pipeline de 26 estágios**, não entra no vetor de 134 features do
braço A nem em nenhum dos três braços do experimento, e nada no schema HDF5,
na padronização ou no treino muda por causa dele.

`q_pose` é um índice normalizado de confiabilidade em `[0, 1]`, **não uma
probabilidade calibrada**: mede o quanto a detecção de pose de um quadro
parece confiável (confiança do detector, cobertura de keypoints e
consistência estrutural com o quadro observado anterior), sem qualquer
garantia de calibração estatística.

## Por que lê `load_pose` cru, sem `impute_missing`

O cálculo lê `keypoints`, `bbox` e `person_found` direto de `load_pose`, e
deliberadamente nunca passa pelo forward-fill de `impute_missing`. Um quadro
preenchido por forward-fill precisa continuar marcado como ausente aqui — do
contrário o componente temporal compararia um quadro consigo mesmo via
preenchimento, em vez de com uma observação real, e ficaria espuriamente
perfeito exatamente nos trechos com pior detecção.

## Fórmula

Para cada quadro `t`, com confiança clipada
`conf_clipped = clip(conf, 0.0, 1.0)`:

- `q_conf[t]` = média das 17 confianças clipadas dos keypoints do quadro.
- `q_valid[t]` = fração dos 17 keypoints com confiança clipada
  `>= VALID_CONF_THRESHOLD` (limiar inclusivo).
- `q_temporal[t]` = `1 - mediana(mudança_relativa)` sobre as arestas do
  esqueleto COCO17 (`gatefall.pose.kinematics.COCO17_SKELETON_EDGES`) comuns
  entre o quadro `t` e o último quadro OBSERVADO anterior (`person_found ==
  True`), com coordenadas normalizadas pela diagonal da bbox
  (`gatefall.pose.loading.normalize_keypoints`). Uma aresta só entra na
  mediana se ambos os keypoints tiverem confiança clipada
  `>= VALID_CONF_THRESHOLD` nos dois quadros comparados. Para cada aresta:

  ```
  mudança_relativa = |L_t - L_prev| / max(L_t, L_prev, TEMPORAL_LENGTH_EPS)
  ```

- `q_pose[t] = q_conf[t] * q_valid[t] * q_temporal[t]`.

Se `person_found[t] == False`, os quatro componentes (`q_conf`, `q_valid`,
`q_temporal`, `q_pose`) são exatamente `0.0` para o quadro `t` — sem exceção,
mesmo que `keypoints`/`bbox` contenham lixo residual no array bruto.

### Constantes

| Constante | Valor | Papel |
| --- | --- | --- |
| `VALID_CONF_THRESHOLD` | `0.5` | Limiar inclusivo de confiança para um keypoint contar em `q_valid` e para uma aresta ser usada em `q_temporal` |
| `MIN_VALID_EDGES` | `3` | Número mínimo de arestas comuns entre os dois quadros para calcular `q_temporal`; abaixo disso, neutro |
| `TEMPORAL_LENGTH_EPS` | `1e-6` | Piso do denominador da mudança relativa, evita divisão por comprimento de aresta ~0 |

### Casos de borda de `q_temporal`

`q_temporal[t] = 1.0` (neutro) quando:

- não existe nenhum quadro observado anterior (primeiro quadro observado da
  sequência, ou todo predecessor tem `person_found == False`); ou
- existem menos que `MIN_VALID_EDGES` arestas em comum, com confiança
  suficiente nos dois lados, entre o quadro atual e o último observado.

Nos dois casos o neutro é intencional: na ausência de evidência suficiente
para medir mudança estrutural, `q_temporal` não penaliza o quadro, mesmo que
a corrupção real seja grande (o selftest cobre exatamente esse caso com poucas
arestas válidas e escala 3x).

## Não é uma probabilidade calibrada

`q_pose` combina três sinais heterogêneos por multiplicação; a escala
resultante não tem interpretação estatística (não é uma probabilidade
posterior, nem uma taxa de erro esperada). Ele serve para comparação relativa
entre quadros e vídeos — por exemplo, para localizar trechos de detecção
ruim — não para decidir um limiar de aceitação calibrado sem validação
adicional.

## `selftest` e `degradation`: sintéticos e determinísticos

`src/gatefall/pose/quality_selftest.py` cobre dois níveis, sempre com dados
sintéticos e seed fixa (`SEED = 20260916`), sem tocar no dataset real:

- `run_selftest` trava a semântica unitária dos quatro componentes: limites
  de `clip_confidence`, `q_conf` como média, limiar inclusivo de `q_valid`,
  `q_temporal` sob escala radial conhecida, o produto
  `q_pose == q_conf * q_valid * q_temporal`, pose estática perfeita dando
  `q_pose == 1.0`, os dois casos neutros de `q_temporal`, o caso de gap
  forward-fillable (comparação contra o último quadro observado, não contra
  o preenchimento) e o zero exato em quadros ausentes.
- `run_degradation` roda uma varredura determinística de severidade
  crescente em quatro canais isolados — atenuação de confiança, dropout de
  keypoints, corrupção estrutural temporal e ausência completa — e verifica
  que o componente-alvo e `q_pose` nunca crescem com o aumento de severidade.

Essa validação de degradação é inteiramente sintética: não há varredura de
corrupção controlada sobre o dado real do Le2i.

## `report`: diagnóstico descritivo sobre o dataset real

`uv run python -m gatefall.pose.quality report --dataset le2i` calcula
`q_pose` sobre todos os vídeos da grade e imprime, por split, os percentis
(p1/p5/p25/p50/p75/p95/p99) dos quatro componentes, além de checagens
fatais: os quatro componentes finitos e limitados a `[0,1]`; exatamente
`0.0` em todo quadro `person_found == False`; total de linhas igual a
`EXPECTED_K_SUM`; e nenhuma linha com `split` nulo após o merge com a grade.

As seções por split (`train`/`val`/`test`) são **diagnósticas e
descritivas** — nenhuma alegação de validação deve se apoiar nelas, e
evidência de degradação (a varredura de severidade monotônica) é exclusiva
do `run_degradation` sintético acima, nunca de uma varredura de corrupção
sobre `val`/`test`.

## Como executar

```bash
uv run python -m gatefall.pose.quality selftest
```

Roda `run_selftest`: casos unitários sintéticos, sem tocar no dataset real.

```bash
uv run python -m gatefall.pose.quality degradation
```

Roda `run_degradation`: varredura determinística de severidade sobre os
quatro canais de corrupção sintética.

```bash
uv run python -m gatefall.pose.quality report --dataset le2i
```

Roda `q_pose` sobre a grade real do Le2i e imprime as checagens e os
percentis por split descritos acima.
