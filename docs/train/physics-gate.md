# Experimento de consistência e provas Lean

O módulo `gatefall.experiments.physics` permite testar a hipótese de que
confiabilidade observável e falhas pareadas ajudam a fusão. Ele é opt-in:
os comandos históricos A/B0/B1/C0/C1 e suas referências continuam disponíveis.
Os resultados desta implementação são testes sintéticos, não novos resultados Le2i.

## O que muda

- Derivadas usam observações válidas, intervalos reais e tempos médios das
  velocidades. Um gap longo reinicia o histórico. Ausência não é velocidade zero.
- A pose passa de 134 para 138 canais: acrescenta presença, validade da velocidade,
  validade da aceleração e idade normalizada da última observação. Todos os braços
  experimentais recebem o mesmo esquema.
- A consistência de pose mede deformação residual após alinhamento por
  translação, rotação e escala. SAM 3 usa variação relativa de área/preenchimento.
  DINOv3 não tem residual físico visual disponível.
- Gate `physics`: `sigmoid(b + wp*qp - wv*qv - ap*rp*vp + av*rv*vv)`.
  Os quatro coeficientes são `softplus` de parâmetros aprendidos; `vp/vv` indicam
  validade dos resíduos. A saída multiplica pose; seu complemento multiplica visual.
- Falhas são ausências em blocos, com reconstrução das features, confiança e
  disponibilidade antes da padronização. São persistidas somente no treino.
- A perda opcional exige margem positiva entre o gate limpo e o degradado,
  além da classificação limpa/corrompida. Pares sem alteração ou sem alternativa
  disponível não contribuem à margem.
- A seleção usa o melhor macro-F1 restrito de validação; empates favorecem a
  primeira época. `last.pt` preserva a última época. O teste é uma chamada separada.

## Preparar, treinar e avaliar

Execute a extração offline das fontes conforme os runbooks existentes. Os
backbones permanecem congelados; o runner não baixa pesos nem extrai imagens.
Exemplo para SAM 3, a partir da raiz do repositório:

```bash
uv sync --locked --dev
uv run python -m gatefall.experiments.physics selftest
uv run python -m gatefall.experiments.physics prepare --dataset le2i --visual sam3 --output data/processed/physics-sam3-s42 --seed 42
uv run python -m gatefall.experiments.physics train --bundle data/processed/physics-sam3-s42 --run-dir runs/local/le2i/physics/C1-s42 --arm C1 --fusion physics --seed 42 --epochs 30 --augment --order-weight 0.1 --order-margin 0.05
uv run python -m gatefall.experiments.physics evaluate --bundle data/processed/physics-sam3-s42 --run-dir runs/local/le2i/physics/C1-s42 --split test
```

Para B0/B1, prepare `--visual dinov3`; para pose isolada, `--visual none`.
O braço A também pode usar o mesmo pacote visual dos controles, ignorando essa
modalidade. Para Le2i-CV, selecione `--dataset le2i-cv` e caminhos locais próprios.
Destinos existentes são recusados; não reutilize um diretório para outra receita.

Os parâmetros de preparação são `--max-gap-seconds 0.5`,
`--outage-probability 0.05` e `--outage-frames 3`. A probabilidade corresponde
ao início de cada bloco, portanto não é a fração final de frames ausentes.

## Matriz das comparações

| Braço | `--fusion` | Pergunta |
| --- | --- | --- |
| A | `pose` | Qual o efeito da cinemática corrigida? |
| B0/C0 | `concat` | A modalidade visual ajuda sem gate? |
| B1/C1 | `constant` | Um peso fixo de 0,5 já explica o ganho? |
| B1/C1 | `quality_concat` | A evidência como entrada já é suficiente? |
| B1/C1 | `learned` | Como se comporta o gate livre com duas qualidades? |
| B1/C1 | `monotonic` | Os sinais restritos ajudam sem resíduos? |
| B1/C1 | `physics` | A consistência acrescenta informação útil? |

Compare primeiro todos sem aumento e sem perda de ordem. Depois compare
`concat`, `learned`, `monotonic` e `physics` com `--augment`; acrescente
`--order-weight 0.1` somente aos gates treináveis. Isso separa efeito da
exposição às falhas e efeito da regularização. Os pesos/margem são valores
iniciais de experimento; qualquer ajuste usa somente treino/validação.

Use as sementes 42–46, as mesmas fontes e partições, 30 épocas e a mesma política
de checkpoint por comparação. Para cada semente, prepare as falhas uma vez e
reutilize o pacote. A TCN conserva janela 24 a 10 Hz, canais 32/32/32, kernel 3,
dilatações 1/2/4 e dropout 0,3. Os controles têm contagens de parâmetros diferentes,
registradas em `config.json`. `--checkpoint-policy last` permite uma ablação
explícita da seleção, aplicada igualmente aos braços.

## Artefatos e auditoria

Cada pacote possui `manifest.json` e um HDF5 por vídeo com hashes das fontes e
partições. Estatísticas são ajustadas somente nos frames válidos de treino,
antes de formar janelas. Validação/teste e corrupções reutilizam essas estatísticas.

O run contém `config.json`, `stats.json`, `metrics.json`, `checkpoint.pt` e
`last.pt`. A avaliação gera `evaluation-test.json` com classificação, eventos,
FA/h, latências em precisão integral e predições por frame. Ela verifica hashes
e a origem das estatísticas antes de inferir. Preparação, treino e avaliação
registram revisão Git, estado modificado e hashes dos módulos Python.

## Garantias e limites

O diretório `formal/` contém provas Lean 4.19.0/Mathlib fixados e um README
que associa cada teorema ao código. Com Lean instalado, execute na pasta
`formal`: `lake exe cache get` e `lake build --wfail`. A CI executa o build
formal e os selftests Python. A integração usa a ação oficial
[leanprover/lean-action](https://github.com/leanprover/lean-action).

Lean prova fórmulas sobre números reais sob hipóteses explícitas. Não verifica
automaticamente o programa PyTorch, o conteúdo das imagens ou a qualidade
das predições. Em float32 a sigmoid pode saturar em 0 ou 1; os testes numéricos
exigem finitude e intervalo fechado. Os testes também verificam causalidade,
gaps, gradientes, corrupção pareada, isolamento das estatísticas, hashes e
treino/avaliação dos cinco braços.

O residual de pose tolera movimento global coerente, mas articulação legítima,
oclusão e perspectiva podem elevar seu valor. A qualidade `q_pose` herdada é
mantida igual em todos os controles e ainda inclui a penalização temporal
legada, sensível à geometria da caixa. Portanto a invariância do novo residual
não implica invariância do gate inteiro. Os resíduos são proxies, não
probabilidades calibradas ou leis de dinâmica 3D.

A avaliação nesta entrega é limpa. Demonstrar robustez exige posteriormente
uma avaliação de falhas pareadas em dados reservados, com o checkpoint e as
estatísticas já fixados; ela não deve participar da seleção. Blur real exige
reinferência das imagens, não ruído artificial em embeddings. Ganhos, sensibilidade
e FA/h reais dependem dos dados e do retreino ainda não executado nesta entrega.
