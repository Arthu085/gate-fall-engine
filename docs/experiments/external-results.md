# Resultados externos — gate físico e Motion Physics

!!! warning "Resultados externos, não canônicos"
    Os números desta página vêm de dois experimentos implementados pelo
    orientador fora deste repositório e executados localmente pelo autor do TCC.
    Eles foram importados aqui apenas para documentação e reprodutibilidade. Eles
    **não** substituem nem atualizam os resultados canônicos A/B0/B1/C0/C1, que
    continuam definidos pelos pipelines pose134, pelas referências em
    `runs/reference/` e pelas páginas de treino e avaliação de cada braço. A
    baseline externa **não** é a baseline original do GateFall.

Os módulos correspondentes são opt-in e só executam por invocação explícita
(`python -m gatefall.experiments.physics` e `python -m gatefall.experiments.motion`).
A descrição dos métodos está em
[Experimento de consistência e provas Lean](../train/physics-gate.md) e em
[Aprendizado de movimento e coerência cinemática](motion-physics.md).

## Proveniência

A implementação e a execução têm autorias distintas:

- **Implementação (orientador):** código, especificações, provas Lean e
  documentação de método nos repositórios e commits abaixo.
- **Execução e validação histórica (autor do TCC):** preparação dos bundles,
  treinos, avaliações de teste, auditorias e análises retrospectivas, executados
  localmente em clones desses commits. As saídas foram consolidadas nos registros
  científicos do projeto no Google Drive: `RESULTS_CONTEXT.md` (seção
  "Resultados exploratórios externos", snapshot de 21/10/2026) e
  `VALIDATION_HISTORY_2026-10.md` (entradas `VAL-EXT-PHYSICS-FORK-2026-10-10` e
  `VAL-EXT-MOTION-PHYSICS-2026-10-14`).

| Item | Gate físico (predecessor) | Motion Physics (extensão) |
| --- | --- | --- |
| Repositório de origem | [rsramos30/gate-fall-engine](https://github.com/rsramos30/gate-fall-engine) | [rsramos30/gate-fall-motion-physics](https://github.com/rsramos30/gate-fall-motion-physics) |
| Commit | `301b32e42e47f29fed1daf607948d11ddf3b4e5d` | `5f6e2ae4ce0622b6add05da207d8a20d29929159` |
| Base | `ef6b3c1` (merge do PR #100 neste repositório) | `301b32e`, mais `9fb8a8a` e `5f6e2ae` |
| Módulo | `gatefall.experiments.physics` | `gatefall.experiments.motion` (reutiliza `physics`) |
| Especificação | `specs/001-physics-informed-gate/` | `specs/002-motion-physics-learning/` |
| Execução local | 10/10/2026; análise retrospectiva em 11–12/10/2026 | 14/10/2026, CUDA em 150/150 |
| Avaliações de teste | 158 | 150 (de 150 treinos) |
| Revisão registrada nos runs | `301b32e` em todos os 158, `git_dirty=true` | `5f6e2ae` em todos os 150, `git_dirty=false` |

O código Python de `src/gatefall/` deste branch é idêntico byte a byte ao de
`5f6e2ae`, que por sua vez contém o de `301b32e` sem alterações em
`gatefall.experiments.physics`. Por isso os hashes de módulos registrados em
`code_provenance.python_sources_sha256` dos runs históricos continuam
verificáveis contra este repositório.

No predecessor, `git_dirty=true` decorre de arquivos não rastreados no clone
local (`data/processed/` e o script de lote `gatefall_lote_ab_final.sh`);
`git status --porcelain` os conta como modificação. Em todos os 158 `config.json`,
o hash de cada módulo Python registrado coincide com o arquivo correspondente
em `301b32e`, de modo que o código executado é o da revisão declarada.

Runs, checkpoints, bundles e logs ficaram nos clones locais
(`runs/local/le2i/` e `runs/local/le2i_cv/`, ignorados pelo Git) e não fazem
parte deste repositório. As tabelas abaixo transcrevem os registros científicos
citados. Antes da transcrição, foram conferidas contra esses artefatos locais,
sem nova inferência nem treino:

- as 26 linhas das matrizes limpas do predecessor, contra os 130
  `evaluation-test.json` limpos;
- as 72 linhas de semente 42 (matriz limpa e ablações), contra as avaliações
  correspondentes;
- os oito contrastes pareados entre fusões;
- os dez contrastes `motion_physics − motion`, contra os
  `summary/paired-deltas.csv`.

As tabelas absolutas da extensão vêm de
`runs/local/motion-physics-metricas-completas.csv`, com 30 linhas agregadas de
dez `summary/summary.json` e SHA-256
`cf4fc9756c14af26625afd682ed782cf926470e49d475e55a3aa020bb528c514`, também
conferido. A comparação `best_val` × `last` e os 40 bootstraps por sujeito
foram calculados em memória na análise retrospectiva, sem gravar novos
artefatos. Seus valores vêm somente dos registros e não foram reconferidos aqui.

## Diferenças de modelo e configuração em relação ao GateFall canônico

As duas linhas externas compartilham a mesma arquitetura, independente dos braços
canônicos:

| Aspecto | Canônico (A/B0/B1/C0/C1) | Externo (`physics` e `motion`) |
| --- | --- | --- |
| Vetor de pose | pose134 | pose138: acrescenta presença, validade da velocidade, validade da aceleração e idade normalizada da última observação |
| Derivadas de pose | Esquema histórico | Somente observações válidas, intervalos reais, gap longo (0,5 s) reinicia o histórico |
| Evidência do gate | `q_pose`, `q_visual` (2 entradas) | evidence8: `q_pose`, `q_visual`, `r_pose`, `r_visual`, `r_pose_valid`, `r_visual_valid`, `pose_present`, `visual_present` |
| Fusões | Concat (B0/C0) e gate aprendido (B1/C1) | `pose`, `concat`, `constant`, `quality_concat`, `learned`, `monotonic`, `physics` |
| Padronização | Estatísticas versionadas em `src/gatefall/features/stats/` | Ajustada nos frames válidos de treino de cada bundle, persistida em `stats.json` |
| Seleção de checkpoint | Checkpoint congelado da última época (30), em Le2i-CS e Le2i-CV; nenhuma seleção por validação ([braço A](../train/baseline-a.md)) | `best_val`: época de maior macro-F1 restrito de validação; `last.pt` também é salvo |
| Falhas sintéticas | Não | Opcional (`--augment`), somente no treino |
| Latência na avaliação | Arredondada pelo protocolo de alarme | Precisão integral (`round_latency=False`) |
| Parâmetros treináveis (C1, exemplo) | 67 885 | 69 455 (C1/`physics`) |

Fora dessas diferenças, as duas implementações usam a mesma receita de treino:
janela de 24 frames a 10 Hz, stride de treino 4, TCN 32/32/32 com kernel 3,
dilatações 1/2/4 e dropout 0,3, AdamW (lr 0,001, weight decay 0,01), cosseno,
entropia cruzada ponderada por classe, clipping 1,0, 30 épocas e batch 64. As
políticas de seleção de checkpoint são diferentes, conforme a tabela; os modelos
externos e canônicos não são selecionados da mesma forma. Os backbones
YOLO-Pose, DINOv3 e SAM 3 permanecem congelados, e as features são lidas dos
artefatos offline existentes.

A extensão Motion Physics acrescenta, só no treino, dois cabeçotes lineares
(posição futura e velocidade média de ombros e quadris, horizontes de 1 e 3
frames), a perda auxiliar mascarada `Lmov` e a regularização de coerência
cinemática 2D `Lcin`, com pesos 0,1 e 0,1 e escala espacial de 0,1 da
diagonal. Os alvos são preparados de forma causal num sidecar ligado por hash
ao bundle e à pose original. Na inferência só o cabeçote de classificação é
usado. As receitas são:

| Receita | `Lmov` | `Lcin` |
| --- | --- | --- |
| `baseline` | desligada | desligada |
| `motion` | ligada | desligada |
| `motion_physics` | ligada | ligada |

## Gate físico: 158 avaliações históricas

Todas as avaliações usam `pose138-valid-intervals-v1`, 30 épocas e `best_val`:

| Bloco | Configurações | Sementes | Protocolos | Avaliações |
| --- | --- | --- | --- | --- |
| Limpo (sem `--augment`, sem ordem) | A/`pose`; B0/`concat`; B1 e C1 × `constant`, `quality_concat`, `learned`, `monotonic`, `physics`; C0/`concat` (13) | 42–46 | Le2i-CS, Le2i-CV | 130 |
| Ablação `--augment` | B0/`concat`, C0/`concat`; B1 e C1 × `learned`, `monotonic`, `physics` (8) | 42 | Le2i-CS, Le2i-CV | 16 |
| Ablação `--augment --order-weight 0.1` | B1 e C1 × `learned`, `monotonic`, `physics` (6) | 42 | Le2i-CS, Le2i-CV | 12 |
| **Total** | | | | **158** |

As auditorias da execução verificaram suporte `(video_id, k_end, label)` e
protocolo de alarme idênticos entre modelos do mesmo protocolo (matriz limpa
CS 65/65, CV 65/65). A contagem é de avaliações, não de treinos novos; parte
dos runs foi reaproveitada entre lotes.

### Matriz limpa, sementes 42–46 (130 avaliações)

Média entre cinco sementes. "DP F1" é o desvio-padrão entre sementes, não
intervalo de confiança. Quedas detectadas e falsos alarmes são médias por
semente e podem ser fracionários.

#### Le2i-CS

| Braço | Fusão | Macro-F1 | DP F1 | Quedas detectadas | Falsos alarmes | FA/h | Latência média (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `pose` | 0,5941 | 0,0290 | 18,0/22 | 35,8 | 208,9494 | 0,53 |
| B0 | `concat` | 0,6396 | 0,0054 | 18,2/22 | 8,8 | 51,3619 | 0,34 |
| B1 | `constant` | 0,6488 | 0,0098 | 19,2/22 | 10,2 | 59,5331 | 0,34 |
| B1 | `quality_concat` | 0,6239 | 0,0379 | 19,2/22 | 9,6 | 56,0311 | 0,32 |
| B1 | `learned` | 0,6567 | 0,0198 | 20,4/22 | 10,4 | 60,7004 | 0,32 |
| B1 | `monotonic` | 0,6215 | 0,0329 | 19,0/22 | 12,4 | 72,3735 | 0,32 |
| B1 | `physics` | 0,6439 | 0,0121 | 19,6/22 | 11,8 | 68,8716 | 0,33 |
| C0 | `concat` | 0,6295 | 0,0157 | 21,0/22 | 13,0 | 75,8755 | 0,38 |
| C1 | `constant` | 0,6252 | 0,0115 | 20,2/22 | 18,0 | 105,0584 | 0,36 |
| C1 | `quality_concat` | 0,6273 | 0,0148 | 20,8/22 | 13,6 | 79,3774 | 0,37 |
| C1 | `learned` | 0,6278 | 0,0270 | 20,4/22 | 18,0 | 105,0584 | 0,37 |
| C1 | `monotonic` | 0,6292 | 0,0110 | 19,6/22 | 15,8 | 92,2179 | 0,37 |
| C1 | `physics` | 0,6332 | 0,0102 | 19,8/22 | 15,6 | 91,0506 | 0,38 |

#### Le2i-CV

| Braço | Fusão | Macro-F1 | DP F1 | Quedas detectadas | Falsos alarmes | FA/h | Latência média (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `pose` | 0,4539 | 0,0210 | 30,6/37 | 28,0 | 168,5055 | 0,49 |
| B0 | `concat` | 0,2970 | 0,0248 | 25,2/37 | 39,0 | 234,7041 | 0,81 |
| B1 | `constant` | 0,2878 | 0,0510 | 21,0/37 | 40,8 | 245,5366 | 0,82 |
| B1 | `quality_concat` | 0,3282 | 0,0507 | 19,0/37 | 35,0 | 210,6319 | 0,68 |
| B1 | `learned` | 0,3016 | 0,0723 | 28,8/37 | 26,4 | 158,8766 | 0,77 |
| B1 | `monotonic` | 0,2660 | 0,0476 | 19,0/37 | 47,0 | 282,8485 | 0,60 |
| B1 | `physics` | 0,2791 | 0,0414 | 25,0/37 | 30,4 | 182,9488 | 0,95 |
| C0 | `concat` | 0,4697 | 0,0194 | 30,0/37 | 15,4 | 92,6780 | 0,50 |
| C1 | `constant` | 0,4615 | 0,0207 | 30,6/37 | 16,0 | 96,2889 | 0,47 |
| C1 | `quality_concat` | 0,4647 | 0,0240 | 29,4/37 | 18,0 | 108,3250 | 0,46 |
| C1 | `learned` | 0,4775 | 0,0328 | 29,4/37 | 19,6 | 117,9539 | 0,45 |
| C1 | `monotonic` | 0,4706 | 0,0145 | 31,4/37 | 15,0 | 90,2708 | 0,46 |
| C1 | `physics` | 0,4650 | 0,0162 | 32,6/37 | 12,2 | 73,4203 | 0,48 |

Nessa matriz, o gate `physics` não tem o maior macro-F1 médio do braço B em
nenhum protocolo: as maiores médias são de `learned` em CS (0,6567) e de
`quality_concat` em CV (0,3282). No
braço C, `physics` tem o maior macro-F1 médio em CS (0,6332), mas detecta menos
quedas e emite mais falsos alarmes que C0 (19,8 contra 21,0 quedas, 15,6
contra 13,0 alarmes). Em CV, C1/`physics` tem o menor FA/h e o maior número de
quedas detectadas do braço C (73,4203 FA/h, 32,6/37), com macro-F1 abaixo de C0
e de `learned`. DINOv3 não tem residual visual físico. Em B, `physics` usa só
os sinais de pose e não comprova consistência visual.

### Ablações de semente 42 (28 avaliações)

Receitas com `--augment` e com `--augment --order-weight 0.1`, apenas na
semente 42. Os controles limpos da semente 42 estão no registro de origem.

| Protocolo | Braço | Fusão | Receita | Macro-F1 | Quedas detectadas | Falsos alarmes | FA/h |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Le2i-CS | B0 | `concat` | `aug` | 0,6258 | 19/22 | 10 | 58,3658 |
| Le2i-CS | B1 | `learned` | `aug` | 0,6640 | 20/22 | 12 | 70,0389 |
| Le2i-CS | B1 | `monotonic` | `aug` | 0,6528 | 22/22 | 6 | 35,0195 |
| Le2i-CS | B1 | `physics` | `aug` | 0,6138 | 18/22 | 13 | 75,8755 |
| Le2i-CS | B1 | `learned` | `aug+order` | 0,6743 | 20/22 | 9 | 52,5292 |
| Le2i-CS | B1 | `monotonic` | `aug+order` | 0,6470 | 18/22 | 9 | 52,5292 |
| Le2i-CS | B1 | `physics` | `aug+order` | 0,6574 | 20/22 | 11 | 64,2023 |
| Le2i-CS | C0 | `concat` | `aug` | 0,6498 | 20/22 | 15 | 87,5486 |
| Le2i-CS | C1 | `learned` | `aug` | 0,6318 | 19/22 | 24 | 140,0778 |
| Le2i-CS | C1 | `monotonic` | `aug` | 0,6293 | 20/22 | 16 | 93,3852 |
| Le2i-CS | C1 | `physics` | `aug` | 0,6531 | 20/22 | 18 | 105,0584 |
| Le2i-CS | C1 | `learned` | `aug+order` | 0,6346 | 18/22 | 36 | 210,1167 |
| Le2i-CS | C1 | `monotonic` | `aug+order` | 0,6241 | 20/22 | 18 | 105,0584 |
| Le2i-CS | C1 | `physics` | `aug+order` | 0,6125 | 17/22 | 24 | 140,0778 |
| Le2i-CV | B0 | `concat` | `aug` | 0,3550 | 25/37 | 21 | 126,3791 |
| Le2i-CV | B1 | `learned` | `aug` | 0,2581 | 23/37 | 32 | 192,5777 |
| Le2i-CV | B1 | `monotonic` | `aug` | 0,2587 | 25/37 | 24 | 144,4333 |
| Le2i-CV | B1 | `physics` | `aug` | 0,2881 | 20/37 | 42 | 252,7583 |
| Le2i-CV | B1 | `learned` | `aug+order` | 0,2440 | 22/37 | 41 | 246,7402 |
| Le2i-CV | B1 | `monotonic` | `aug+order` | 0,2770 | 19/37 | 31 | 186,5597 |
| Le2i-CV | B1 | `physics` | `aug+order` | 0,2944 | 24/37 | 24 | 144,4333 |
| Le2i-CV | C0 | `concat` | `aug` | 0,4774 | 34/37 | 11 | 66,1986 |
| Le2i-CV | C1 | `learned` | `aug` | 0,4755 | 33/37 | 13 | 78,2347 |
| Le2i-CV | C1 | `monotonic` | `aug` | 0,4715 | 31/37 | 17 | 102,3069 |
| Le2i-CV | C1 | `physics` | `aug` | 0,4225 | 32/37 | 15 | 90,2708 |
| Le2i-CV | C1 | `learned` | `aug+order` | 0,4728 | 20/37 | 31 | 186,5597 |
| Le2i-CV | C1 | `monotonic` | `aug+order` | 0,4466 | 32/37 | 14 | 84,2528 |
| Le2i-CV | C1 | `physics` | `aug+order` | 0,4213 | 32/37 | 15 | 90,2708 |

São resultados de uma semente, obtidos após a exploração do teste, e não
justificam trocar receita nem escolher hiperparâmetros. As falhas sintéticas
atuam só no treino; essas ablações não medem robustez a falhas na inferência.
Exemplos:

- B-CS: `monotonic` com `aug` detectou 22/22 quedas com 6 falsos alarmes.
- B-CV: B0/`concat` com `aug` teve o maior macro-F1 (0,3550).
- C-CV: `concat` com `aug` superou `physics` com `aug` em macro-F1, quedas e
  falsos alarmes.

### Checkpoint `best_val` × `last`

Na análise retrospectiva, os 158 `last.pt` foram inferidos em CPU com os mesmos
bundles, estatísticas de treino, suporte e protocolo de alarme, sem retreino e
sem gravar relatórios. Antes disso, um run `best_val` reinferido reproduziu
exatamente as predições salvas. A época selecionada teve média 12,53 (mínimo 4,
máximo 28).

| Protocolo | Pares | Δ macro-F1 médio (`best_val − last`) | `best_val` maior | `last` maior |
| --- | ---: | ---: | ---: | ---: |
| Le2i-CS | 79 | −0,0090 | 25 | 54 |
| Le2i-CV | 79 | +0,0148 | 59 | 20 |

A seleção por validação não é uniformemente superior. Os 79 pares de cada
protocolo compartilham dados e configurações e não são observações
independentes. Esta comparação é descritiva e não define nem sugere uma política
de checkpoint escolhida pelo teste.

### Contrastes pareados entre fusões (matriz limpa)

Diferenças `physics − controle`, médias de cinco diferenças pareadas por
semente. O DP é entre essas cinco diferenças.

| Protocolo | Comparação | Δ macro-F1 | DP Δ macro-F1 | Δ FA/h | Δ sensibilidade |
| --- | --- | ---: | ---: | ---: | ---: |
| Le2i-CS | `B1 physics` − `B0 concat` | +0,004286 | 0,0163 | +17,509728 | +0,063636 |
| Le2i-CS | `B1 physics` − `B1 learned` | −0,012803 | 0,0179 | +8,171206 | −0,036364 |
| Le2i-CS | `C1 physics` − `C0 concat` | +0,003705 | 0,0147 | +15,175097 | −0,054545 |
| Le2i-CS | `C1 physics` − `C1 learned` | +0,005381 | 0,0281 | −14,007782 | −0,027273 |
| Le2i-CV | `B1 physics` − `B0 concat` | −0,017901 | 0,0435 | −51,755266 | −0,005405 |
| Le2i-CV | `B1 physics` − `B1 learned` | −0,022553 | 0,0765 | +24,072217 | −0,102703 |
| Le2i-CV | `C1 physics` − `C0 concat` | −0,004686 | 0,0104 | −19,257773 | +0,070270 |
| Le2i-CV | `C1 physics` − `C1 learned` | −0,012549 | 0,0172 | −44,533601 | +0,086486 |

### Bootstrap pareado por sujeito (40 comparações)

As 40 comparações são os quatro pares acima × cinco sementes × dois protocolos,
com receita limpa e checkpoints `best_val`. Cada uma usa 1 000 réplicas pareadas
por sujeito (RNG 42, IC percentil de 95%), todas válidas para macro-F1, FA/h e
sensibilidade. O teste tem 2 sujeitos em Le2i-CS e 5 em Le2i-CV.

Os ICs são por semente e não formam um intervalo conjunto sobre sementes e
sujeitos. Com tão poucos clusters, não sustentam superioridade populacional.

No recorte mais favorável, C1/`physics` − C0/`concat` em Le2i-CV, o Δ FA/h é
−54,162487 nas sementes 42 (IC [−81,420332; −13,262111]) e 43 (IC
[−81,64689; −29,95839]), mas +30,09027 na semente 45 (IC [0; +56,12576]).
Ele é negativo nas sementes 42 a 44, positivo na 45 e nulo na 46, ou seja, não
é uniforme. Na semente 42 desse par, Δ macro-F1 = +0,000899 (IC
[−0,00299158; +0,00857648]) e Δ sensibilidade = +0,108108 (IC [0; +0,18]).
O registro de origem resume as 40 comparações, mas só detalha esses intervalos.
Os ICs das demais comparações foram documentados anteriormente, porém não estão
disponíveis para transcrição.

### Conclusão do predecessor

Não há superioridade global do gate físico. Em Le2i-CV, C1/`physics` tem um
trade-off operacional descritivo: menos FA/h e mais sensibilidade que C0 e
`learned`, com macro-F1 médio menor. Os controles `constant` e `monotonic` são
competitivos, e a necessidade causal do residual físico não foi demonstrada.

## Motion Physics: 150 treinos e 150 avaliações

Matriz: A/B0/B1/C0/C1 × três receitas × sementes 42–46 × Le2i-CS e Le2i-CV.
Todos os runs são limpos (sem `--augment` e sem perda de ordem) e usam
`best_val`. Fusões:

- A/`pose`: bundle SAM 3, com a entrada visual ignorada;
- B0/`concat` e B1/`physics`: bundle DINOv3;
- C0/`concat` e C1/`physics`: bundle SAM 3.

As sementes compartilham o bundle de cada braço e protocolo. Cada braço e
protocolo tem 15 runs e um `summary.json` que recusa comparações com bundle,
alvos, pesos, código ou ambiente divergentes. A auditoria reportada nos
registros foi 150/150 PASS, com `selection_performed=false`.

As 150 avaliações não se somam às 158 do predecessor. A receita `baseline` foi
treinada novamente nesta rodada. Seus agregados de macro-F1 e FA/h coincidem
com as configurações correspondentes da matriz limpa do predecessor (A/`pose`,
B0/`concat`, B1/`physics`, C0/`concat`, C1/`physics`), como esperado pelo
controle que reproduz o treino anterior.

### Métricas absolutas

Média ± desvio-padrão entre as cinco sementes. O conjunto de teste tem 22 quedas
em Le2i-CS e 37 em Le2i-CV:

- **Sensibilidade e quedas detectadas** referem-se a eventos reais de queda.
- **Falsos alarmes** são alarmes sem queda correspondente e não fazem parte das
  quedas detectadas.
- **Latência** é a média do atraso dos eventos detectados, não o tempo de
  inferência.

#### Le2i-CS

| Braço | Receita | Macro-F1 | Sensibilidade | Quedas detectadas | Falsos alarmes | FA/h | Latência média (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `baseline` | 0,5941 ± 0,0290 | 0,8182 ± 0,0850 | 18,0 ± 1,9 / 22 | 35,8 ± 5,3 | 208,95 ± 30,99 | 0,531 ± 0,052 |
| A | `motion` | 0,5957 ± 0,0194 | 0,8455 ± 0,0886 | 18,6 ± 1,9 / 22 | 33,0 ± 2,2 | 192,61 ± 13,05 | 0,494 ± 0,046 |
| A | `motion_physics` | 0,5995 ± 0,0148 | 0,8455 ± 0,0689 | 18,6 ± 1,5 / 22 | 32,6 ± 1,7 | 190,27 ± 9,77 | 0,519 ± 0,121 |
| B0 | `baseline` | 0,6396 ± 0,0054 | 0,8273 ± 0,0380 | 18,2 ± 0,8 / 22 | 8,8 ± 3,1 | 51,36 ± 18,18 | 0,344 ± 0,033 |
| B0 | `motion` | 0,6201 ± 0,0203 | 0,8636 ± 0,0321 | 19,0 ± 0,7 / 22 | 9,2 ± 2,8 | 53,70 ± 16,20 | 0,354 ± 0,055 |
| B0 | `motion_physics` | 0,6347 ± 0,0199 | 0,9091 ± 0,0321 | 20,0 ± 0,7 / 22 | 8,2 ± 2,9 | 47,86 ± 16,71 | 0,354 ± 0,025 |
| B1 | `baseline` | 0,6439 ± 0,0121 | 0,8909 ± 0,0518 | 19,6 ± 1,1 / 22 | 11,8 ± 3,3 | 68,87 ± 19,09 | 0,329 ± 0,020 |
| B1 | `motion` | 0,6244 ± 0,0112 | 0,8455 ± 0,0518 | 18,6 ± 1,1 / 22 | 9,8 ± 2,6 | 57,20 ± 15,11 | 0,324 ± 0,022 |
| B1 | `motion_physics` | 0,6375 ± 0,0037 | 0,8545 ± 0,0203 | 18,8 ± 0,4 / 22 | 11,0 ± 1,6 | 64,20 ± 9,23 | 0,334 ± 0,017 |
| C0 | `baseline` | 0,6295 ± 0,0157 | 0,9545 ± 0,0455 | 21,0 ± 1,0 / 22 | 13,0 ± 2,9 | 75,88 ± 17,02 | 0,376 ± 0,022 |
| C0 | `motion` | 0,6201 ± 0,0121 | 0,9000 ± 0,0674 | 19,8 ± 1,5 / 22 | 18,6 ± 3,5 | 108,56 ± 20,47 | 0,330 ± 0,022 |
| C0 | `motion_physics` | 0,6308 ± 0,0071 | 0,9455 ± 0,0380 | 20,8 ± 0,8 / 22 | 13,8 ± 4,5 | 80,54 ± 26,23 | 0,342 ± 0,043 |
| C1 | `baseline` | 0,6332 ± 0,0102 | 0,9000 ± 0,0380 | 19,8 ± 0,8 / 22 | 15,6 ± 1,9 | 91,05 ± 11,38 | 0,376 ± 0,020 |
| C1 | `motion` | 0,6372 ± 0,0101 | 0,9182 ± 0,0380 | 20,2 ± 0,8 / 22 | 14,8 ± 4,1 | 86,38 ± 23,85 | 0,362 ± 0,041 |
| C1 | `motion_physics` | 0,6222 ± 0,0140 | 0,8909 ± 0,0518 | 19,6 ± 1,1 / 22 | 16,4 ± 2,1 | 95,72 ± 12,10 | 0,344 ± 0,022 |

#### Le2i-CV

| Braço | Receita | Macro-F1 | Sensibilidade | Quedas detectadas | Falsos alarmes | FA/h | Latência média (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `baseline` | 0,4539 ± 0,0210 | 0,8270 ± 0,1262 | 30,6 ± 4,7 / 37 | 28,0 ± 6,0 | 168,51 ± 36,36 | 0,487 ± 0,070 |
| A | `motion` | 0,4599 ± 0,0161 | 0,8541 ± 0,0779 | 31,6 ± 2,9 / 37 | 22,4 ± 3,4 | 134,80 ± 20,23 | 0,519 ± 0,061 |
| A | `motion_physics` | 0,4498 ± 0,0338 | 0,7946 ± 0,1057 | 29,4 ± 3,9 / 37 | 25,6 ± 5,2 | 154,06 ± 31,15 | 0,504 ± 0,137 |
| B0 | `baseline` | 0,2970 ± 0,0248 | 0,6811 ± 0,1637 | 25,2 ± 6,1 / 37 | 39,0 ± 16,8 | 234,70 ± 100,97 | 0,810 ± 0,166 |
| B0 | `motion` | 0,2797 ± 0,0420 | 0,5676 ± 0,2253 | 21,0 ± 8,3 / 37 | 47,6 ± 20,9 | 286,46 ± 125,92 | 0,718 ± 0,107 |
| B0 | `motion_physics` | 0,2881 ± 0,0340 | 0,5838 ± 0,2361 | 21,6 ± 8,7 / 37 | 42,2 ± 20,1 | 253,96 ± 120,92 | 0,799 ± 0,217 |
| B1 | `baseline` | 0,2791 ± 0,0414 | 0,6757 ± 0,1209 | 25,0 ± 4,5 / 37 | 30,4 ± 19,3 | 182,95 ± 116,27 | 0,947 ± 0,207 |
| B1 | `motion` | 0,2985 ± 0,0457 | 0,6541 ± 0,1924 | 24,2 ± 7,1 / 37 | 27,8 ± 22,1 | 167,30 ± 133,11 | 0,734 ± 0,206 |
| B1 | `motion_physics` | 0,2814 ± 0,0510 | 0,5838 ± 0,2933 | 21,6 ± 10,9 / 37 | 42,2 ± 26,3 | 253,96 ± 158,50 | 0,511 ± 0,225 |
| C0 | `baseline` | 0,4697 ± 0,0194 | 0,8108 ± 0,1253 | 30,0 ± 4,6 / 37 | 15,4 ± 8,6 | 92,68 ± 52,05 | 0,500 ± 0,073 |
| C0 | `motion` | 0,4555 ± 0,0271 | 0,7676 ± 0,0824 | 28,4 ± 3,0 / 37 | 19,6 ± 5,4 | 117,95 ± 32,58 | 0,479 ± 0,065 |
| C0 | `motion_physics` | 0,4422 ± 0,0268 | 0,7514 ± 0,0774 | 27,8 ± 2,9 / 37 | 21,0 ± 3,7 | 126,38 ± 22,52 | 0,411 ± 0,040 |
| C1 | `baseline` | 0,4650 ± 0,0162 | 0,8811 ± 0,0410 | 32,6 ± 1,5 / 37 | 12,2 ± 3,8 | 73,42 ± 23,07 | 0,476 ± 0,046 |
| C1 | `motion` | 0,4483 ± 0,0329 | 0,8108 ± 0,1011 | 30,0 ± 3,7 / 37 | 18,4 ± 6,8 | 110,73 ± 40,95 | 0,430 ± 0,051 |
| C1 | `motion_physics` | 0,4449 ± 0,0166 | 0,8378 ± 0,0715 | 31,0 ± 2,6 / 37 | 15,8 ± 5,1 | 95,09 ± 30,51 | 0,388 ± 0,065 |

### Contraste controlado

O contraste primário da contribuição física é **`motion_physics − motion`**.
As duas receitas recebem a mesma tarefa auxiliar de movimento e diferem apenas
pela regularização cinemática. `motion_physics − baseline` mistura o efeito da
tarefa auxiliar com o da regularização e é secundário. Médias de cinco
diferenças pareadas por semente. Δ macro-F1 e Δ sensibilidade positivos e
Δ FA/h negativo indicam direção favorável:

| Protocolo | Braço | Δ macro-F1 | DP Δ macro-F1 | Δ sensibilidade | Δ FA/h | Δ latência (s) | Sementes com Δ macro-F1 > 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Le2i-CS | A | +0,0038 | 0,0150 | 0,0000 | −2,33 | +0,0247 | 3/5 |
| Le2i-CS | B0 | +0,0145 | 0,0134 | +0,0455 | −5,84 | −0,0001 | 4/5 |
| Le2i-CS | B1 | +0,0131 | 0,0086 | +0,0091 | +7,00 | +0,0097 | 5/5 |
| Le2i-CS | C0 | +0,0107 | 0,0071 | +0,0455 | −28,02 | +0,0121 | 5/5 |
| Le2i-CS | C1 | −0,0150 | 0,0212 | −0,0273 | +9,34 | −0,0181 | 1/5 |
| Le2i-CV | A | −0,0101 | 0,0417 | −0,0595 | +19,26 | −0,0147 | 3/5 |
| Le2i-CV | B0 | +0,0084 | 0,0160 | +0,0162 | −32,50 | +0,0811 | 3/5 |
| Le2i-CV | B1 | −0,0171 | 0,0872 | −0,0703 | +86,66 | −0,2233 | 2/5 |
| Le2i-CV | C0 | −0,0133 | 0,0304 | −0,0162 | +8,43 | −0,0676 | 1/5 |
| Le2i-CV | C1 | −0,0034 | 0,0257 | +0,0270 | −15,65 | −0,0422 | 3/5 |

Comparação das três receitas em macro-F1 médio, a partir das linhas da tabela
verificada:

| Protocolo | Braço | `motion_physics − motion` | `motion_physics − baseline` | `motion − baseline` |
| --- | --- | --- | --- | --- |
| Le2i-CS | A | +0,0038 | +0,0054 | +0,0016 |
| Le2i-CS | B0 | +0,0145 | −0,0049 | −0,0195 |
| Le2i-CS | B1 | +0,0131 | −0,0063 | −0,0194 |
| Le2i-CS | C0 | +0,0107 | +0,0013 | −0,0094 |
| Le2i-CS | C1 | −0,0150 | −0,0110 | +0,0040 |
| Le2i-CV | A | −0,0101 | −0,0041 | +0,0060 |
| Le2i-CV | B0 | +0,0084 | −0,0088 | −0,0173 |
| Le2i-CV | B1 | −0,0171 | +0,0023 | +0,0194 |
| Le2i-CV | C0 | −0,0133 | −0,0275 | −0,0142 |
| Le2i-CV | C1 | −0,0034 | −0,0201 | −0,0167 |
| **Grupos com média maior** | | **5/10** | **3/10** | **4/10** |

`motion_physics − motion` aumentou o macro-F1 médio em 5 dos 10 grupos
braço/protocolo (A, B0, B1 e C0 no Le2i-CS; B0 no Le2i-CV).
`motion_physics − baseline` aumentou em apenas 3 dos 10, e nenhum braço mantém
esse contraste positivo nos dois protocolos.

Não há superioridade global demonstrada. As diferenças ficam abaixo de 0,03 em
valor absoluto, na mesma ordem do desvio-padrão entre sementes. No Le2i-CV, a
regularização reduziu o macro-F1 médio em quatro dos cinco braços. Em C1,
sensibilidade e FA/h mudam de direção entre protocolos, e B1-CV mostra
heterogeneidade forte (DP Δ macro-F1 0,0872).

O critério favorável definido antes da rodada era `motion_physics` reduzir
FA/h sem reduzir sensibilidade frente a `motion`, de forma consistente entre
sementes. Nas médias, isso ocorre em 5 dos 10 grupos (A, B0 e C0 no Le2i-CS;
B0 e C1 no Le2i-CV).

### Relação com o GateFall canônico

Os registros de origem também comparam, de forma descritiva, as médias canônicas
das sementes 42–46 com a receita externa `motion_physics`. Em Le2i-CV, a própria
baseline externa já supera o macro-F1 canônico nos cinco braços. Assim, a
diferença entre repositórios antecede a nova perda cinemática e não mede o
efeito da física.

Os sistemas diferem em pose134 × pose138, evidence8, preparo dos dados e
seleção de checkpoint (`best_val` externo × época 30 canônica). Essa comparação
não tem interpretação causal e não é reproduzida aqui.

## Limitações da validação

- **Não canônico.** A baseline externa usa pose138, evidence8, padronização por
  bundle, seleção `best_val` e latência sem arredondamento. Seus números não
  são comparáveis diretamente com as referências pose134 de época 30, nem
  atribuem causalidade à mudança de representação. Nenhuma comparação isolou
  pose134 × pose138.
- **Inferência estatística limitada.** O desvio-padrão de cinco sementes mede
  variação de inicialização, não diversidade de sujeitos. O teste tem 2 sujeitos
  em Le2i-CS e 5 em Le2i-CV. Os 40 bootstraps pertencem ao predecessor e não são
  intervalos de confiança da matriz Motion Physics; esta não tem bootstrap
  próprio.
- **Teste já consultado.** O teste foi consultado ao longo da sequência
  exploratória. Nenhuma escolha de braço, receita, checkpoint ou hiperparâmetro
  deve ser feita a partir destes números.
- **Avaliação limpa.** Nenhuma avaliação com falhas pareadas em dados reservados
  foi feita. O aumento de falhas atua só no treino, e não houve blur, recorte ou
  oclusão reais.
- **Escopo intra-Le2i.** Le2i-CV é mudança de ambiente e câmera dentro do Le2i,
  não transferência entre datasets.
- **Artefatos fora do repositório.** Runs, bundles, sidecars e logs permanecem
  nos clones locais. Esta página não reexecutou treinos nem avaliações.
- **Provas formais.** O projeto Lean em `formal/` prova fórmulas sobre números
  reais para as derivadas cinemáticas, o gate e a margem de ordem do
  predecessor. Não verifica o programa PyTorch, os dados nem a qualidade das
  predições, e não cobre a extensão Motion Physics.
- **Escopo físico.** `Lcin` é coerência cinemática 2D em coordenadas de imagem.
  Não modela gravidade, forças, contato ou dinâmica 3D.

## Integração neste repositório

Arquivos importados sem alteração das revisões de origem:

- `src/gatefall/experiments/`;
- a alteração compatível de `src/gatefall/eval/shared/events.py` (parâmetro
  `round_latency`, cujo padrão preserva o arredondamento canônico);
- `specs/001-*` e `specs/002-*`;
- `formal/`;
- as duas páginas de método e as entradas do `mkdocs.yml`;
- as regras de `.gitignore`;
- as etapas de CI (selftest `physics` e job Lean).

Desvios intencionais da origem:

- Não foram importados os artefatos do Spec Kit (`.specify/`, `.agents/skills/`),
  que são ferramenta de autoria e não fazem parte do experimento.
  `specs/001-physics-informed-gate/tasks.md` ainda cita a constituição em
  `.specify/memory/`, como registro histórico.
- O aviso do `README.md` da origem descrevia o fork e não foi importado.
- A seção de instalação de [Aprendizado de movimento](motion-physics.md) passou a
  partir da raiz deste repositório em vez de clonar o fork.
- As duas páginas de método ganharam um aviso apontando para esta página e
  deixaram de afirmar que não houve avaliação real.
- A CI passou a executar também o selftest `motion`.

Saídas do experimento são gravadas somente nos diretórios passados por
`--output`/`--run-dir`; nenhum comando canônico chama esses módulos.
