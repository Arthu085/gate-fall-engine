# Resultados externos — gate físico e Motion Physics

!!! warning "Resultados externos, não canônicos"
    Os números desta página vêm de dois experimentos conduzidos pelo orientador
    fora deste repositório e importados aqui apenas para documentação e
    reprodutibilidade. Eles **não** substituem nem atualizam os resultados
    canônicos A/B0/B1/C0/C1, que continuam definidos pelos pipelines pose134,
    pelas referências em `runs/reference/` e pelas páginas de treino e avaliação
    de cada braço. A baseline externa **não** é a baseline original do GateFall.

Os módulos correspondentes são opt-in e só executam por invocação explícita
(`python -m gatefall.experiments.physics` e `python -m gatefall.experiments.motion`).
A descrição dos métodos está em
[Experimento de consistência e provas Lean](../train/physics-gate.md) e em
[Aprendizado de movimento e coerência cinemática](motion-physics.md).

## Proveniência

| Item | Gate físico (predecessor) | Motion Physics (extensão) |
| --- | --- | --- |
| Repositório de origem | [rsramos30/gate-fall-engine](https://github.com/rsramos30/gate-fall-engine) | [rsramos30/gate-fall-motion-physics](https://github.com/rsramos30/gate-fall-motion-physics) |
| Commit | `301b32e42e47f29fed1daf607948d11ddf3b4e5d` | `5f6e2ae4ce0622b6add05da207d8a20d29929159` |
| Base | `ef6b3c1` (merge do PR #100 neste repositório) | `301b32e`, mais `9fb8a8a` e `5f6e2ae` |
| Módulo | `gatefall.experiments.physics` | `gatefall.experiments.motion` (reutiliza `physics`) |
| Especificação | `specs/001-physics-informed-gate/` | `specs/002-motion-physics-learning/` |
| Avaliações de teste históricas | 158 | 150 (de 150 treinos) |
| Revisão registrada nos runs | `301b32e` em todos os 158, `git_dirty=true` | `5f6e2ae` em todos os 150, `git_dirty=false` |

O código Python de `src/gatefall/` deste branch é idêntico byte a byte ao de
`5f6e2ae`, que por sua vez contém o de `301b32e` sem alterações em
`gatefall.experiments.physics`. Por isso os hashes de módulos registrados em
`code_provenance.python_sources_sha256` dos runs históricos continuam
verificáveis contra este repositório.

No predecessor, `git_dirty=true` decorre de arquivos não rastreados no checkout
do orientador (`data/processed/` e o script de lote `gatefall_lote_ab_final.sh`);
`git status --porcelain` os conta como modificação. Em todos os 158 `config.json`,
o hash de cada módulo Python registrado coincide com o arquivo correspondente
em `301b32e`, de modo que o código executado é o da revisão declarada.

Os runs, bundles e logs ficaram nos checkouts locais do orientador
(`runs/local/le2i/` e `runs/local/le2i_cv/`, ignorados pelo Git) e não fazem
parte deste repositório. As tabelas da extensão abaixo vêm de
`runs/local/motion-physics-metricas-completas.csv`, com 30 linhas agregadas e
SHA-256 `cf4fc9756c14af26625afd682ed782cf926470e49d475e55a3aa020bb528c514`,
conferido antes da transcrição. Cada linha aponta para o
`summary/summary.json` do braço e protocolo de origem.

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
| Falhas sintéticas | Não | Opcional (`--augment`), somente no treino |
| Latência na avaliação | Arredondada pelo protocolo de alarme | Precisão integral (`round_latency=False`) |
| Parâmetros treináveis (C1, exemplo) | 67 885 | 69 455 (C1/`physics`) |

Ambas mantêm janela de 24 frames a 10 Hz, stride de treino 4, TCN 32/32/32 com
kernel 3, dilatações 1/2/4 e dropout 0,3, AdamW (lr 0,001, weight decay 0,01),
cosseno, entropia cruzada ponderada por classe, clipping 1,0, 30 épocas,
batch 64 e seleção `best_val` por macro-F1 restrito de validação. Os
backbones YOLO-Pose, DINOv3 e SAM 3 permanecem congelados, e as features são
lidas dos artefatos offline existentes.

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
| Piloto `--augment` | B0/`concat`, C0/`concat`; B1 e C1 × `learned`, `monotonic`, `physics` (8) | 42 | Le2i-CS, Le2i-CV | 16 |
| Piloto `--augment --order-weight 0.1` | B1 e C1 × `learned`, `monotonic`, `physics` (6) | 42 | Le2i-CS, Le2i-CV | 12 |
| **Total** | | | | **158** |

!!! note "Métricas agregadas pendentes"
    Não existe um agregado verificado dessas 158 avaliações. Esta página não
    recalcula nem estima seus números; as métricas por configuração ficam
    pendentes de uma agregação com integridade verificada a partir dos
    `evaluation-test.json` originais.

As 158 avaliações são distintas dos 150 treinos da extensão: os controles
`baseline` da Motion Physics foram treinados novamente e não reutilizam
checkpoints do predecessor.

## Motion Physics: 150 treinos e 150 avaliações

Matriz: A/B0/B1/C0/C1 × três receitas × sementes 42–46 × Le2i-CS e Le2i-CV.
Todos os runs são limpos (sem `--augment` e sem perda de ordem). Fusões: A/`pose`
(bundle SAM 3, visual ignorado), B0/`concat` e B1/`physics` (bundle DINOv3),
C0/`concat` e C1/`physics` (bundle SAM 3). Cada braço e protocolo tem 15 runs e
um `summary.json` que recusa comparações com bundle, alvos, pesos, código ou
ambiente divergentes.

Valores: média ± desvio-padrão entre as cinco sementes. O conjunto de teste tem
22 quedas no Le2i-CS e 37 no Le2i-CV.

### Le2i-CS

| Braço | Receita | Macro-F1 | Sensibilidade | Quedas detectadas | FA/h | Latência média (s) |
| --- | --- | --- | --- | --- | --- | --- |
| A | `baseline` | 0,594 ± 0,029 | 0,818 ± 0,085 | 18,0 / 22 | 208,9 ± 31,0 | 0,53 ± 0,05 |
| A | `motion` | 0,596 ± 0,019 | 0,845 ± 0,089 | 18,6 / 22 | 192,6 ± 13,1 | 0,49 ± 0,05 |
| A | `motion_physics` | 0,599 ± 0,015 | 0,845 ± 0,069 | 18,6 / 22 | 190,3 ± 9,8 | 0,52 ± 0,12 |
| B0 | `baseline` | 0,640 ± 0,005 | 0,827 ± 0,038 | 18,2 / 22 | 51,4 ± 18,2 | 0,34 ± 0,03 |
| B0 | `motion` | 0,620 ± 0,020 | 0,864 ± 0,032 | 19,0 / 22 | 53,7 ± 16,2 | 0,35 ± 0,05 |
| B0 | `motion_physics` | 0,635 ± 0,020 | 0,909 ± 0,032 | 20,0 / 22 | 47,9 ± 16,7 | 0,35 ± 0,03 |
| B1 | `baseline` | 0,644 ± 0,012 | 0,891 ± 0,052 | 19,6 / 22 | 68,9 ± 19,1 | 0,33 ± 0,02 |
| B1 | `motion` | 0,624 ± 0,011 | 0,845 ± 0,052 | 18,6 / 22 | 57,2 ± 15,1 | 0,32 ± 0,02 |
| B1 | `motion_physics` | 0,638 ± 0,004 | 0,855 ± 0,020 | 18,8 / 22 | 64,2 ± 9,2 | 0,33 ± 0,02 |
| C0 | `baseline` | 0,629 ± 0,016 | 0,955 ± 0,045 | 21,0 / 22 | 75,9 ± 17,0 | 0,38 ± 0,02 |
| C0 | `motion` | 0,620 ± 0,012 | 0,900 ± 0,067 | 19,8 / 22 | 108,6 ± 20,5 | 0,33 ± 0,02 |
| C0 | `motion_physics` | 0,631 ± 0,007 | 0,945 ± 0,038 | 20,8 / 22 | 80,5 ± 26,2 | 0,34 ± 0,04 |
| C1 | `baseline` | 0,633 ± 0,010 | 0,900 ± 0,038 | 19,8 / 22 | 91,1 ± 11,4 | 0,38 ± 0,02 |
| C1 | `motion` | 0,637 ± 0,010 | 0,918 ± 0,038 | 20,2 / 22 | 86,4 ± 23,9 | 0,36 ± 0,04 |
| C1 | `motion_physics` | 0,622 ± 0,014 | 0,891 ± 0,052 | 19,6 / 22 | 95,7 ± 12,1 | 0,34 ± 0,02 |

### Le2i-CV

| Braço | Receita | Macro-F1 | Sensibilidade | Quedas detectadas | FA/h | Latência média (s) |
| --- | --- | --- | --- | --- | --- | --- |
| A | `baseline` | 0,454 ± 0,021 | 0,827 ± 0,126 | 30,6 / 37 | 168,5 ± 36,4 | 0,49 ± 0,07 |
| A | `motion` | 0,460 ± 0,016 | 0,854 ± 0,078 | 31,6 / 37 | 134,8 ± 20,2 | 0,52 ± 0,06 |
| A | `motion_physics` | 0,450 ± 0,034 | 0,795 ± 0,106 | 29,4 / 37 | 154,1 ± 31,2 | 0,50 ± 0,14 |
| B0 | `baseline` | 0,297 ± 0,025 | 0,681 ± 0,164 | 25,2 / 37 | 234,7 ± 101,0 | 0,81 ± 0,17 |
| B0 | `motion` | 0,280 ± 0,042 | 0,568 ± 0,225 | 21,0 / 37 | 286,5 ± 125,9 | 0,72 ± 0,11 |
| B0 | `motion_physics` | 0,288 ± 0,034 | 0,584 ± 0,236 | 21,6 / 37 | 254,0 ± 120,9 | 0,80 ± 0,22 |
| B1 | `baseline` | 0,279 ± 0,041 | 0,676 ± 0,121 | 25,0 / 37 | 182,9 ± 116,3 | 0,95 ± 0,21 |
| B1 | `motion` | 0,299 ± 0,046 | 0,654 ± 0,192 | 24,2 / 37 | 167,3 ± 133,1 | 0,73 ± 0,21 |
| B1 | `motion_physics` | 0,281 ± 0,051 | 0,584 ± 0,293 | 21,6 / 37 | 254,0 ± 158,5 | 0,51 ± 0,22 |
| C0 | `baseline` | 0,470 ± 0,019 | 0,811 ± 0,125 | 30,0 / 37 | 92,7 ± 52,0 | 0,50 ± 0,07 |
| C0 | `motion` | 0,455 ± 0,027 | 0,768 ± 0,082 | 28,4 / 37 | 118,0 ± 32,6 | 0,48 ± 0,06 |
| C0 | `motion_physics` | 0,442 ± 0,027 | 0,751 ± 0,077 | 27,8 / 37 | 126,4 ± 22,5 | 0,41 ± 0,04 |
| C1 | `baseline` | 0,465 ± 0,016 | 0,881 ± 0,041 | 32,6 / 37 | 73,4 ± 23,1 | 0,48 ± 0,05 |
| C1 | `motion` | 0,448 ± 0,033 | 0,811 ± 0,101 | 30,0 / 37 | 110,7 ± 40,9 | 0,43 ± 0,05 |
| C1 | `motion_physics` | 0,445 ± 0,017 | 0,838 ± 0,072 | 31,0 / 37 | 95,1 ± 30,5 | 0,39 ± 0,06 |

### Contraste controlado

O contraste primário da contribuição física é **`motion_physics − motion`**:
as duas receitas recebem a mesma tarefa auxiliar de movimento e diferem apenas
pela regularização cinemática. `motion_physics − baseline` mistura o efeito da
tarefa auxiliar com o da regularização e é secundário. As diferenças abaixo são
entre médias de macro-F1 das mesmas cinco sementes (equivalentes à média das
diferenças pareadas por semente) e foram calculadas a partir das linhas da
tabela verificada:

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
braço/protocolo (A, B0, B1 e C0 no Le2i-CS; B0 no Le2i-CV); `motion_physics − baseline`
aumentou em apenas 3 dos 10. Não há superioridade global demonstrada: as
diferenças ficam abaixo de 0,03 em valor absoluto, na mesma ordem do
desvio-padrão entre sementes, e no Le2i-CV a regularização reduziu o macro-F1
médio em quatro dos cinco braços. O critério favorável definido antes da rodada
era `motion_physics` reduzir FA/h sem reduzir sensibilidade frente a `motion`,
de forma consistente entre sementes. Nas médias, isso ocorre em 5 dos 10 grupos
(A, B0 e C0 no Le2i-CS; B0 e C1 no Le2i-CV); a consistência por semente não
consta do agregado verificado.

## Limitações da validação

- **Não canônico.** A baseline externa usa pose138, evidence8, padronização por
  bundle e latência sem arredondamento; seus números não são comparáveis
  diretamente com as referências pose134 nem atribuem causalidade à mudança de
  representação.
- **Sem inferência estatística.** O desvio-padrão de cinco sementes não é
  intervalo de confiança entre sujeitos. Nenhum teste estatístico, bootstrap por
  sujeito ou correção para múltiplas comparações foi executado sobre esses runs.
- **Avaliação limpa.** Nenhuma avaliação de robustez com falhas pareadas em dados
  reservados foi feita; o aumento de falhas atua só no treino.
- **Predecessor sem agregado.** As 158 avaliações do gate físico não têm métricas
  agregadas verificadas (ver acima).
- **Artefatos fora do repositório.** Runs, bundles, sidecars e logs permanecem
  nos checkouts locais do orientador; esta página não reexecutou treinos nem
  avaliações, e seus números não foram recomputados.
- **Provas formais.** O projeto Lean em `formal/` prova fórmulas sobre números
  reais para as derivadas cinemáticas, o gate e a margem de ordem do
  predecessor. Não verifica o programa
  PyTorch, os dados nem a qualidade das predições, e não cobre a extensão
  Motion Physics.
- **Escopo físico.** `Lcin` é coerência cinemática 2D em coordenadas de imagem.
  Não modela gravidade, forças, contato ou dinâmica 3D.

## Integração neste repositório

Arquivos importados sem alteração das revisões de origem: `src/gatefall/experiments/`,
a alteração compatível de `src/gatefall/eval/shared/events.py` (parâmetro
`round_latency`, cujo padrão preserva o arredondamento canônico), `specs/001-*`,
`specs/002-*`, `formal/`, as duas páginas de método, as entradas do `mkdocs.yml`,
as regras de `.gitignore` e as etapas de CI (selftest `physics` e job Lean).

Desvios intencionais da origem:

- Não foram importados os artefatos do Spec Kit (`.specify/`, `.agents/skills/`),
  que são ferramenta de autoria e não fazem parte do experimento;
  `specs/001-physics-informed-gate/tasks.md` ainda cita a constituição em
  `.specify/memory/`, como registro histórico.
- O aviso do `README.md` da origem descrevia o fork e não foi importado.
- A seção de instalação de [Aprendizado de movimento](motion-physics.md) passou a
  partir da raiz deste repositório em vez de clonar o fork, e as duas páginas de
  método ganharam um aviso apontando para esta página.

Saídas do experimento são gravadas somente nos diretórios passados por
`--output`/`--run-dir`; nenhum comando canônico chama esses módulos.
