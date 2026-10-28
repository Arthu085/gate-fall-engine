# Especificação: estudo comparativo GateFall 2026

**Branch**: `feat/gatefall-comparative-study-2026` · **Base**: `main` em `ef6b3c1`
**Origem**: plano de estudo do orientador de 23/10/2026.
**Estado**: infraestrutura implementada e verificada com dados sintéticos; todas as execuções reais estão **pendentes**.

## Objetivo

Comparar oito configurações sob uma única interface de treino, inferência e avaliação causal.
O estudo usa os protocolos Le2i-CS/Le2i-CV, o UP-Fall como dataset primário, a transferência
congelada para o GMDCSA-24, uma matriz separada de aprendizado de movimento e ablações de gate
fora da tabela primária. O estudo estende o trabalho A/B/C já concluído. Ele não redesenha os
modelos históricos e não autoriza arquiteturas fora do plano.

## Cenários de uso e aceitação

### US1 — Preparar datasets novos com proveniência (P1)

O aluno coloca o UP-Fall e o GMDCSA-24 nos caminhos documentados. Em seguida, gera manifestos
versionados, partições por sujeito, grade causal a 10 fps e relatório de cobertura.

Aceitação:

- Nenhum sujeito ou gravação aparece em mais de uma partição.
- As duas câmeras de um ensaio UP-Fall ficam na mesma partição.
- Arquivos ausentes, rótulos não verificáveis e rótulos que passam do fim da fonte falham com
  mensagem explícita.
- As classes OmniFall 10–15 viram `IGNORE` e são contadas como não suportadas.

### US2 — Extrair features uma única vez (P1)

YOLO-Pose, qualidade, DINOv3 e SAM 3 rodam congelados e gravam um HDF5 por vídeo.

Aceitação:

- `extract-all` pula vídeos já extraídos.
- Os pacotes conferem identidade, `K`, partição e proveniência de cada arquivo.
- Um relatório de cobertura lista faltas e desalinhamentos sem carregar backbones.

### US3 — Treinar e avaliar oito métodos comparáveis (P1)

Os oito métodos são A, B0, B1 learned, C0, C1 learned, B1 physics, C1 physics e TCNTE adaptado.
Todos usam as mesmas partições, a mesma grade, as mesmas janelas, as mesmas sementes 42–46, o
mesmo protocolo de alarme e a mesma seleção por validação.

Aceitação:

- O teste só é consultado depois do treino.
- A predição em t não muda quando só os quadros posteriores a t mudam.
- A normalização depende apenas do treino.

### US4 — Transferência externa congelada (P2)

Checkpoints e estatísticas do UP-Fall são aplicados ao GMDCSA-24 sem adaptação.

Aceitação:

- O pacote externo contém só a partição `test`.
- Schema e fonte visual precisam coincidir.
- A saída é única por run e nunca é sobrescrita.

### US5 — Matriz de movimento (P2)

A matriz tem cinco braços × três receitas (`baseline`, `motion`, `motion_physics`) × cinco sementes
× dois protocolos Le2i, num total de 150 treinos antes do reuso.

Aceitação:

- Runs externos só são reaproveitados após auditoria de compatibilidade.
- A comparação primária é `motion_physics − motion`.
- O texto nunca chama a regularização de PINN 3D.

### US6 — Relatórios reprodutíveis (P2)

Tabelas, contrastes pareados, IC por cluster, figuras, linhas do tempo qualitativas, custo e
manifesto são gerados só a partir de avaliações verificadas.

Aceitação: o relatório é recusado quando faltam avaliações, quando as sementes estão incompletas,
quando os rótulos de teste divergem, quando o protocolo diverge, quando há adaptação na
transferência ou quando um artefato foi alterado.

### Casos de borda

- Ensaios UP-Fall sem rótulo (8-11-2 e 8-11-3).
- Lacunas internas de rótulo e quadros descartados na sequência de imagens.
- Câmera com início deslocado.
- Classe sem suporte no teste.
- Mais da metade dos eventos perdida, o que deixa a latência mediana indefinida.
- Run externo com pacote diferente.
- Destino de run já existente.
- Árvore Git suja no momento da execução.

## Requisitos funcionais

- **FR-001**: adaptadores `upfall` (sequência de imagens) e `gmdcsa24` (vídeo) sob o contrato
  `DatasetAdapter`, preservando sujeito, atividade, ensaio, câmera, timestamp e gravação.
- **FR-002**: partição `upfall-subject-v1` por sujeito, congelada no código. O split `up_fall-cs`
  do OmniFall repete sujeitos entre partições e é auditado, não usado.
- **FR-003**: rótulos OmniFall na revisão `68e5cee`, com o mapeamento de
  [`contracts/class-mapping.md`](contracts/class-mapping.md); `fall`, `fallen` e `lying`
  permanecem distintos.
- **FR-004**: grade causal a 10 fps. Para sequências com lacuna maior que um passo, o quadro
  escolhido é o último em ou antes de t. Os passos desalinhados ficam marcados como
  `source_gap` sem perder o rótulo.
- **FR-005**: extração congelada, determinística e em cache por vídeo, com schema e proveniência
  verificados ao montar pacotes.
- **FR-006**: oito métodos primários e dez variantes de gate na interface
  `gatefall.experiments.physics`. Todos usam pose138 + evidence8, e o pose134 histórico não é
  usado no estudo.
- **FR-007**: TCNTE independente e adaptado (o upstream não declara licença), com todos os
  desvios em [`research.md`](research.md).
- **FR-008**: receitas de movimento importadas de `5f6e2ae`, com o futuro restrito à supervisão
  de treino.
- **FR-009**: métricas recalculadas das predições por janela:
  - F1 e suporte por classe, macro-F1 sobre as classes com suporte e macro-F1 restrito de 8
    classes (histórico);
  - eventos detectados e perdidos, e sensibilidade;
  - detecção durante a queda × só após a queda;
  - FA total e FA/h com denominador explícito;
  - latência dos detectados, com perdas contabilizadas.
- **FR-010**: custo separado em extração (perfil isolado), classificação (tempo por janela) e
  ponta a ponta.
- **FR-011**: diferenças pareadas por semente e bootstrap por cluster (sujeito ou gravação),
  com o mesmo sorteio para os dois métodos e para todas as sementes. Janelas sobrepostas
  nunca são tratadas como observações independentes.
- **FR-012**: seleção de checkpoint só por validação (`best_val`, macro-F1 restrito de 8
  classes, critério histórico mantido para todos os métodos); nenhum limiar ou hiperparâmetro é
  ajustado no teste, e a decisão por janela é o argmax.
- **FR-013**: registro do plano com identidade de run, proveniência escrita uma única vez,
  ledger JSONL de execução e reuso apenas de saídas verificadas.
- **FR-014**: comandos `plan`, `commands`, `run` (padrão `--dry-run`), `report`, `reuse-audit`,
  `features-report`, `profile-extraction` (exige `--execute`) e `selftest`.

## Critérios de sucesso

- **SC-001**: selftests sintéticos de staged, physics, motion, TCNTE e study passam em CPU.
- **SC-002**: o plano tem 80 (Le2i), 40 (UP-Fall), 40 avaliações de transferência, 150
  (movimento) e 50 (ablações) passos de treino/avaliação, num total de 300 treinos únicos.
- **SC-003**: os nove classificadores, incluindo o de movimento, são invariantes a quadros após t.
- **SC-004**: resultados científicos estão **PENDENTES**. Nenhuma métrica real é declarada
  antes de execução e relatório verificados.

## Escopo e trade-offs

- **Substitui** a reexecução das cinco configurações históricas sobre pose134. Elas continuam
  como referência congelada em `runs/reference`.
- **CAUCAFall v4** é só um fallback documentado e não entra no plano.
- Novos backbones, novas métricas além das listadas e um quarto dataset ficam fora do escopo.
- O estudo é RGB monocular; nenhum descritor de profundidade entra no estudo.
