# Feature Specification: Fusão orientada por consistência do movimento

**Feature Branch**: `codex/physics-informed-gate`

**Created**: 2026-10-09

**Status**: Implementada e validada sintética/formalmente; retreino real pendente de dados

**Input**: Criar uma spec e implementar as alterações discutidas, usando Spec Kit
 e Lean para verificar a lógica matemática.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Comparar modelos com cinemática válida (Priority: P1)
O pesquisador precisa repetir os cinco braços sem transformar ausência de
observação em aceleração artificial, preservando a reprodução histórica.

**Why this priority**: Todos os braços consomem pose; um artefato comum impede
atribuir mudanças de desempenho ao gate.

**Independent Test**: Movimento uniforme com observações ausentes mantém
aceleração nula; movimento uniformemente acelerado mantém a aceleração correta.

**Acceptance Scenarios**:
1. **Given** observações válidas separadas por um gap curto, **When** derivadas
são estimadas, **Then** usam intervalos reais e velocidades medidas, com validade explícita.
2. **Given** gap longo ou início sem observação, **When** o histórico é insuficiente,
**Then** a medida fica indisponível e finita, sem inventar movimento.
3. **Given** referências publicadas, **When** um experimento novo é executado,
**Then** referências e estatísticas antigas permanecem intactas.

### User Story 2 - Medir adaptação por confiabilidade (Priority: P1)
O pesquisador compara concatenação, gate constante, qualidade como entrada,
gate livre, gate monotônico e gate informado por consistência, com a mesma receita.

**Why this priority**: Separar benefício da informação visual, das entradas
auxiliares e da ponderação evita atribuir ganhos ao mecanismo errado.

**Independent Test**: Aumentar inconsistência de uma modalidade, mantendo as
outras entradas fixas, nunca aumenta seu multiplicador no gate de consistência.

**Acceptance Scenarios**:
1. **Given** movimento corporal coerente, **When** o corpo gira/translada,
**Then** o novo residual não trata movimento global automaticamente como erro
de pose. A qualidade legada é mantida comum aos controles; não se exige
invariância do gate completo à rotação/perspectiva.
2. **Given** uma articulação espúria, **When** a consistência é medida,
**Then** existe evidência de deformação e informação de disponibilidade.
3. **Given** DINOv3, **When** se solicita evidência física visual,
**Then** sua indisponibilidade é explícita; diferenças de embedding não viram leis físicas.

### User Story 3 - Aprender com falhas controladas (Priority: P2)
O pesquisador treina com falhas em blocos de uma modalidade e regularização
que exige resposta mínima do gate, sem contaminar a avaliação limpa.

**Why this priority**: Direção correta sem amplitude útil não protege o classificador.

**Independent Test**: Uma resposta insuficiente produz perda positiva e gradiente
não nulo; os controles recebem exatamente as mesmas falhas.

**Acceptance Scenarios**:
1. **Given** mesma semente e vídeo, **When** falhas são geradas,
**Then** a variante é reproduzível e a outra modalidade permanece idêntica.
2. **Given** um par realmente alterado, **When** o gate não atinge a margem,
**Then** a regularização contribui ao aprendizado.
3. **Given** val/test, **When** a avaliação limpa é executada,
**Then** nenhuma corrupção de treino nem ajuste de estatística é aplicado.

### User Story 4 - Avaliar, reproduzir e auditar a lógica (Priority: P2)
O pesquisador seleciona checkpoints por validação, avalia eventos sem perda
intermediária de precisão e verifica as propriedades matemáticas em Lean.

**Why this priority**: Ganhos precisam ser rastreáveis e distinguidos de hipóteses.

**Independent Test**: Um treino sintético completo produz checkpoints último e
selecionado, métricas, proveniência e relatório de alarmes; as provas compilam.

**Acceptance Scenarios**:
1. **Given** empate na melhor validação, **When** o checkpoint é selecionado,
**Then** vence a primeira época empatada, sem consultar o teste.
2. **Given** latências conhecidas, **When** se calcula a média,
**Then** não há arredondamento intermediário no novo protocolo.
3. **Given** os teoremas formais, **When** Lean verifica o projeto,
**Then** monotonicidade, limites e propriedades da perda/cinemática são aceitos sem lacunas.

### Edge Cases
- Nenhuma pessoa, modalidade ausente, corpo degenerado, poucos pontos confiáveis.
- Gap longo, zero observações, somente uma ou duas observações válidas.
- Movimento rápido coerente; deformação por perspectiva; ambas modalidades ruins.
- Artefatos não finitos, partições sobrepostas, cache alterado, versões incompatíveis.
- Margem inválida, regularização solicitada em controle sem gate, dados reais ausentes.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Disponibilizar experimento versionado com cinemática corrigida comum aos cinco braços.
- **FR-002**: Expor disponibilidade das derivadas e reiniciar histórico após intervalo máximo declarado.
- **FR-003**: Calcular consistência causal de pose e proxy temporal SAM 3, com validade explícita.
- **FR-004**: Oferecer controles sem gate, constante, qualidade como entrada, livre, monotônico e consistência.
- **FR-005**: Restringir sinais do gate monotônico e do gate de consistência; indicar que são hipóteses.
- **FR-006**: Oferecer falhas temporais reproduzíveis, CE em exemplos limpos/corrompidos e margem positiva configurável.
- **FR-007**: Ajustar estatísticas apenas no treino e reutilizá-las em val/test e corrupções.
- **FR-008**: Salvar último e melhor checkpoint, política e época escolhida, métricas e hashes das fontes.
- **FR-009**: Avaliar classificação e eventos na mesma grade; manter precisão de latência no experimento novo.
- **FR-010**: Proibir escrita sobre fontes e referências e recusar artefatos incompatíveis.
- **FR-011**: Disponibilizar provas Lean e testes sintéticos causais, numéricos e de integração.
- **FR-012**: Documentar comandos de preparação, treino, avaliação e matriz das ablações.
- **FR-013**: Distinguir software verificado de retreinos reais ainda não executados.

### Key Entities
- Observação: vídeo, instante, modalidade, valor, confiança e disponibilidade.
- Pacote experimental: fontes, partições, features limpas e variantes reproduzíveis.
- Receita: braço, modo de fusão, sementes, política de checkpoint e corrupções.
- Resultado: histórico, checkpoints, predições, métricas e proveniência.
- Prova: hipótese, propriedade e vínculo documentado com a implementação.

## Success Criteria *(mandatory)*

### Measurable Outcomes
- **SC-001**: Casos analíticos de movimento uniforme/acelerado passam com erro numérico inferior a 1e-4.
- **SC-002**: Alterar o futuro não modifica nenhum sinal calculado no prefixo observado.
- **SC-003**: Todos os controles podem completar treino/avaliação sintéticos sem dados externos.
- **SC-004**: Provas formais compilam sem sorry/admit e testes confirmam gradientes finitos.
- **SC-005**: Nenhum arquivo de referência é alterado e todos os resultados novos identificam sua receita.
- **SC-006**: O experimento permite medir sensibilidade, FA/h e latência; não exige melhoria presumida.

## Assumptions
- Esta entrega implementa uma rodada experimental opt-in, preservando o pipeline
  histórico para comparação; não promove novos resultados automaticamente.
- Falhas em blocos de modalidade são a primeira corrupção suportada. Blur real
  exige reinferência offline e não será simulado por ruído em embeddings.
- Sem novos backbones, sensores, reconstrução 3D, gravidade em pixels ou leis rígidas
  de comprimento 2D. A rodada substitui expansão para novos backbones.
- Mudança de proporção DINOv3, recorte de pessoa, novos negativos/partições e
  calibração de alarmes constituem ablações futuras, não alterações simultâneas.
- Dados, pesos e features reais não acompanham o clone. A entrega inclui fluxo
  executável e validação sintética; eficácia depende de retreino posterior.
