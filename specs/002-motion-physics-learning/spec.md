# Especificação: aprendizado de movimento com coerência cinemática

**Branch**: codex/motion-physics-learning · **Data**: 2026-10-13
**Pedido**: novo fork, especificação e implementação para posterior teste pelo aluno.

## Cenários de uso e aceitação
### US1 — Preparar supervisão (P1)
O aluno reaproveita características existentes e prepara respostas de movimento das poses originais.
Aceitação: alvos futuros no mesmo vídeo; ausências, baixa confiança e finais mascarados; fontes divergentes recusadas; nenhuma janela de classificação excluída por falta de alvo.
### US2 — Comparar três receitas (P1)
Baseline aprende classificação; motion acrescenta posição futura e velocidade média; motion_physics acrescenta concordância entre deslocamento e velocidade.
Aceitação: mesmos dados, detector, inicialização, sementes, minibatches e política de seleção. Coerência produz gradiente na representação compartilhada.
### US3 — Avaliar e explicar (P2)
Aluno executa inferência sem alvos e obtém diferenças pareadas por semente.
Aceitação: seleção exclusivamente por validação, avaliação causal; resumo recusa grupos incompletos, duplicados e configurações incompatíveis.
### Casos de borda
Vídeo curto, finais, pessoa ausente, pontos pouco confiáveis, fonte adulterada, zero supervisão válida, destino existente, protocolos incompatíveis.

## Requisitos
- FR-001: três receitas comparáveis, com mesmas janelas de classificação.
- FR-002: futuro somente como supervisão, nunca entrada do detector.
- FR-003: alvos de ombros/quadris em coordenadas fixas da imagem.
- FR-004: prever posição futura e velocidade média independentemente; cobrar relação deslocamento/tempo. Não impor aceleração zero ou gravidade em pixels.
- FR-005: mascarar observações inválidas sem imputar respostas; registrar cobertura por classe e horizonte no treino.
- FR-006: pesos explícitos e registros separados das perdas.
- FR-007: preservar fontes e referências; hashes verificáveis e destinos novos.
- FR-008: inferência dispensa sidecar e cabeçotes auxiliares.
- FR-009: oferecer preparação, treino, avaliação, matriz pareada, resumo e verificação sintética.
- FR-010: não chamar esta regularização cinemática 2D de PINN completa nem prometer melhoria antes do teste.

## Entidades
Bundle existente, pacote auxiliar de alvos, receita de treino, checkpoints, avaliação e relatório pareado.

## Critérios de sucesso
- SC-001: três receitas completam fluxo sintético com valores finitos.
- SC-002: alvos futuros não alteram logits com entrada/parâmetros fixos.
- SC-003: coerência perfeita tem perda zero; contradição tem perda positiva e gradientes.
- SC-004: janelas sem alvos continuam na classificação.
- SC-005: aluno executa cinco sementes sem editar código.
- SC-006: benefício científico PENDENTE. Comparar principalmente motion_physics menos motion; menor FA/h sem menor sensibilidade média é um sinal operacional favorável, acompanhado de deltas por semente, F1 e latência. Não declarar significância automaticamente.

## Premissas e escopo
RGB e extratores congelados. Cabeçotes auxiliares são a extensão treinável autorizada nesta rodada. Substitui PINN 3D, novos backbones e datasets. Não altera classes ou alarmes. Dados reais serão executados pelo aluno. Não enviar mensagens ao aluno.
