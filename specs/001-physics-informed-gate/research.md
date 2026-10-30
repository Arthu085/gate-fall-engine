# Research: decisões e evidências

## Compatibilidade e cinemática
**Decision**: runner opt-in com features v2 e estatísticas próprias.
**Rationale**: o bug de aceleração foi reproduzido no commit ef6b3c1; mudar o
carregador histórico silenciosamente invalidaria checkpoints e stats publicados.
**Alternatives considered**: substituir default global (rejeitado por incompatibilidade).

## Consistência geométrica
**Decision**: residual de similaridade robusta 2D e continuidade suave de área/preenchimento SAM.
**Rationale**: as fontes disponíveis são PoseArrays e V_t; V_t não contém máscaras
para IoU e DINOv3 não fornece geometria física diretamente. Uma revisão de pesquisa
independente confirmou essas limitações. Flags distinguem residual indisponível de zero medido.
**Alternatives considered**: gravidade em pixels, comprimento 2D rígido, magnitude
de aceleração como erro (rejeitados por projeção e risco de suprimir quedas reais).

## Aprendizado de adaptação
**Decision**: falhas em blocos de modalidade, CE em variantes e hinge com margem positiva.
**Rationale**: residual fixo somado à CE tem gradiente zero; hinge sem margem pode
ser identicamente nulo num gate monotônico e permite gate constante.
**Alternatives considered**: ruído arbitrário nos descritores (não equivale a
corrupção física); blur de embeddings (inválido); blur RGB com backbone (ablação futura).

## Verificação formal
**Decision**: Lean/Mathlib fixados, teoremas reais para sigmoid, gate, perda e derivadas.
**Rationale**: provas são complementares aos selftests e não certificam PyTorch,
aritmética float32, calibração, validade das proxies ou eficácia em dados reais.
**Alternatives considered**: somente exemplos numéricos (insuficientes para propriedades universais).

## Referências primárias
- PhysCap: https://vcai.mpi-inf.mpg.de/projects/PhysCap/
- PhysPT: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_PhysPT_Physics-aware_Pretrained_Transformer_for_Estimating_Human_Dynamics_from_Monocular_CVPR_2024_paper.html
- Lean: https://lean-lang.org/install/manual/
- Spec Kit oficial: commit 7666b7463ffc8b3dce49f75be017f163a46792fa.

Nenhuma referência constitui prova de melhoria do GateFall. Todas as decisões
necessárias à implementação estão resolvidas; eficácia será medida por ablação.
