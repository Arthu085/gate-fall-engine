# Data model

## PreparedBundle
`manifest.json`: schema_version, dataset, visual_kind (none/dinov3/sam3),
preparation_config, sources (hashes da grade e dos arquivos), code_provenance,
lista de vídeos com id, split, caminho relativo,
hash do HDF5, número de quadros. Partições permitidas: train/val/test, disjuntas por vídeo.
Dados persistidos em HDF5 por vídeo; nenhum caminho pode escapar da raiz do pacote.

## VideoFeatures
`labels [K]`, `pose [K,138]`, `visual [K,V]`, `evidence [K,8]`;
`pose_corrupt [K,138]`, `visual_corrupt [K,V]`, `evidence_pose_corrupt [K,8]`,
`evidence_visual_corrupt [K,8]`, `pose_changed [K]`, `visual_changed [K]`.
V=0/1536/10. Evidência e flags finitas em [0,1]. Colunas de evidência:
q_pose,q_visual,r_pose,r_visual,r_pose_valid,r_visual_valid,pose_present,visual_present.
Variantes armazenadas apenas no treino; validação/teste limpos.

## Standardization
Média e desvio dos canais contínuos estimados sobre quadros únicos do treino;
flags/qualidades sem padronização. Guardar contagens e versão. Em entradas ausentes,
zerar representação padronizada da modalidade e manter flags explícitas.

## ExperimentConfig / Result
schema_version, arm, fusion, seed, epochs, batch_size, lr, weight_decay,
checkpoint_policy (best_val/last), augment, order_weight, order_margin,
TCN fixa, feature_schema, manifest_sha256, stat hashes e label names.
Margem em (0,1), peso finito >=0; ordem exige augment e modo com gate treinável.
Dois checkpoints (last.pt, checkpoint.pt), histórico completo, época selecionada,
config/stats/metrics JSON. Avaliação explícita salva reports por split sem overwrite,
incluindo predições por vídeo/frame. Proveniência registra revisão Git, estado
modificado e hashes do código Python em preparação, treino e avaliação.

## FormalProof
Projeto Lean fixado, sem sorry/admit; README mapeia cada teorema à função Python
e explica hipóteses e fronteira entre números reais e ponto flutuante.
