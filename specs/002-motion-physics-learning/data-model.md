# Contrato de dados
Sidecar motion-manifest.json: schema motion-v1, hash manifest physics, horizontes inteiros positivos distintos, confiança [0,1], fps, normalização image_diagonal, vídeos somente train e hashes raw/targets.
Por vídeo: current[K,4,2], future[K,H,4,2], velocity[K,H,4,2], valid[K,H,4]. Float32 finito/bool; inválidos zerados. Endpoints presentes e confiantes. Ausência intermediária não altera a definição de velocidade média dos endpoints; não se afirma trajetória intermediária observada.
Config motion: receita, pesos nominais, escala fixa, horizontes, hash sidecar, inicialização e cobertura nas janelas efetivamente treinadas.
Resumo: mesmas fontes, detector, política, sementes e suporte vídeo/quadro/rótulo; não combinar protocolos ou receitas distintas.
