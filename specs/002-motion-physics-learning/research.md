# Decisões de pesquisa
- Cinemática 2D auxiliar: dados sustentam trajetória aparente, não forças/gravidade métrica. PINN 3D fora do escopo.
- Keypoints [5,6,11,12] divididos pela diagonal fixa da imagem. A caixa variável introduziria movimento artificial. Revisão independente motion_design_review confirmou desenho.
- Velocidade média: residual p_futura_prevista - p_atual_observada - dt*v_media_prevista. Não pressupõe velocidade constante.
- Duas saídas independentes: derivar velocidade da própria posição prevista tornaria o residual tautológico.
- Mesmos cabeçotes e inicialização em todas as receitas: isolar tarefa extra da restrição física.
- Precedente: PhysPT, CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_PhysPT_Physics-aware_Pretrained_Transformer_for_Estimating_Human_Dynamics_from_Monocular_CVPR_2024_paper.html. Inspiração geral, não reprodução nem garantia.
