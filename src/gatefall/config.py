"""Constantes experimentais compartilhadas entre as etapas de dados do GateFall."""

# O protocolo de segmentação temporal do OmniFall (arXiv:2505.19889v1, §4.3)
# reamostra os datasets staged para 10 fps; usar o mesmo valor preserva a
# comparabilidade com os números publicados.
TARGET_FPS = 10.0

# Mantido em 10 mesmo que apenas 8 classes tenham suporte no le2i-cs, para que
# a cabeça de classificação permaneça idêntica quando o benchmark for
# estendido; apenas a métrica fica restrita às classes com suporte.
NUM_CLASSES = 10

# Marca timestamps da grade não cobertos por nenhum segmento anotado — 6,72%
# da filmagem do Le2i, concentrados majoritariamente no prefixo de sala vazia
# de Lecture_room e Office.
IGNORE_LABEL = -1

# 2,4 s em TARGET_FPS; cobre o p75 da duração dos segmentos `fall` (2,35 s) e
# dá timesteps suficientes para uma TCN dilatada de 3 níveis, kernel 3, campo
# receptivo 29.
WINDOW_FRAMES = 24

# Em stride 1, janelas de treino consecutivas se sobrepõem em 96% e viram
# quase-duplicatas.
TRAIN_STRIDE = 4

# Uma predição por quadro da grade, o que dá avaliação quadro a quadro e
# resolução de latência de 0,1 s.
EVAL_STRIDE = 1
