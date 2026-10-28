# Aprender movimento com coerência cinemática

## Em linguagem simples

O GateFall já usa movimento como entrada. Nesta experiência, ele também precisa
**prever movimento durante o treinamento**. Dois cabeçotes aprendem separadamente
onde estarão ombros/quadris e qual será sua velocidade média nos próximos instantes.
Uma penalização adicional cobra concordância entre essas duas previsões.

Não é uma PINN de dinâmica 3D. É supervisão auxiliar com regularização cinemática
2D. Não medimos gravidade, forças, equilíbrio ou contato com o chão. Ainda não há
resultado real demonstrando melhoria desta versão.

## O que muda

| Receita | Classificação | Previsão de movimento | Coerência cinemática |
|---|---|---|---|
| baseline | sim | perda desligada | desligada |
| motion | sim | sim | desligada |
| motion_physics | sim | sim | sim |

As três receitas têm os mesmos cabeçotes e pesos iniciais. Na baseline, os
cabeçotes auxiliares ficam sem gradiente e não alteram o detector. As sementes,
minibatches, entradas, TCN, épocas, augmentation e seleção são iguais.
O controle baseline reproduz o treino anterior para a mesma configuração.
Não comparar esta baseline diretamente com a versão histórica pose134 para
atribuir causalidade à nova perda.

O teste principal da contribuição física é **motion_physics menos motion**.
Comparar somente motion_physics com baseline confundiria física e tarefa extra.

## Dados e alvos

Reutilizar bundles preparados por `gatefall.experiments.physics prepare` e
os HDF5 originais de pose YOLO. Não baixar novamente os vídeos nem executar
YOLO/DINO/SAM. O pacote auxiliar (sidecar) contém somente alvos de treino.
Ele não altera o bundle e verifica o SHA-256 da pose original contra sua
proveniência. Caminhos podem mudar, conteúdos não.

Precisamos dos HDF5 originais porque o bundle antigo não guarda os metadados
necessários para recuperar com segurança as coordenadas isotrópicas da imagem.
Se a pose original foi reextraída e o hash mudou, preparar outro bundle compatível;
não remover a verificação.

- Pontos COCO: ombro esquerdo/direito e quadril esquerdo/direito (5, 6, 11, 12).
- Coordenadas: x e y divididos pela mesma diagonal fixa da imagem.
- Horizontes padrão: 1 e 3 quadros a 10 Hz, isto é, 0,1 e 0,3 s.
- Presença e confiança >=0,5 nos dois endpoints, por articulação; valores finitos.
- Finais de vídeo e alvos inválidos são mascarados, sem forward-fill.
- A velocidade média pode ser definida entre endpoints válidos mesmo com
  ausência intermediária; não estimamos a trajetória interna desse intervalo.
- Toda janela de classificação permanece, mesmo sem alvo auxiliar.
- Se não houver supervisão válida em nenhuma janela utilizável de treino, abortar.
- Cobertura por classe/horizonte é registrada: conferir especialmente a classe fall.

## Função de treinamento

O estado final causal da TCN alimenta classificação e os dois cabeçotes lineares.
Se p0 é a posição atual observada, p1 a posição futura e dt o horizonte:

```text
velocidade-alvo = (p1 - p0) / dt
Lmov = Huber((p1_prevista - p1) / s)
     + Huber(dt * (v_prevista - velocidade-alvo) / s)
Lcin = Huber((p1_prevista - p0 - dt * v_prevista) / s)
Ltotal = Lclass + lambda_mov * Lmov + lambda_cin * Lcin
```

s = 0,1 diagonal da imagem; ambas as parcelas de movimento e a coerência
estão em unidades espaciais comparáveis. Huber usa beta=1, redução pela
quantidade de coordenadas válidas. Batch sem alvos contribui zero finito.
Pesos nominais padrão: 0,1 e 0,1, desligados conforme a receita.

A identidade utiliza **velocidade média**, não exige velocidade constante nem
penaliza aceleração brusca. As duas previsões são independentes; derivar uma
automaticamente da outra eliminaria o sinal de regularização.
São medidas em imagem, não metros por segundo. Movimento de câmera/perspectiva
e erros do YOLO continuam sendo limitações. A perda acopla previsões redundantes,
não acrescenta uma nova observação independente do corpo.

Com `--augment`, a classificação também vê ausências; a supervisão de movimento
é calculada somente sobre a entrada limpa, igualmente nas receitas auxiliares.
A perda de ordenação do gate é opcional e idêntica nas três receitas quando usada.

Na inferência, apenas o cabeçote de classificação é executado. Nenhum alvo,
quadro futuro ou espera pelo horizonte é necessário.

## Instalação e verificação

O módulo foi importado do fork
[rsramos30/gate-fall-motion-physics](https://github.com/rsramos30/gate-fall-motion-physics)
no commit `5f6e2ae`, que já contém o gate físico (`301b32e`). Ambiente
recomendado: Linux/WSL, Python 3.12 e uv, a partir da raiz deste repositório.

```bash
uv sync --frozen
uv run python -m gatefall.experiments.motion selftest
```

Isso verifica engenharia; não mede desempenho real nem treina os extratores.

## Primeira comparação: manter o gate físico e mudar somente o aprendizado

Os caminhos abaixo são exemplos; substituir pelos diretórios existentes.
O run primário predefinido é **C1/physics**, sem augmentation nem order,
30 épocas, best_val e sementes 42–46, separadamente em CS e CV.
As três receitas são novas execuções; não usar checkpoints antigos como um
dos controles da comparação.

```bash
uv run python -m gatefall.experiments.motion prepare --bundle /dados/bundles/sam-cs --pose-root /dados/pose --output runs/local/le2i/motion-targets-sam
uv run python -m gatefall.experiments.motion compare --bundle /dados/bundles/sam-cs --targets runs/local/le2i/motion-targets-sam --output runs/local/le2i/motion-c1 --arm C1 --fusion physics
uv run python -m gatefall.experiments.motion prepare --bundle /dados/bundles/sam-cv --pose-root /dados/pose --output runs/local/le2i_cv/motion-targets-sam
uv run python -m gatefall.experiments.motion compare --bundle /dados/bundles/sam-cv --targets runs/local/le2i_cv/motion-targets-sam --output runs/local/le2i_cv/motion-c1 --arm C1 --fusion physics
```

Cada compare produz 15 treinos (3 receitas x 5 sementes), treina todos antes
de avaliar o teste e grava um resumo descritivo. Dois protocolos: 30 treinos.
Não alterar pesos/horizontes após observar teste. Uma exploração posterior deve
ser escolhida usando validação e identificada separadamente.

## Aplicar também aos cinco braços

A extensão é compartilhada; para a matriz A/B/C completa, repetir compare com:

| Braço | Bundle | --fusion | Diretório de saída |
|---|---|---|---|
| A | SAM do protocolo (visual ignorado) | pose | motion-a |
| B0 | DINO do protocolo | concat | motion-b0 |
| B1 | DINO do protocolo | physics | motion-b1 |
| C0 | SAM do protocolo | concat | motion-c0 |
| C1 | SAM do protocolo | physics | motion-c1 |

Preparar sidecar próprio para cada bundle DINO, assim como os dois SAM acima;
o hash liga cada sidecar ao seu bundle. A pode reutilizar o bundle/sidecar SAM
porque seu classificador ignora a entrada visual. Manter mesmos parâmetros em
todos os braços. Matriz completa: 150 treinos (5 braços x 3 receitas x 5 sementes
x 2 protocolos), incluindo os 30 primários. Não repetir C1 se já concluído.

Outros gates continuam disponíveis, mas representam outra comparação; não
trocar o gate entre as três receitas de uma mesma comparação.

## Execução individual e retomada

Destinos existentes são recusados. Um treino interrompido antes de concluir
não publica o run; não há retomada de época. Os runs concluídos são preservados.
Se compare interromper, executar individualmente somente os runs faltantes com
os mesmos parâmetros; depois avaliar os concluídos sem evaluation-test.json.

```bash
uv run python -m gatefall.experiments.motion train --bundle /dados/bundles/sam-cs --targets runs/local/le2i/motion-targets-sam --run-dir runs/local/le2i/motion-c1/seed-42/motion_physics --arm C1 --fusion physics --recipe motion_physics --seed 42
uv run python -m gatefall.experiments.motion evaluate --bundle /dados/bundles/sam-cs --run-dir runs/local/le2i/motion-c1/seed-42/motion_physics
uv run python -m gatefall.experiments.motion summarize --runs runs/local/le2i/motion-c1/seed-42/baseline runs/local/le2i/motion-c1/seed-42/motion runs/local/le2i/motion-c1/seed-42/motion_physics --output runs/local/le2i/resumo-seed42
```

Resumo de cinco sementes: passar os 15 diretórios em --runs. Mesmo protocolo,
braço, bundle, alvos, pesos, código de treino/avaliação, ambiente e suporte;
três receitas por semente. A comparação verifica dispositivo, versões de PyTorch
e NumPy e precisão temporal da avaliação.
Apenas comparar receitas com a mesma configuração. Em Windows/PowerShell,
usar os mesmos comandos em uma linha e caminhos entre aspas quando necessário.

## O que entregar ao orientador

- `summary/summary.json`: médias, DP entre sementes e diferenças pareadas.
- `summary/runs.csv` e `paired-deltas.csv`: planilhas das execuções e deltas.
- Por run: config, stats, metrics, checkpoint, last e evaluation-test.
- Manifests do bundle e do sidecar, revisão Git e ambiente usado.

Na leitura: FA/h menor é melhor; sensibilidade e F1 maiores são melhores.
Latência menor só é comparável considerando também quais quedas foram detectadas.
DP de cinco sementes não é intervalo de confiança entre pessoas. Resumo não
executa teste estatístico nem seleciona vencedor.

O principal resultado favorável seria motion_physics reduzir FA/h sem reduzir
sensibilidade média frente a motion, com comportamento consistente nas sementes.
Mostrar sempre F1, latência e todos os deltas, inclusive perdas. Se só Lcin
melhorar, a hipótese de melhoria na detecção permanece sem confirmação.
Nenhuma meta de uso clínico ou produção é certificada por essa rodada.

## Texto simples para o TCC

“Além de reconhecer atividades, o GateFall foi adaptado para aprender a prever
o movimento corporal. Durante o treinamento, estimativas independentes de
posição futura e velocidade média são incentivadas a respeitar sua relação
cinemática. Comparamos o detector original, a supervisão adicional de movimento
e essa supervisão acrescida da restrição, sob as mesmas condições experimentais.”

Referência conceitual: [PhysPT, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_PhysPT_Physics-aware_Pretrained_Transformer_for_Estimating_Human_Dynamics_from_Monocular_CVPR_2024_paper.html).
Esta implementação não reproduz o modelo 3D ou as forças do PhysPT.
