# Documentação do GateFall

O GateFall é um projeto acadêmico de detecção de quedas em vídeo RGB monocular.
O braço A (pose + TCN) está implementado. No braço B (DINOv3), estão
implementados a extração offline de features, o braço B0 (fusão por
concatenação simples com pose) e o braço B1 (fusão adaptativa por gate escalar
sobre os proxies de qualidade). No braço C (SAM 3), estão implementados a
extração offline do descritor de máscara `V_t`, o braço C0 (fusão por
concatenação simples com pose) e o braço C1 (gate adaptativo). A, B0, B1,
C0 e C1 possuem avaliadores de eventos e resultados registrados nas referências.

## Comece aqui

- [Arquitetura](architecture/overview.md): limites entre adapters, dados,
  features, treino e avaliação.
- [Tecnologias](architecture/technology-stack.md): ferramentas, componentes
  implementados e licenças.
- [Organização dos dados](data/organization.md): layout local e migração dos
  caminhos legados.
- [OmniFall](data/omnifall.md) e [Le2i](data/le2i.md): fontes, proveniência e
  preparação.
- [Referência de comandos](reference/commands.md): sintaxe, pré-requisitos,
  efeitos e recursos necessários.
- [Runbook dos pipelines experimentais](runbooks/pipelines.md): reprodução
  completa em um comando ou depuração etapa a etapa.

## Contratos científicos

- [Manifesto e verificação](data/manifest-verification.md)
- [Contrato temporal](data/temporal-contract.md)
- [Padronização de pose](data/pose-standardization.md)
- [Padronização DINOv3](data/dinov3-standardization.md)
- [Treino do braço A](train/baseline-a.md)
- [Treino do braço B0 (fusão pose + DINOv3)](train/baseline-b0.md)
- [Features de qualidade (`q_pose`, `q_visual`)](data/quality-features.md)
- [Treino do braço B1 (fusão adaptativa por gate)](train/baseline-b1.md)
- [Padronização do descritor SAM 3](data/sam3-standardization.md)
- [Treino do braço C0 (fusão pose + SAM 3)](train/baseline-c0.md)
- [Treino do braço C1 (gate adaptativo com SAM 3)](train/baseline-c1.md)
- [Avaliação por eventos do braço A](eval/baseline-a-events.md)
- [Avaliação por eventos do braço B0](eval/baseline-b0-events.md)
- [Avaliação por eventos do braço B1](eval/baseline-b1-events.md)
- [Avaliação por eventos do braço C0](eval/baseline-c0-events.md)
- [Avaliação por eventos do braço C1](eval/baseline-c1-events.md)

Os resultados históricos ficam em `runs/reference/`; novas reproduções ficam
em `runs/local/` e nunca substituem as referências.
