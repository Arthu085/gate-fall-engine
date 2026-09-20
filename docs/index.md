# Documentação do GateFall

O GateFall é um projeto acadêmico de detecção de quedas em vídeo RGB monocular.
O braço A (pose + TCN) está implementado; do braço B (DINOv3), a extração
offline de features, a arma B0 (fusão por concatenação simples com pose) e a
arma B1 (fusão adaptativa por gate escalar sobre os proxies de qualidade) já
estão implementadas; o braço C (SAM 3) está planejado.

## Comece aqui

- [Arquitetura](architecture/overview.md): limites entre adapters, dados,
  features, treino e avaliação.
- [Tecnologias](architecture/technology-stack.md): ferramentas implementadas,
  componentes planejados e licenças.
- [Organização dos dados](data/organization.md): layout local e migração dos
  caminhos legados.
- [OmniFall](data/omnifall.md) e [Le2i](data/le2i.md): fontes, proveniência e
  preparação.
- [Referência de comandos](reference/commands.md): sintaxe, pré-requisitos,
  efeitos e recursos necessários.
- [Runbook do pipeline A](runbooks/pipeline-a.md): reprodução completa em um
  comando ou depuração etapa a etapa.

## Contratos científicos

- [Manifesto e verificação](data/manifest-verification.md)
- [Contrato temporal](data/temporal-contract.md)
- [Padronização de pose](data/pose-standardization.md)
- [Padronização DINOv3](data/dinov3-standardization.md)
- [Treino do braço A](train/baseline-a.md)
- [Treino da arma B0 (fusão pose + DINOv3)](train/b0-fusion.md)
- [Features de qualidade (`q_pose`, `q_visual`)](data/quality-features.md)
- [Treino da arma B1 (fusão adaptativa por gate)](train/b1-adaptive-gate.md)
- [Avaliação por eventos](eval/baseline-a-events.md)

Os resultados históricos ficam em `runs/reference/`; novas reproduções ficam
em `runs/local/` e nunca substituem as referências.
