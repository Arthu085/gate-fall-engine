# Documentação do GateFall

O GateFall é um projeto acadêmico de detecção de quedas em vídeo RGB monocular.
O braço A (pose + TCN) está implementado; os braços B (DINOv3) e C (SAM 3)
estão planejados.

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
- [Treino do braço A](train/baseline-a.md)
- [Avaliação por eventos](eval/baseline-a-events.md)

Os resultados históricos ficam em `runs/reference/`; novas reproduções ficam
em `runs/local/` e nunca substituem as referências.
