# Documentação do GateFall

O GateFall é um projeto acadêmico de detecção de quedas em vídeo RGB monocular.
O objetivo, os experimentos planejados e a instalação estão resumidos no
[README](https://github.com/Arthu085/gate-fall-engine#readme).

Esta documentação concentra os detalhes necessários para preparar e verificar
os dados sem duplicar a visão geral do projeto:

- [Organização dos dados](data/organization.md): diretórios, artefatos,
  versionamento e fluxo de preparação.
- [OmniFall](data/omnifall.md): origem das anotações, snapshot fixado,
  contagens e proveniência.
- [Le2i](data/le2i.md): fonte dos vídeos, download manual, extração e
  correspondência de caminhos.
- [Manifesto e verificação](data/manifest-verification.md): contrato Parquet,
  ingestão e critérios críticos e informativos de verificação.

O repositório ainda está na preparação da camada de dados. A extração de
features, o treino e a avaliação não estão implementados.
