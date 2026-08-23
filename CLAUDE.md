# CLAUDE.md — GateFall

Este arquivo existe **apenas** para registrar o que diverge de
`~/.claude/CLAUDE.md` e as invariantes experimentais do projeto. Tudo que não
estiver declarado aqui continua valendo integralmente a partir do arquivo
global, sem repetição.

---

## Declared overrides

Cada item nomeia a regra global que substitui e dá a justificativa. O escopo do
override é a regra nomeada — nada além dela é afetado.

- **Docs e comentários em português.** Substitui a regra de idioma do global
  (§5, "Docs and code comments: English by default"). Código, identificadores,
  mensagens de commit e títulos/descrições de PR permanecem em inglês — isso o
  global não permite sobrescrever, e este repositório não sobrescreve.
  *Justificativa:* é um TCC avaliado por banca brasileira.

- **Comentários são permitidos, sem restrição formal, mas usados com
  moderação.** Substitui integralmente a proibição absoluta do global (§6, "No
  comments in production code, ever"). A régua é esta: código saturado de
  comentário é código mal fatorado — antes de comentar, tente um nome melhor ou
  uma função menor. O comentário que se paga é o que ancora fórmula, convenção
  de sinal, ordem de eixos/canais, unidade, ou a referência do artigo de onde o
  descritor veio.
  *Justificativa:* código científico e numérico carrega significado que nenhum
  identificador sustenta sozinho.

- **PR obrigatória apenas para mudanças sob `src/`.** Substitui a exigência
  abrangente de PR do pipeline global (§3). Commits diretos são permitidos em
  `notebooks/` e `scripts/`.
  *Justificativa:* exploração descartável não paga o custo de uma PR.

- **Nenhum teste é exigido; o passo `test-writer` do `code-workflow` está
  desabilitado neste repositório.** Substitui o passo obrigatório de testes do
  pipeline global (§3); o `implementer` verifica contra o gate do repositório e
  nada mais muda na sequência.
  *Justificativa:* a validação aqui é experimental — métricas sob protocolo
  fixo —, não unitária.

---

## Invariantes experimentais

Não são overrides: são regras de domínio do projeto. Violá-las invalida o
resultado, não apenas o estilo do código.

1. **A, B e C são idênticas exceto no conteúdo de cada linha da janela.**
   Tamanho de janela, split treino/teste, seed, codificador temporal e número de
   épocas são os mesmos nas três configurações; só muda o vetor de features por
   timestep. Qualquer alteração que afete apenas uma delas invalida a
   comparação — se precisar mudar, muda nas três e re-roda as três.

2. **O split treino/teste é sempre por vídeo, preferencialmente por sujeito.
   Nunca por janela.** Janelas do mesmo vídeo nos dois lados do split vazam
   informação e inflam a métrica.

3. **O pipeline é RGB monocular.** Nenhum descritor derivado de mapa de
   profundidade entra em qualquer braço — incluindo as colunas D–K do CSV de
   features do URFD, que são calculadas a partir do sensor de profundidade do
   Kinect.

4. **Backbones congelados, features pré-computadas offline.** YOLO-Pose, DINOv3
   e SAM 3 não são treinados nem ajustados. Treinam apenas a cabeça de fusão e a
   TCN.

5. **Features são agrupadas em HDF5 por vídeo.** Nunca um arquivo por frame.

---

## Risco dominante

O risco dominante deste projeto é **escopo, não dificuldade técnica**. Toda
adição proposta — uma quarta configuração, um dataset extra, mais um backbone,
mais uma métrica — precisa declarar **o que ela desloca em troca**. Sem essa
contrapartida explícita, a resposta padrão é não.
