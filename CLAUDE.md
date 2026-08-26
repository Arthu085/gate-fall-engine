# CLAUDE.md — GateFall

Este arquivo reúne as convenções e as invariantes experimentais que devem ser seguidas em qualquer
trabalho neste repositório. Estas instruções são autocontidas e não pressupõem arquivos de
contexto, skills ou agentes personalizados instalados fora do repositório.

## Convenções do repositório

- **Documentação, comentários de código e descrições de PR usam português brasileiro.** Código,
  identificadores, mensagens de commit e títulos de PR permanecem em inglês.
- **Comentários são permitidos, mas devem ser usados com moderação.** Prefira primeiro um nome mais
  claro ou uma função menor. Preserve comentários que registrem uma fórmula, convenção de sinal,
  ordem de eixos ou canais, unidade ou o artigo do qual um descritor foi derivado.
- **Testes automatizados não são obrigatórios.** Não pressuponha a existência de um framework de
  testes nem introduza um apenas para atender a uma alteração que não exige testes.
- **Alterações de documentação seguem a responsabilidade de cada superfície.** Atualize somente a
  superfície que ficou incorreta. Mantenha o `README.md` limitado à visão geral do projeto,
  instalação, uso básico e links; registre detalhes técnicos na página correspondente do MkDocs.
  Não duplique a mesma explicação nas duas superfícies.
- **Toda adição, modificação ou remoção de código Python exige `uv run pyright`.** Execute essa
  verificação antes de considerar a alteração concluída.

## Invariantes experimentais

1. **A, B e C são idênticas, exceto pelo conteúdo de cada linha na janela.** O tamanho da janela, a
   divisão entre treino e teste, a semente, o codificador temporal e o número de épocas são
   idênticos entre as configurações; somente o vetor de características por instante muda. Aplique
   qualquer alteração compartilhada necessária às três configurações e execute novamente as três.
2. **Divida os dados de treino e teste por vídeo, preferencialmente por sujeito, nunca por janela.**
   Janelas do mesmo vídeo nos dois conjuntos causam vazamento de informação e inflam as métricas.
3. **O pipeline é RGB monocular.** Nenhum descritor derivado de profundidade pode entrar em qualquer
   ramo, inclusive as colunas D–K do CSV de características do URFD produzido a partir da
   profundidade do Kinect.
4. **Mantenha os backbones congelados e pré-compute as características offline.** Não treine nem
   faça ajuste fino de YOLO-Pose, DINOv3 ou SAM 3. Quando o treinamento for implementado, treine
   somente a cabeça de fusão e a TCN.
5. **Ao implementar o armazenamento em HDF5, agrupe as características por vídeo.** Nunca crie um
   arquivo por frame.

## Risco dominante

O risco dominante é o escopo, não a dificuldade técnica. Toda adição proposta — uma quarta
configuração, outro conjunto de dados, outro backbone ou outra métrica — deve declarar
explicitamente o que será retirado para abrir espaço. Sem essa troca, a resposta padrão é não.
