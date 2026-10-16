# Investigação não canônica de perda e confusão de classes

Esta investigação usa apenas as linhas `train` oficiais. Em Le2i-CS, as três
dobras existentes retêm sujeitos; em Le2i-CV, as três dobras existentes retêm
ambientes. O conjunto oficial `test` não seleciona receita, seed, checkpoint,
limiar ou regra de parada. O checkpoint é sempre o da última época. A métrica
restrita mantém as oito classes existentes.

## Critérios registrados antes dos treinos

As quatro receitas têm as mesmas entradas, arquitetura, janelas, 30 épocas,
seed, dobra e avaliação. `baseline` usa a entropia cruzada ponderada original.
`soft` usa pesos `n_c^-0,5`, calculados apenas nas janelas de treino da dobra.
`hard` acrescenta à perda original até metade da contribuição dos exemplos de
treino rotulados `other` ou `walk` que o modelo classifica como a outra classe.
`combined` aplica ambas as mudanças. O termo extra não repete janelas nem
usa rótulos do retido. Classes ausentes recebem peso zero, como na receita
original.

A seed 42 é o piloto para cada protocolo. Uma receita avança para as seeds
42–46 somente se melhora a Macro-F1 restrita pareada em pelo menos duas das
três dobras, com média das três diferenças de pelo menos 0,020, sem queda
maior que 0,050 na revocação de `fall` ou `fallen` em nenhuma dobra. A
decisão final exige mediana das diferenças das quinze comparações de pelo
menos 0,020, mediana positiva em pelo menos duas dobras, e ausência de queda
maior que 0,050 na revocação média de `fall` ou `fallen` em qualquer dobra.
Também exige redução dos falsos positivos combinados de `other` e `walk` sem
queda maior que 0,050 na revocação de `standing`. Quando uma avaliação de
eventos por dobra estiver validada, exige sensibilidade de eventos sem queda
maior que 0,050 e nenhum aumento superior a 10% em falsos alarmes por hora
ou 0,5 s na latência mediana. Sem essa avaliação, nenhuma receita é promovida
com base somente nas métricas de janela.

As contagens de janelas sobrepostas não são observações independentes. Três
grupos de sujeitos ou ambientes não permitem inferir generalização ampla.
Um resultado neste conjunto é exploratório, pois o `test` oficial já foi
exposto em análises anteriores. A meta 0,800 é aspiracional e não altera os
critérios. B0, B1, C0 e C1 só serão treinados com a mesma receita se o piloto
de A satisfizer os critérios. Essa expansão substitui novas investigações
de backbone ou de amostragem; não cria uma quarta configuração canônica.

## Execução e resultados

Prepare as dobras com os comandos já documentados em
[Le2i-CS](../train/subject-folds.md) e
[Le2i-CV](../train/internal-folds.md). `pilot` treina as quatro receitas na
seed 42, na ordem controle, `soft`, `hard`, `combined`, em cada dobra.
`multiseed` lê a decisão do piloto e executa as seeds 43–46 somente para as
receitas aprovadas e seus controles pareados.

```bash
uv run python -m gatefall.train.imbalance_experiment pilot \
  --protocol cs \
  --fold-root runs/local/le2i/subject_folds

uv run python -m gatefall.train.imbalance_experiment pilot \
  --protocol cv \
  --fold-root runs/local/le2i_cv_internal_folds

uv run python -m gatefall.train.imbalance_experiment multiseed \
  --protocol cs --fold-root runs/local/le2i/subject_folds
uv run python -m gatefall.train.imbalance_experiment multiseed \
  --protocol cv --fold-root runs/local/le2i_cv_internal_folds

uv run python -m gatefall.train.imbalance_experiment report --protocol both
uv run python -m gatefall.train.imbalance_experiment.selftests
```

O subcomando `train` permite retomar uma única combinação com `--protocol`,
`--fold-root`, `--fold`, `--recipe` e `--seed`. `--force` refaz somente um run experimental dentro de
`runs/local/imbalance_experiment/`. O arquivo
`runs/local/imbalance_experiment/comparison.json` contém cada run, a matriz de
confusão, precisão, revocação, F1, FP e FN por classe, diferenças pareadas,
decisões, ausências e limitações. CS e CV aparecem em objetos separados.

Todos os artefatos ficam em `runs/local/` e não substituem `runs/reference/`
nem baselines anteriores.

## Piloto Le2i-CS

As 12 execuções da seed 42 terminaram nas três dobras por sujeito. Diferenças
de Macro-F1 restrita contra o controle da mesma dobra:

| Dobra retida | Controle | `soft` | `hard` | `combined` |
| --- | ---: | ---: | ---: | ---: |
| Sujeito 0 | 0,5584 | −0,0146 | +0,0064 | +0,0017 |
| Sujeito 1 | 0,6576 | −0,0412 | −0,0007 | −0,0308 |
| Sujeitos 3, 4, 6 e 8 | 0,6157 | +0,0087 | −0,0020 | +0,0112 |
| Média das diferenças | — | −0,0157 | +0,0012 | −0,0060 |

Nenhuma receita atinge a melhora média mínima de 0,020. `soft` e `hard`
melhoram só uma dobra; `combined` melhora duas, mas com média negativa. A
guarda de revocação de `fall` e `fallen` passou nas três receitas. As seeds
43–46 não serão executadas em CS pelo critério registrado. A melhor Macro-F1
observada foi 0,6576 no controle; 0,800 não foi observado neste piloto.

Média das diferenças pareadas de F1 por classe, com a variação total dos
falsos positivos de `other` e `walk` nas três dobras:

| Receita | `standing` | `other` | `walk` | `fall` | `fallen` | Δ FP `other`+`walk` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soft` | −0,1502 | +0,0265 | +0,0061 | +0,0030 | −0,0067 | +309 |
| `hard` | +0,0345 | −0,0066 | −0,0046 | +0,0019 | +0,0012 | +31 |
| `combined` | −0,1040 | +0,0271 | +0,0045 | +0,0057 | −0,0047 | +333 |

Os totais de FP contam janelas sobrepostas. `soft` e `combined` pioram
`standing` e aumentam os FP das duas classes de confusão. `hard` melhora
`standing`, mas não reduz os FP combinados nem melhora a Macro-F1 de forma
relevante.

## Piloto Le2i-CV

As 12 execuções da seed 42 terminaram nas três dobras por ambiente. Diferenças
pareadas de Macro-F1 restrita:

| Ambiente retido | Controle | `soft` | `hard` | `combined` |
| --- | ---: | ---: | ---: | ---: |
| `Coffee_room_01` | 0,6119 | +0,0114 | −0,0072 | +0,0115 |
| `Coffee_room_02` | 0,6094 | +0,0059 | −0,0082 | −0,0097 |
| `Lecture_room` | 0,4471 | +0,0378 | −0,0010 | +0,0392 |
| Média das diferenças | — | +0,0184 | −0,0055 | +0,0137 |

`soft` melhora as três dobras, mas a média de +0,0184 fica abaixo do limiar
registrado de +0,0200. `hard` não melhora nenhuma. `combined` melhora duas,
mas sua média também fica abaixo do limiar. A guarda de revocação de `fall`
e `fallen` passou. As seeds 43–46 não serão executadas em CV.

Média das diferenças pareadas de F1 por classe e variação total dos FP de
`other` e `walk`:

| Receita | `standing` | `other` | `walk` | `fall` | `fallen` | Δ FP `other`+`walk` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `soft` | −0,0183 | +0,0230 | +0,0942 | +0,0244 | +0,0282 | +82 |
| `hard` | −0,0383 | −0,0036 | +0,0181 | +0,0044 | −0,0026 | +177 |
| `combined` | −0,0486 | +0,0081 | +0,0961 | +0,0217 | +0,0103 | +210 |

Mesmo em CV, nenhuma receita reduz os FP combinados. A revocação média de
`standing` cai 0,0650 com `soft` e 0,0867 com `combined`; isso também impede
promoção pelo critério final. A maior Macro-F1 observada foi 0,6234.

## Decisão e limites

**Nenhuma receita é promovida.** As 24 execuções reais do piloto cobrem as
quatro receitas, três dobras e os dois protocolos na seed 42. Nenhuma receita
passou para múltiplas seeds. Por isso B0, B1, C0 e C1 não foram treinados, e
nenhum artefato canônico mudou. O comando `multiseed` confirmou a regra de
parada nos dois protocolos.

O alvo 0,800 **não foi observado**: o máximo foi 0,6576 em CS, no controle da
dobra que retém o sujeito 1, e 0,6234 em CV, em `combined` na dobra que retém
`Coffee_room_01`. São resultados exploratórios da seed 42 em dobras do
`train` oficial, não validação independente. Há só três grupos retidos em
cada protocolo, com muitos quadros e janelas correlacionados. O `test`
oficial já era conhecido por estudos anteriores e não foi lido nesta rodada.

Uma repetição independente de `baseline` e `soft` na dobra CS do sujeito 0,
com a mesma seed 42, produziu checkpoints com SHA-256 idênticos aos da rodada
principal (`c994f116f270…` e `65a933ec769e…`, respectivamente). O controle
também coincidiu byte a byte com o run histórico da Etapa 4B em CS; o
controle da dobra CV `Coffee_room_01` coincidiu com o da Etapa 2C. Esses
pares verificam o determinismo observado nesta máquina, sem provar
reprodutibilidade entre hardware ou versões de bibliotecas.

Sensibilidade de eventos de queda, falsos alarmes por hora e latência ficam
`null` no relatório: o protocolo validado de eventos cobre os splits oficiais,
mas ainda não foi validado para estas dobras retidas. Não há evidência de
preservação da sensibilidade de eventos; a guarda de revocação de janelas de
`fall` e `fallen` não a substitui. Nenhuma decisão usa o `test` oficial.

O relatório JSON consolidado fica em
`runs/local/imbalance_experiment/comparison.json`. Ele conserva todas as
matrizes de confusão, precisão, revocação, F1, FP, FN, suportes, hashes,
diferenças pareadas e combinações não executadas. Os arquivos de resultado
locais não são versionados; os comandos acima os reproduzem.
