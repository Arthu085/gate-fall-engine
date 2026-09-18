# Generalização entre ambientes (Le2i-CV)

Esta página documenta o protocolo Le2i `cv` (cross-view/cross-environment) e o
run de referência do braço A treinado e avaliado sob ele. Ela relata fatos
medidos; não substitui a leitura do [protocolo `cs`](../train/baseline-a.md),
que é cross-subject e não testa generalização a ambientes não vistos.

## Fonte e isolamento

O protocolo `cv` vem do config OmniFall `le2i-cv`, na **mesma revisão pinada**
já usada pelo `cs` (`68e5cee56a4bad38cca4aea791cac248f96e79a0` — ela serve
`parquet/le2i-cs/*` e `parquet/le2i-cv/*`; não é um fallback para upstream
não pinado). Veja [OmniFall](../data/omnifall.md) para os dois protocolos.

Os artefatos do `cv` são isolados dos equivalentes `cs`, sem compartilhar
arquivo algum:

| Artefato | `cs` | `cv` |
| --- | --- | --- |
| Labels | `data/labels/omnifall/` | `data/labels/omnifall_cv/` |
| Manifesto/grade | `data/processed/le2i/` | `data/processed/le2i_cv/` |
| Estatísticas de padronização | `pose_le2i_cs.json` | `pose_le2i_cv.json` |
| Run local do braço A | `runs/local/le2i/` | `runs/local/le2i_cv/` |

O braço DINOv3 não está disponível sob o `cv`: seus comandos aceitam somente
`--dataset le2i` (ver [features DINOv3](../data/dinov3-features.md)).

Não existe `runs/reference/le2i_cv/`: a promoção de um run a referência nunca
é automática. `gatefall.protocol_isolation_selftest` cobre que os paths do
adapter `cv` nunca caem sob caminhos `cs`, que `load_annotation_splits`
respeita o protocolo pedido, e que a receita de treino congelada do braço A é
idêntica entre protocolos exceto o path/hash de padronização.

## Composição dos splits

O split é disjunto por ambiente e câmera, cobrindo os mesmos 190 vídeos do
Le2i que o `cs` divide de outra forma:

| Split | Segmentos | Vídeos | Ambientes | Câmeras |
| --- | ---: | ---: | --- | --- |
| `train` | 490 | 97 | Coffee_room_01, Coffee_room_02, Lecture_room | 1, 2, 5 |
| `val` | 195 | 33 | Office | 6 |
| `test` | 282 | 60 | Home_01, Home_02 | 3, 4 |

Contagem de janelas (train/val/test): stride 1 total 17905/6607/5982, stride 1
usável 16595/5922/5918, stride 4 usável 4168/1483/1507.

### Le2i-CV não é cross-subject

O split é disjunto por ambiente e câmera, mas os **subjects se sobrepõem**
entre splits: subjects sobrepostos `[0, 1, 3, 5, 7]`. Os ids de subject do
Le2i não são globalmente únicos entre ambientes, então essa sobreposição é
uma propriedade do dataset, não um bug do split. Le2i-CV portanto **não** é
cross-subject — a invariante do repositório de dividir por vídeo, nunca por
janela (`CLAUDE.md`, invariante 2), continua satisfeita, mas não implica
disjunção de subject.

### Classes ausentes do treino

`lie_down` e `lying` têm zero janelas de treino no `cv`. `lie_down` ainda
aparece no teste (1 janela em stride 4, suporte 5 em stride 1). A política de
macro-F1 restrito já exclui os índices dessas duas classes, então a métrica
principal não é diluída por essa ausência — mas ela precisa estar
documentada, e é sinalizada automaticamente pelo relatório de generalização
(`labels_absent_from_train`).

## Fatores de domínio que co-variam entre os splits

Ambiente/câmera, resolução, fps e qualidade de pose **co-variam** entre
train/val/test — este é o ponto central da ressalva de confounding abaixo:

| Fator | train | val | test |
| --- | --- | --- | --- |
| Resolução | 320x240 em 97/97 vídeos | 320x240 em 33/33 vídeos | mista: 320x180 em 25 vídeos, 320x240 em 35 |
| FPS | 25,0 | 25,0 | 24,0003840061441 |
| Cobertura de detecção de pose | 0,9124 | 0,8931 | 0,8898 |
| `q_pose` mediana (p50) | 0,6360 | 0,5814 | 0,5976 |

## Resultados — braço A, seed 42

Recipe, arquitetura, janelamento, otimizador e checkpoint-de-última-época
idênticos ao `cs`; estatísticas de padronização calculadas somente no
`train` do `cv`; protocolo de alarme congelado; sem early stopping,
adaptação ou seleção guiada pelo teste. 30 épocas.

| Split | Macro-F1 restrito (CV) | Macro-F1 restrito (CS, referência) |
| --- | ---: | ---: |
| Treino | 0,8869 | 0,8674 |
| Validação | 0,5645 | 0,6584 |
| Teste | 0,4082 | 0,6231 |

Eventos:

| Split | Sensibilidade | Alarmes falsos | Alarmes falsos/hora | Latência média |
| --- | ---: | ---: | ---: | ---: |
| CV val | 0,8824 (17 eventos) | 24 | 130,77 | 0,5s |
| CV test | 0,9459 (37 eventos) | 15 | 90,27 | 0,6s |
| CS val (referência) | 1,0 (13 eventos) | 0 | 0,00 | 0,4s |
| CS test (referência) | 1,0 (22 eventos) | 10 | 58,37 | 0,5s |

Os dois runs compartilham agora a **mesma geração do pipeline de features**:
a referência do `cs` foi retreinada do zero sob a metodologia corrente (ver
"Migração de referência: pipeline de features atual" em [Treino — Braço A
(TCN)](../train/baseline-a.md#migracao-de-referencia-pipeline-de-features-atual)),
de modo que a defasagem de pipeline que antes confundia esta comparação
deixou de existir.

Resta **um** fator de confounding, e ele basta para impedir a leitura causal:
protocolo e splits diferentes — ambiente/câmera, resolução, fps e qualidade
de pose co-variam entre os splits do `cv`, e o `cs` responde uma pergunta
diferente (cross-subject, não cross-domain). Por isso a comparação CV-vs-CS
continua não sendo uma ablação nem o efeito isolado do domain shift.

## Interpretação científica

Le2i-CV é evidência de generalização entre ambientes, mas é um domain shift
**confundido**: ambiente/câmera, resolução, fps e qualidade de pose
co-variam entre os splits, então nenhuma queda de desempenho observada pode
ser atribuída causalmente a um único desses fatores isolado. A queda
acentuada de macro-F1 entre treino e teste no CV (0,8869 → 0,4082) é
consistente com esse domain shift confundido, mas o relatório não isola qual
fator — ambiente, câmera, resolução, fps ou qualidade de pose — domina o
efeito.

O resultado cross-subject do Le2i-CS **não** deve ser descrito como evidência
cross-domain: o treino do Le2i-CS já contém os seis ambientes do Le2i, então
ele não testa generalização a um ambiente não visto. Os dois protocolos
respondem perguntas diferentes e não são substitutos um do outro. Que os
dois runs compartilhem hoje a mesma geração do pipeline de features remove um
confounding, mas não torna as magnitudes comparáveis: a diferença de
protocolo e de splits permanece e continua sendo suficiente para impedir
atribuir a distância entre CS e CV ao domain shift. Nem as métricas do `cv`
nem sua grade de splits foram usadas para ajustar, selecionar ou promover
qualquer coisa no baseline `cs`.

## Como executar

```bash
uv run python scripts/fetch_labels.py --protocol cv
uv run python -m gatefall.pipeline run --dataset le2i-cv --arm A
```

O relatório de generalização roda como último estágio do pipeline para
`le2i-cv` e também pode ser gerado isoladamente:

```bash
uv run python -m gatefall.eval.generalization_report report \
  --dataset le2i-cv --output runs/local/le2i_cv/baseline_a/generalization_report.json
```

Para `le2i-cv`, o comando falha (código diferente de zero) se ambiente/câmera
deixarem de ser disjuntos entre splits, ou se a contagem de vídeos/segmentos
por split não bater com a revisão pinada — ambos indicam que a configuração
upstream mudou de um jeito que invalida a leitura de generalização acima.
Para `le2i-cs`, a mesma checagem de disjunção de ambiente/câmera é apenas
informativa, porque o protocolo é cross-subject por desenho.
