# Contrato: mapeamento de classes

Os rótulos vêm de segmentos temporais OmniFall (revisão `68e5cee56a4bad38cca4aea791cac248f96e79a0`,
config `labels`, filtrada por `dataset`). Nenhuma atividade de origem é mapeada por regra
própria; o rótulo de cada instante é o segmento OmniFall que o cobre.

| OmniFall | Nome | GateFall | Observação |
| --- | --- | --- | --- |
| 0 | walk | 0 `walk` | |
| 1 | fall | 1 `fall` | alarme positivo; início do evento |
| 2 | fallen | 2 `fallen` | alarme positivo; janela de associação |
| 3 | sit_down | 3 `sit_down` | |
| 4 | sitting | 4 `sitting` | |
| 5 | lie_down | 5 `lie_down` | transição voluntária; distinta de `fall` |
| 6 | lying | 6 `lying` | deitado voluntário; distinto de `fallen` |
| 7 | stand_up | 7 `stand_up` | |
| 8 | standing | 8 `standing` | |
| 9 | other | 9 `other` | |
| 10–15 | kneel_down … jump | `IGNORE` (-1) | não suportadas; contadas em `coverage.json`, nunca fundidas em `other` |
| sem segmento | — | `IGNORE` (-1) | lacunas de anotação; nunca preenchidas |

## Regras

- `fall`, `fallen` e `lying` permanecem distintos. Só `fall` e `fallen` disparam alarme.
- Um segmento que termina depois do fim da fonte, além da tolerância de 0,5 s, falha com
  `UnverifiableLabelsError`.
- O fim de cada segmento é exclusivo; a grade usa `labels_for_grid`, a mesma função do Le2i.
- Fontes sem nenhum rótulo, como os ensaios UP-Fall 8-11-2 e 8-11-3, entram em
  `unlabeled_sources` e saem do manifesto.
- Passos desalinhados na sequência de imagens mantêm o rótulo e são marcados `source_gap`.
  Assim, uma lacuna da câmera não parte um evento em dois.
- O macro-F1 primário (`macro_f1_supported`) é a média do F1 sobre as classes com suporte no
  teste avaliado. Esse conjunto é fixado pelos rótulos, nunca pelas predições, e é idêntico para
  todos os métodos do mesmo dataset. O macro-F1 restrito de 8 classes (0–4, 7–9) é publicado
  apenas para comparação histórica.
