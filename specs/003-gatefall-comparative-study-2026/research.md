# Pesquisa e decisões

## Fontes externas e licenças

| Fonte | Revisão | Licença | Uso |
| --- | --- | --- | --- |
| `rsramos30/gate-fall-engine` (`codex/physics-informed-gate`) | `301b32e` | MIT | `gatefall.experiments.physics` e spec 001 importados. |
| `rsramos30/gate-fall-motion-physics` | `5f6e2ae` (descende de `301b32e`) | MIT | `gatefall.experiments.motion`, `eval/shared/events.py` (`round_latency`) e spec 002 importados. |
| `SomeOtherScenery/fall_detection` | `23b7d32` | **nenhuma declarada** | Lido como referência de arquitetura; nenhum código copiado. |
| OmniFall (Hugging Face) | `68e5cee56a4bad38cca4aea791cac248f96e79a0` | do dataset | Rótulos `up_fall`, `GMDCSA24` e splits de referência. |

O projeto Lean (`formal/`) e as ferramentas Spec Kit dos forks permanecem nos repositórios de origem.

## UP-Fall

- O dataset tem 17 sujeitos, 11 atividades, 3 ensaios e 2 câmeras, num total de 1118 vídeos de
  câmera rotulados (559 ensaios). As fontes são sequências de imagens com timestamp no nome do
  arquivo.
- Os ensaios (8, 11, 2) e (8, 11, 3) não têm rótulo no OmniFall. Eles são registrados como
  `unlabeled_sources` e excluídos do manifesto.
- **O split `up_fall-cs` do OmniFall não é disjunto por sujeito.** Os sujeitos 11, 15 e 16
  aparecem em `train` e em `val`/`test` com os mesmos caminhos, o que viola o invariante 2.
- A partição `upfall-subject-v1` mantém os sujeitos de `val`/`test` designados pelo OmniFall e os
  remove de `train`:
  - `train`: 1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 17;
  - `val`: 11;
  - `test`: 2, 4, 15, 16.
- Com quatro sujeitos de teste, o IC por sujeito é largo. O relatório também publica o IC por
  gravação (ensaio, com as duas câmeras juntas) como análise secundária, e nenhum dos dois é
  apresentado como mais preciso do que é.

## GMDCSA-24

- O dataset tem 4 sujeitos e 160 clipes MP4 (`Subject N/{ADL,Fall}/NN.mp4`). É usado só como teste
  externo, e nenhum sujeito entra em treino ou validação.
- O namespace é `data/processed/gmdcsa24_external` e `data/features/gmdcsa24_external`, separado do
  experimento de rótulos fracos que já usa `data/processed/gmdcsa24`.

## CAUCAFall v4

O CAUCAFall v4 é um fallback documentado caso o UP-Fall fique indisponível. Ele não foi
implementado. Adotá-lo desloca o UP-Fall como dataset primário e exige um novo adaptador e uma
nova partição por sujeito.

## Backend comum (pose138)

Os oito métodos usam `pose138-valid-intervals-v1`, que soma ao pose134 a presença e as validades
de velocidade e aceleração, mais a idade da observação. Assim, learned × physics compara apenas
o gate. O pose134 histórico permanece nas cinco configurações de referência, que não são
retreinadas. Para B1 physics com DINOv3, `r_visual = 0` por construção. O gate físico reduz-se
então ao termo de qualidade mais o resíduo de pose, e isso deve constar no artigo.

## TCNTE adaptado ao protocolo do estudo

| Aspecto | Upstream (`23b7d32`) | Adaptação |
| --- | --- | --- |
| Entrada | 18 quadros × 12 articulações (x, y), min–max por quadro | 24 quadros a 10 fps (janela do estudo); mesmas 12 articulações COCO 5–16; min–max por quadro calculado sobre o bloco `kp_xy` do pose138, que é afim isotrópico por quadro dos pixels e dá resultado equivalente. |
| Ausência de pessoa | não tratada | quadro zerado (presença = canal 134). |
| Causalidade | convolução não causal | TCN com preenchimento só à esquerda; janela termina em t. |
| TCN | [8, 16, 32], kernel 3, dilatações 1/2/4, weight norm, dropout 0,2 | idêntico. |
| Encoder | posição aprendida, 1 camada, d=32, 4 cabeças, ff=32, dropout 0,1, média | idêntico. |
| Saída | 2 classes, softmax antes da focal (softmax duplo) | 10 classes, logits; focal sobre log-softmax uma vez. |
| Perda | focal γ=3, pesos [1, 30] | focal γ=3, pesos de classe do treino do estudo normalizados. |
| Otimização | Adam 1e-4, 100 épocas | Adam 1e-4, sem agendador; 30 épocas e batch 64, como todos os métodos. |
| Seleção | melhor acurácia de **teste** nas últimas 10 épocas | `best_val` (macro-F1 restrito de 8 classes na validação), como todos. |
| Dependências | módulo `common` ausente, caminhos Windows fixos | nenhuma; mesma CLI dos demais métodos. |

O resultado é reportado como **“TCNTE adaptado ao protocolo do estudo”**, nunca como reprodução
dos escores binários publicados.

## Runs externos de movimento

O contexto do projeto relata 150 pares treino/avaliação externos. Nenhum artefato desses runs
existe nos forks nem neste checkout. `reuse-audit` confere, para cada run:

- schema e receita;
- braço, fusão e semente;
- épocas, batch e política de seleção;
- `manifest_sha256` do pacote local;
- `targets_sha256`;
- hashes de checkpoint.

Runs incompatíveis permanecem como evidência histórica, rotulada à parte, e nunca são
agregados com os do estudo.

## Estatística

- A unidade de reamostragem é o cluster: sujeito (UP-Fall, GMDCSA-24) ou gravação (Le2i).
- O mesmo sorteio vale para tratamento e referência e para todas as sementes. O IC percentil é
  calculado sobre a média entre sementes.
- A semente do bootstrap é 20260, com 2000 reamostras.
- As diferenças por semente mostram média, desvio e contagem de sinais.
- Nenhum teste de significância é declarado automaticamente.
