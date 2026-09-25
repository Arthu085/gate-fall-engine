# Qualidade SAM 3 (`q_sam3`)

`gatefall.sam3.quality` define `q_sam3`, a proxy operacional de qualidade por
quadro da fonte SAM 3 do braço C, e valida essa proxy contra degradações
controladas. O índice é causal, determinístico, finito e limitado a `[0, 1]`,
mas **não é uma confiança nem uma probabilidade calibrada**. Esta etapa cobre
a parte SAM 3 do PEND-013; ela não implementa C1 nem gate, não retreina o C0 e
não altera `V_t`, a seleção de instância nem o contrato de armazenamento
descritos na [fundação SAM 3](sam3-foundation.md).

## Definição

Para cada quadro `t` do `.h5` do SAM 3:

\[
q_{sam3}(t) = \operatorname{clip}(\mathit{sam\_score}_t, 0, 1) \cdot \mathit{present}_t,
\]

em que `sam_score` é o dataset gravado fora de `V_t` exatamente para este uso
— o score do SAM 3 da instância escolhida pela seleção contínua — e `present`
é o canal 0 de `V_t`. O produto zera o índice quando não há instância
selecionada ou quando a máscara selecionada é degenerada: nesse último caso a
extração grava `V_t` zerado, mas mantém o score da instância em `sam_score`.
Valores fora de `[0, 1]` além de uma tolerância de `1e-6`, não finitos ou um
`present` não binário fazem o cálculo falhar.

Propriedades:

- **Nativa do SAM 3.** Usa só a saída do próprio detector; não reutiliza a
  fórmula de luminância, contraste e nitidez do
  [`q_visual` do DINOv3](dinov3-quality.md) nem lê pixels.
- **Causal.** Depende só do quadro atual; a seleção que escolhe a instância já
  é causal (histórico de bboxes passadas, sem olhar o futuro).
- **Determinística.** Função pura dos artefatos gravados.
- **Faixa efetiva.** O processador filtra instâncias com score abaixo de
  `0,5`, então os valores observados ficam em `{0} ∪ [0,5; 1]`. A proxy não
  reescala essa faixa: uma transformação afim não mudaria ordenação nem
  correlações, e fica para a política de gate decidir limiares.
- **`n_instances` fica de fora.** Várias pessoas no quadro indicam ambiguidade
  de cena, não degradação visual; o número de instâncias entra só como
  diagnóstico.

## Validação controlada

O comando `validate` usa somente `train` e `val`; `test` nunca participa da
seleção, da distribuição limpa nem do sweep, e os `.h5` de `test` nem são
abertos. Ele:

1. confere a proveniência homogênea dos `.h5` de `train`/`val`
   (`collect_sam3_provenance`);
2. calcula a distribuição limpa de `q_sam3` sobre todos os quadros de
   `train`/`val`, a partir dos `.h5` já gravados, por split e rótulo e por
   faixa de `n_instances` (`0`, `1`, `2+`), com taxas de `present` e de
   quadros com várias instâncias;
3. seleciona os mesmos eventos da validação DINOv3 (`select_validation_samples`:
   até oito eventos por split e rótulo, quadro central de cada sequência
   contígua de rótulo, ordem por SHA-256, `IGNORE` excluído), de modo que o
   hash da seleção é comparável entre as duas fontes;
4. sobe o runtime isolado oficial (`Sam3RuntimeSegmenter`) e falha se
   `sam3_checkpoint_sha256`, `sam3_source_revision` ou
   `sam3_inference_autocast_dtype` do worker divergirem dos artefatos gravados;
5. reinfere o SAM 3 sobre o quadro limpo e sobre cada degradação.

As degradações são aplicadas ao quadro RGB `uint8` na resolução nativa do
vídeo, que é o que o SAM 3 recebe — não ao recorte `224 × 224` do DINOv3. As
grades de severidade repetem as do DINOv3, mas os raios do blur são em pixels
nativos:

| Degradação | Severidades |
| --- | --- |
| Box blur | raios `0, 2, 3, 6, 12` px nativos |
| Subexposição | ganhos `1, 0.75, 0.5, 0.25, 0.125` |
| Sobre-exposição | ganhos `1, 0.75, 0.5, 0.25, 0.125` em relação ao branco |
| Contraste | ganhos `1, 0.75, 0.5, 0.25, 0.125` em torno da mediana da luminância |

A severidade `0` é idêntica ao quadro limpo e reaproveita a inferência limpa.
Cada quadro degradado passa pela mesma política de seleção da extração, com o
quadro limpo como histórico: um `InstanceSelector` novo recebe primeiro as
instâncias limpas e depois as degradadas, então a instância degradada é
escolhida por continuidade de IoU com a instância limpa, não pelo maior score.
A referência de fidelidade é o IoU entre a máscara limpa e a máscara
degradada selecionadas (ausência conta como máscara vazia; duas ausências
concordam com IoU `1`).

Execução de referência:

```bash
uv run python -m gatefall.sam3.quality validate \
  --dataset le2i \
  --events-per-class 8 \
  --checkpoint data/scratch/weights/sam3/sam3.pt
```

O relatório JSON enviado para stdout (o progresso por evento vai para stderr)
inclui:

- proveniência dos artefatos e manifesto do runtime;
- hash e contagens da seleção;
- distribuições limpas por split e rótulo e por faixa de `n_instances`;
- concordância entre o `q_sam3` reinferido no quadro limpo e o gravado. A
  reinferência parte de um seletor sem histórico, então diverge do valor
  gravado quando a extração causal escolheu outra instância em cenas com
  várias pessoas;
- distribuições de `q_sam3`, taxa de `present` e IoU por severidade;
- por degradação: fração de passos adjacentes não crescentes, fração de
  eventos cujo `q_sam3` cai entre o limpo e a severidade máxima, Spearman
  agregado severidade × `q_sam3`, Pearson e Spearman agregados
  `q_sam3` × IoU e as medianas/quartis das correlações intraevento. Eventos
  com `q_sam3` constante nas cinco severidades não definem correlação e são
  contados em `constant_q_events`, em vez de entrarem como zero.

### Execução piloto no Le2i

Uma execução piloto com `--events-per-class 1` rodou numa GTX 1650
(`sam3_inference_autocast_dtype=float16`, igual aos artefatos gravados, com
checkpoint e revisão upstream coincidentes). Ela selecionou 17 eventos, um
por classe de `train` e de `val`, com hash
`f9b71e820aedd4bb22725fac2e677dda6f2f9e0148093302c1864321c119b01c`; `test`
não participou. A execução levou cerca de 70 minutos (quase 290 inferências,
cerca de 15 s por quadro nesse hardware).

**Distribuição limpa (todos os quadros de `train`/`val`).** Com detecção, a
mediana de `q_sam3` por classe fica entre `0,935` e `0,970`. A maior parte da
variação vem de `present`: 1.368 quadros de `train` com `n_instances = 0` (em
grande parte `IGNORE`, cuja taxa de `present` é `0,10`) valem `0`. Quadros
com várias instâncias não recebem índice menor (mediana `0,968` contra `0,964`
com uma instância em `train`), o que confirma que `n_instances` não mede
degradação. O `q_sam3` reinferido no quadro limpo coincidiu com o gravado em
16 de 17 eventos (`|Δ| ≤ 1e-3`, diferença máxima `0,0015`).

| Degradação | Não crescente | Queda limpo → máx. | Spearman intraevento severidade × q, mediana [Q25, Q75] | Spearman intraevento q × IoU, mediana [Q25, Q75] |
| --- | ---: | ---: | ---: | ---: |
| Blur | 0,941 | 17/17 | −0,97 [−1,00, −0,90] | 1,00 [1,00, 1,00] |
| Contraste | 0,838 | 15/17 | −0,90 [−0,97, −0,82] | 0,89 [0,82, 0,97] |
| Sobre-exposição | 0,853 | 16/17 | −0,90 [−1,00, −0,90] | 0,90 [0,90, 1,00] |
| Subexposição | 0,853 | 16/17 | −1,00 [−1,00, −0,90] | 1,00 [0,90, 1,00] |

Nenhum evento teve `q_sam3` constante. Medianas agregadas nos extremos:

| Condição | `q_sam3` mediana | `present` | IoU mediana |
| --- | ---: | ---: | ---: |
| Limpa | 0,960 | 1,00 | 1,000 |
| Blur, raio 12 | 0,000 | 0,12 | 0,000 |
| Contraste, ganho 0,125 | 0,942 | 1,00 | 0,968 |
| Sobre-exposição, ganho 0,125 | 0,937 | 1,00 | 0,947 |
| Subexposição, ganho 0,125 | 0,938 | 1,00 | 0,969 |

Leitura: dentro de cada evento, `q_sam3` cai com a severidade e acompanha a
fidelidade da máscara nas quatro degradações, o que apoia congelar a
definição. A resposta é, porém, assimétrica: o blur derruba a detecção
(`present` vai a `0,12` no raio 12), enquanto as degradações fotométricas
movem a mediana só de `0,960` para cerca de `0,94`, porque o SAM 3 continua
segmentando bem a pessoa. Por isso a correlação agregada entre eventos é
fraca nessas três (Pearson `q × IoU` de `0,12` a `0,64`) e forte no blur
(`0,99`). O piloto tem um evento por classe; a execução de referência com
oito eventos por classe é a evidência que decide o congelamento.

## Limitações

- O score do SAM 3 mede a confiança do detector em "person", não a qualidade
  da imagem: uma pessoa bem visível em um quadro degradado pode manter score
  alto, e uma pessoa parcialmente ocluída em um quadro limpo pode ter score
  baixo.
- A faixa `(0; 0,5)` nunca ocorre por causa do limiar do processador; a
  proxy satura em `0` quando a detecção some.
- Os resultados são evidência diagnóstica no Le2i, não um limiar de gating
  nem garantia de generalização.

## Selftest

```bash
uv run python -m gatefall.sam3.quality selftest
```

O selftest usa um segmentador falso (score proporcional ao contraste) e `.h5`
sintéticos em diretório temporário. Ele verifica fórmula, limites, rejeições,
causalidade por quadro, determinismo, identidade e monotonicidade das
degradações, box blur contra uma referência ingênua, IoU de máscaras,
continuidade da seleção no sweep, reaproveitamento da inferência limpa,
agregações com eventos constantes, leitura restrita a `train`/`val` e a
restrição da CLI ao Le2i `cs`. Não acessa vídeos, dataset real, checkpoint,
runtime isolado nem GPU e, portanto, não substitui a validação real.
