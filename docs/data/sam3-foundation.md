# Fundação SAM 3 (braço C)

`src/gatefall/sam3/` implementa a extração offline do descritor `V_t` do
braço C a partir do backbone congelado **SAM 3** (`facebook/sam3`). Esta
etapa cobre apenas a fundação de dados: descritores de máscara, seleção de
instância, armazenamento e proveniência. Ela **não implementa** C0, C1, um
`q_visual` específico do SAM, fusão, gate, cross-attention, treino da TCN do
braço C nem avaliação por eventos — e **não fecha o PEND-015**.

## Aviso de honestidade: o que os selftests provam e o que não provam

`uv run python -m gatefall.sam3.selection selftest` e
`uv run python -m gatefall.sam3.extract selftest` rodam apenas contra
fixtures sintéticas e um segmentador falso injetado (`Sam3Segmenter`
protocolo, sem `torch`/`transformers`). Eles validam a matemática dos
descritores, a política de seleção contínua, o armazenamento HDF5, a
verificação de proveniência e o alinhamento de quadro — nada disso exercita
o modelo `facebook/sam3` real. **A extração real ainda não foi rodada**:
não há evidência de que o SAM 3 produza máscaras válidas sobre vídeo real do
Le2i. Só `uv run python -m gatefall.sam3.extract verify-frame-alignment`
roda o hardware real, e mesmo assim valida apenas alinhamento de quadro, não
qualidade de máscara.

## Somente o protocolo cs

Todas as operações de `gatefall.sam3` aceitam apenas `--dataset le2i` (Le2i
sob o protocolo `cs`). Um adapter `le2i-cv` é rejeitado por
`ensure_sam3_dataset_supported` em todo ponto de entrada, pelo mesmo motivo
do braço DINOv3 (ver [Features DINOv3](dinov3-features.md#somente-o-protocolo-cs)):
`sam3_root` não é derivado do protocolo, então uma extração sob `cv`
sobrescreveria os `.h5` do `cs`.

## Modelo e prompt

- Modelo congelado: `facebook/sam3`, carregado no sub-projeto isolado
  `sam3_runtime/` (ver abaixo).
- Inferência quadro a quadro (sem tracking de vídeo, sem SAM 3.1).
- Prompt de texto fixo e único: `"person"`. Nenhuma variação de prompt
  relacionada a queda e nenhuma bbox de pose é usada como prompt — a
  seleção de instância (abaixo) é independente da pose.

## Descritor `V_t`

Cada quadro produz um vetor `V_t ∈ R^10`, calculado só a partir da máscara
binária da instância selecionada (`gatefall.sam3.descriptors`), com ordem de
canal congelada:

```text
present, mask_area_norm, centroid_x_norm, centroid_y_norm,
bbox_w_norm, bbox_h_norm, fill_ratio, eccentricity, sin_2theta, cos_2theta
```

Convenção de eixos: `x` é coluna, `y` é linha, origem no canto superior
esquerdo — a mesma convenção usada pelo restante do pipeline.

A bbox é a extensão inclusiva de pixel da própria máscara (`bbox_from_mask`),
e `fill_ratio` usa essa mesma bbox (`m00 / (bbox_w * bbox_h)`), não uma caixa
externa (por exemplo, de detecção).

Normalização: quantidades em `x` (centróide, largura da bbox) são divididas
pela largura do quadro; quantidades em `y` (centróide, altura da bbox), pela
altura do quadro. `mask_area_norm` é a contagem de pixels da máscara dividida
pela área total do quadro (`largura × altura`).

### Momentos de imagem, excentricidade e orientação

`mu20`, `mu02` e `mu11` são os momentos centrais de segunda ordem da máscara
em torno do seu centróide `(xbar, ybar)`. A excentricidade segue a convenção
do `skimage.measure.regionprops` — excentricidade da elipse com os mesmos
momentos de segunda ordem da máscara — calculada a partir dos autovalores

```text
lambda1 = ((mu20 + mu02) + r) / 2
lambda2 = ((mu20 + mu02) - r) / 2
r = sqrt((mu20 - mu02)^2 + (2*mu11)^2)
eccentricity = sqrt(1 - lambda2/lambda1)   # 0 quando lambda1 <= 0
```

O ângulo do eixo principal é `theta = 0.5*atan2(2*mu11, mu20-mu02)`, mas o
descritor não guarda `theta` diretamente: ele é codificado como
`(sin_2theta, cos_2theta) = ((2*mu11)/r, (mu20-mu02)/r)`. Isso existe porque
o eixo principal de uma máscara não tem sentido de seta — `theta` e
`theta + pi` descrevem o mesmo eixo — então `theta` bruto é descontínuo
nessa fronteira; codificar o ângulo dobrado remove a ambiguidade
`pi`-periódica. Quando `r == 0` (máscara isotrópica), ambos os componentes
são zerados.

### Detecção ausente: zero exato, sem imputação

Quando não há instância selecionada no quadro, ou a máscara selecionada é
degenerada (nenhum pixel de primeiro plano), `compute_descriptor` devolve o
descritor totalmente zerado, **incluindo `present=0`** — sem preenchimento
por continuidade (forward-fill) e sem interpolação. Isso difere
deliberadamente de `pose/loading.py`, cujo forward-fill existe para evitar
picos nas features cinemáticas derivadas (velocidade, aceleração); `V_t` não
tem derivadas temporais, então não há esse risco a mitigar.

## Seleção de instância

`gatefall.sam3.selection.InstanceSelector` é causal, um objeto por vídeo,
sem reset, e independente de pose (o módulo não importa `gatefall.pose`).
A cada quadro:

1. Sem histórico (`_last_bbox is None`) ou sem sobreposição com a bbox
   anterior, a instância de maior score do SAM é escolhida (aquisição).
2. Com histórico, a continuidade prioriza a bbox de maior IoU contra a
   última bbox selecionada; o score do SAM só desempata IoUs iguais. O score
   nunca decide sozinho enquanto houver continuidade geométrica possível.
3. Uma lacuna de detecção (quadro sem nenhuma instância) não apaga
   `_last_bbox` — a continuidade retoma normalmente quando a detecção
   volta em um quadro futuro.

O score do SAM entra apenas em aquisição e desempate; a política nunca
ordena candidatos só por score quando há sobreposição geométrica.

## Armazenamento HDF5

Um arquivo por vídeo, agrupado por vídeo (nunca um arquivo por quadro), em
`data/features/le2i/sam3/<env>/<video_name>.h5` (`adapter.sam3_root`), com:

- dataset `v_t`: shape `[K, 10]`, `float32`;
- dataset `sam_score`: shape `[K]`, `float32`;
- dataset `n_instances`: shape `[K]`, `int16`;
- atributos de proveniência: `model_name`, `text_prompt`,
  `sam3_checkpoint_sha256`, `sam3_runtime_lock_sha256`, `target_fps`, além de
  `video_id`, `env`, `split`, `subject`, `K`, `fps`, `width`, `height` e os
  campos do manifesto de runtime reportados pelo worker.

O score do SAM é gravado **fora** de `v_t`, em `sam_score`, por design: o
contrato de `V_t` é congelado em 10 dimensões, e um canal de confiança do
detector não pode vazar para dentro dele.

A gravação é atômica (`.tmp` + `os.replace`) e, após o `replace`, o arquivo é
relido do disco e comparado byte a byte com os arrays e atributos em memória
antes de a gravação ser considerada bem-sucedida (`verify_written_file`).

## Isolamento de ambiente: `sam3_runtime/`

O SAM 3 roda em `sam3_runtime/`, um sub-projeto `uv` isolado com seu próprio
`pyproject.toml`. `gatefall.sam3.runtime` nunca importa `torch`/`transformers`
do SAM 3 diretamente nem o script `sam3_runtime/run_sam3.py`: a comunicação é
só por um subprocesso de vida longa, um quadro por vez, via protocolo de fio
com prefixo de tamanho (todo inteiro é `uint32` big-endian):

- ao iniciar, o worker imprime uma linha JSON com o manifesto de runtime
  (versões, hash do checkpoint) antes de processar qualquer quadro;
- por quadro, o cliente escreve um cabeçalho JSON com prefixo de tamanho
  (`height`, `width`, `text_prompt`) seguido do RGB cru do quadro, também
  com prefixo de tamanho; o worker responde com um `.npz` com prefixo de
  tamanho contendo `masks` (`[N,H,W]` bool) e `scores` (`[N]` float32).

O `pyproject.toml`/`uv.lock` da raiz do repositório permanecem intocados —
o ambiente A/B (braços A e B) não pode mudar por causa do braço C. O
trade-off dessa fronteira é um segundo ambiente `uv` para sincronizar e a
sobrecarga de IPC do protocolo de fio; esse custo só é pago no caminho de
extração real (`extract`/`extract-all`/`verify-frame-alignment`), nunca na
CI, que só roda os `selftest` sintéticos contra um segmentador falso.

### Setup único e isolado

```bash
cd sam3_runtime && uv sync
```

`sam3_runtime/uv.lock` é deliberadamente **não commitado**: o operador roda
`uv sync` no sub-projeto antes de extrair. Consequência: enquanto esse
arquivo não existir, `sam3_runtime_lock_sha256` (e, de forma análoga,
`sam3_checkpoint_sha256` enquanto o checkpoint não existir) é gravado como
string vazia na proveniência, e o código emite um aviso em stderr sempre que
um hash de proveniência resolve vazio.

## Limitações residuais conhecidas

1. **`extract` (vídeo único) pode reportar "já existe e é válido" com base
   em um lock desatualizado.** O caminho de `extract` para um único vídeo lê
   `sam3_runtime/uv.lock` para checar o `.h5` existente **antes** de o
   subprocesso do runtime sincronizar o sub-projeto isolado. Se
   `sam3_runtime/pyproject.toml` mudar e `uv sync` ainda não tiver sido
   rodado, o hash do lock desatualizado pode fazer o comando pular uma
   reextração necessária. `extract-all` não tem esse problema: ele resolve
   os hashes depois de entrar no contexto do runtime e os repassa para cada
   vídeo. Contorno: rode `uv sync` em `sam3_runtime/` antes de extrair, ou
   passe `--force`.
2. **`extract-all` aborta o lote inteiro se o subprocesso do runtime morrer
   no meio.** `run_sam3_extract_all` interrompe com uma mensagem clara em
   vez de continuar processando os vídeos restantes contra um subprocesso
   morto (o que produziria falhas enganosas de pipe/EOF), então o resumo por
   vídeo não é impresso nesse caso. Isso difere deliberadamente de
   `dinov3 extract-all`, que não tem subprocesso capaz de morrer no meio do
   lote.

## Como executar

```bash
uv run python -m gatefall.sam3.selection selftest
```

Testa a política de seleção contínua (aquisição, continuidade por IoU,
desempate por score, lacuna de detecção) contra casos sintéticos.

```bash
uv run python -m gatefall.sam3.extract selftest
```

Roda checagens sintéticas dos descritores, do armazenamento, da guarda de
protocolo `cs` e das fixtures de alinhamento de quadro, tudo contra um
segmentador falso — não toca no checkpoint do SAM 3, no sub-projeto
`sam3_runtime/` nem no dataset real.

```bash
uv run python -m gatefall.sam3.extract extract --video-id <ENV/VIDEO> [--runtime-dir DIR] [--checkpoint PATH] [--force] [--dataset le2i]
```

Extrai `V_t` de um único vídeo e grava o `.h5`. Requer `sam3_runtime/`
sincronizado (`uv sync`) e o checkpoint do SAM 3 disponível. Um `.h5`
existente e válido é pulado, a menos que `--force` seja passado (ver
limitação 1 acima sobre o lock).

```bash
uv run python -m gatefall.sam3.extract extract-all [--runtime-dir DIR] [--checkpoint PATH] [--force] [--dataset le2i]
```

Extrai `V_t` de todos os vídeos listados em `frames.parquet`, reutilizando o
mesmo subprocesso do runtime. Ao final, imprime quantos vídeos foram
processados, pulados e com falha — exceto quando o runtime morre no meio do
lote (ver limitação 2 acima), caso em que o comando aborta antes do resumo.

```bash
uv run python -m gatefall.sam3.extract report [--dataset le2i]
```

Valida a cobertura dos `.h5` de SAM 3 já extraídos contra `frames.parquet` e
a homogeneidade de proveniência entre eles.

```bash
uv run python -m gatefall.sam3.extract verify-frame-alignment [--runtime-dir DIR] [--checkpoint PATH] [--dataset le2i]
```

Confere que `decode_frames` devolve o quadro correto para o `src_index`
pedido, recomputando o vídeo inteiro (não só posições isoladas) para uma
amostra fixa de vídeos — a seleção de instância tem estado causal, então
checar uma posição sem o histórico de seleção anterior não validaria a
mesma decisão tomada pela extração. Roda o SAM 3 real via
`Sam3RuntimeSegmenter`; não faz parte do `selftest` nem da CI.

## Licença do SAM 3

O código e o checkpoint do SAM 3 utilizados aqui têm licença própria, definida
pela Meta/`facebook/sam3` — **não verificada nem afirmada por esta
documentação**. Quem for rodar a extração real deve consultar os termos
oficiais diretamente na fonte do modelo antes de baixar, redistribuir ou
publicar resultados obtidos com ele; a licença MIT deste repositório não se
estende a esses materiais.
