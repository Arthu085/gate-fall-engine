# Fundação SAM 3 (braço C)

`src/gatefall/sam3/` implementa a extração offline do descritor `V_t` do
braço C a partir do backbone congelado **SAM 3** (runtime oficial
`facebookresearch/sam3`). Esta
etapa cobre apenas a fundação de dados: descritores de máscara, seleção de
instância, armazenamento e proveniência. Ela **não implementa** C0, C1, um
`q_visual` específico do SAM, fusão, gate, cross-attention, treino da TCN do
braço C nem avaliação por eventos — e **não fecha o PEND-015**.

## O que a CI prova e o que foi validado manualmente

### CI: somente selftests sintéticos

`uv run python -m gatefall.sam3.selection selftest` e
`uv run python -m gatefall.sam3.extract selftest` rodam apenas contra
fixtures sintéticas e um segmentador falso injetado (`Sam3Segmenter`
protocolo, sem `torch`/`sam3`). Eles validam a matemática dos
descritores, a política de seleção contínua, o armazenamento HDF5, a
verificação de proveniência e o alinhamento de quadro — nada disso exercita
o modelo SAM 3 real, e a CI nunca sobe o runtime isolado. Os três comandos
que sobem o worker real do SAM 3 via `Sam3RuntimeSegmenter` são `extract`,
`extract-all` e `verify-frame-alignment`; os comandos `selftest` e
`sam3 report` são livres de hardware (`report` apenas abre os `.h5` já
gravados e nunca constrói o modelo nem sobe o runtime isolado).

### Validação manual em hardware real

A extração real foi validada manualmente, fora da CI, no head `faf4cf2`:

- **Vídeo único em duas GPUs.** Em uma GTX 1650 local e em uma Tesla T4 do
  Kaggle, `extract` sobre `coffee_room_01/video_1` concluiu a construção do
  modelo, o carregamento do checkpoint e a inferência real, com `K=62`
  quadros e `n_present=60`, sob autocast CUDA FP16
  (`sam3_inference_autocast_dtype=float16`) e proveniência HDF5 válida.
- **Lote completo na T4.** `extract-all` gravou os 190 vídeos / 30.494
  quadros do Le2i; 28.591 quadros (`93,76%`) têm `present==1`.
- **Gate de integridade.** `sam3 report` passou em todas as checagens de
  cobertura, estrutura, split e proveniência.
- **Alinhamento de quadro.** `verify-frame-alignment` terminou com código de
  retorno 0 na amostra fixa (`coffee_room_01/video_1` e `home_01/video_1`),
  sem falhas; um empate exato entre vizinhos foi reportado como
  inconclusivo, por design.

Essa validação cobre a fundação offline e nada além dela: construção do
modelo, recuo para FP16, extração do Le2i, integridade dos artefatos e
alinhamento de quadro. Ela não audita a qualidade semântica de cada máscara
(`verify-frame-alignment` valida apenas alinhamento de quadro), não exercita
os ramos BF16 e de CPU e não implementa nem valida C0, C1, `q_visual`
específico do SAM, treino da fusão, cross-attention ou avaliação de alarme.

### Histórico: falhas anteriores à validação final

As tentativas abaixo precederam a validação acima e ficam registradas porque
motivaram as correções atuais do runtime isolado. Nenhuma delas descreve o
estado atual.

1. A primeira tentativa real de smoke test do worker falhou na
   inicialização, antes de qualquer inferência, com
   `ModuleNotFoundError: No module named 'pkg_resources'` (ver
   [teto de `setuptools`](#teto-de-setuptools-no-sam3_runtime) abaixo).
2. A segunda, já com o teto de `setuptools` em vigor (`setuptools==81.0.0`,
   `import pkg_resources` funcionando), falhou de novo antes da construção do
   modelo com `ModuleNotFoundError: No module named 'einops'` (ver
   [dependências adicionadas por lacuna de empacotamento upstream](#dependencias-adicionadas-por-lacuna-de-empacotamento-upstream)
   abaixo).
3. A terceira esbarrou primeiro em um OOM de memória do host WSL — uma
   limitação de ambiente, não um defeito do repositório — superado após
   aumentar a alocação de memória do WSL. A rodada corrigida atravessou a
   construção do modelo, o carregamento do checkpoint e o manifesto de
   inicialização do worker, e falhou no primeiro quadro real do Le2i dentro
   de `Sam3Processor.set_image()` com `RuntimeError: mat1 and mat2 must have
   the same dtype, but got BFloat16 and Float` (ver
   [política de precisão de inferência (autocast)](#politica-de-precisao-de-inferencia-autocast)
   abaixo). O `EOFError` observado do lado do processo pai foi apenas
   consequência da saída do worker, não uma falha de protocolo.

## Somente o protocolo cs

Todas as operações de `gatefall.sam3` aceitam apenas `--dataset le2i` (Le2i
sob o protocolo `cs`). Um adapter `le2i-cv` é rejeitado por
`ensure_sam3_dataset_supported` em todo ponto de entrada, pelo mesmo motivo
do braço DINOv3 (ver [Features DINOv3](dinov3-features.md#somente-o-protocolo-cs)):
`sam3_root` não é derivado do protocolo, então uma extração sob `cv`
sobrescreveria os `.h5` do `cs`.

## Modelo e prompt

- Modelo congelado: **SAM 3 base** (não SAM 3.1), carregado no sub-projeto
  isolado `sam3_runtime/` (ver abaixo) via `build_sam3_image_model(...)` do
  pacote oficial `facebookresearch/sam3`, com `enable_inst_interactivity=False`.
- Inferência quadro a quadro (sem tracking de vídeo).
- Prompt de texto fixo e único: `"person"`, aplicado via
  `Sam3Processor.set_image(...)` seguido de `set_text_prompt("person", state)`.
  Nenhuma variação de prompt relacionada a queda e nenhuma bbox de pose é
  usada como prompt — a seleção de instância (abaixo) é independente da pose.
- `Sam3Processor(confidence_threshold=0.5)` já filtra instâncias abaixo do
  limiar antes de as máscaras chegarem ao lado GateFall do protocolo de fio.
- O quadro RGB é convertido para `PIL.Image` antes de `set_image` porque a
  API upstream interpreta mal a forma de um `ndarray` HWC passado direto.
- Um checkpoint **SAM 3.1** não é validado quanto a compatibilidade por este
  repositório — o operador deve fornecer um checkpoint SAM 3 base.

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
  `sam3_checkpoint_sha256`, `sam3_runtime_lock_sha256`, `sam3_source_revision`,
  `sam3_inference_autocast_dtype`, `target_fps`, além de `video_id`, `env`,
  `split`, `subject`, `K`, `fps`,
  `width`, `height` e os campos do manifesto de runtime reportados pelo
  worker. `sam3_source_revision` é o commit git upstream resolvido (ver
  abaixo) a partir do `direct_url.json` (PEP 610) da distribuição `sam3`
  instalada no sub-projeto isolado. `sam3_inference_autocast_dtype` registra
  o dtype de autocast (`bfloat16` ou `float16`) sob o qual as máscaras do
  vídeo foram de fato inferidas (ver
  [política de precisão de inferência (autocast)](#politica-de-precisao-de-inferencia-autocast));
  misturar artefatos FP16 e BF16 em um mesmo conjunto é uma falha de
  proveniência, porque os descritores passariam a vir de duas precisões
  numéricas diferentes sem que nada no `.h5` distinguisse as duas metades.

O score do SAM é gravado **fora** de `v_t`, em `sam_score`, por design: o
contrato de `V_t` é congelado em 10 dimensões, e um canal de confiança do
detector não pode vazar para dentro dele.

A gravação é atômica (`.tmp` + `os.replace`) e, após o `replace`, o arquivo é
relido do disco e comparado byte a byte com os arrays e atributos em memória
antes de a gravação ser considerada bem-sucedida (`verify_written_file`).

## Isolamento de ambiente: `sam3_runtime/`

O SAM 3 roda em `sam3_runtime/`, um sub-projeto `uv` isolado com seu próprio
`pyproject.toml` e `sam3_runtime/uv.lock` **commitado** no repositório.
`gatefall.sam3.runtime` nunca importa `torch`/`sam3` diretamente nem o script
`sam3_runtime/run_sam3.py`: a comunicação é só por um subprocesso de vida
longa, um quadro por vez, via protocolo de fio com prefixo de tamanho (todo
inteiro é `uint32` big-endian):

- ao iniciar, o worker imprime uma linha JSON com o manifesto de runtime
  (versões, `sam3_source_revision`, hash do checkpoint e
  `sam3_inference_autocast_dtype`, o dtype de autocast resolvido para o
  dispositivo daquele processo) antes de processar qualquer quadro;
- por quadro, o cliente escreve um cabeçalho JSON com prefixo de tamanho
  (`height`, `width`, `text_prompt`) seguido do RGB cru do quadro, também
  com prefixo de tamanho; o worker responde com um `.npz` com prefixo de
  tamanho contendo `masks` (`[N,H,W]` bool) e `scores` (`[N]` float32). O
  modelo devolve máscaras `[N,1,H,W]`; o worker as reduz (`squeeze`) para o
  `[N,H,W]` do protocolo de fio antes de enviar.

`sam3_runtime/pyproject.toml` fixa
`sam3 @ git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40`
(commit exato, não uma tag — o repositório upstream `facebookresearch/sam3`
não publica tags de release), `numpy>=1.26,<2`, porque o pacote oficial
exige NumPy abaixo de 2, e três dependências adicionais —
`einops`, `pycocotools` e `psutil` — exigidas para simplesmente importar o
pacote `sam3`, não para treino ou tracking de vídeo (ver
[dependências adicionadas por lacuna de empacotamento upstream](#dependencias-adicionadas-por-lacuna-de-empacotamento-upstream)
abaixo). Isso diverge do `numpy>=2.5.2` da raiz do
repositório; a divergência é segura porque só bytes cruzam a fronteira do
subprocesso (o protocolo de fio acima) e o `pyright` da raiz só cobre
`src`/`scripts`, nunca `sam3_runtime/`.

O `pyproject.toml`/`uv.lock` da raiz do repositório permanecem intocados —
o ambiente A/B (braços A e B) não pode mudar por causa do braço C. O
trade-off dessa fronteira é um segundo ambiente `uv` para sincronizar e a
sobrecarga de IPC do protocolo de fio; esse custo só é pago no caminho de
extração real (`extract`/`extract-all`/`verify-frame-alignment`), nunca na
CI, que só roda os `selftest` sintéticos contra um segmentador falso.

### Teto de `setuptools` no `sam3_runtime/`

O commit `2345a4ad109ac29c569da749c91d84f10dc08c40` do SAM 3 ainda importa
`pkg_resources` em `sam3/model_builder.py` (usado para localizar o asset do
tokenizer BPE), e o `setuptools` removeu `pkg_resources` a partir da versão
`82.0.0`. `sam3_runtime/pyproject.toml` declara, sob `[tool.uv]`:

```toml
constraint-dependencies = ["setuptools>=77.0.3,<82"]
```

Isso é uma entrada de `constraint-dependencies`, não uma dependência normal,
porque o código do runtime nunca importa `setuptools` diretamente — ele só
chega de forma transitiva via `torch`, que declara `setuptools>=77.0.3` sem
teto superior. O piso `>=77.0.3` espelha o próprio piso do `torch`, então o
teto nunca aperta o que o `torch` já exige, só impede a resolução de subir
para uma versão sem `pkg_resources`. Com esse teto, `sam3_runtime/uv.lock`
foi regenerado e agora resolve `setuptools==81.0.0` (antes, `84.0.0`); só a
entrada de `setuptools` e um novo bloco `[manifest] constraints` mudaram no
lock, nenhuma outra dependência foi afetada.

A evidência que motivou esse teto foi um smoke test real do SAM 3 que
falhou na inicialização do worker com
`ModuleNotFoundError: No module named 'pkg_resources'`, antes de qualquer
inferência — quando o runtime isolado havia instalado `setuptools==84.0.0`.
Isso prova apenas que o piso está fechado; a extração real só foi validada
depois (ver [validação manual em hardware real](#validacao-manual-em-hardware-real)).

Uma checagem sintética em `extract selftest`
(`_check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal` e
seus auxiliares) lê `sam3_runtime/pyproject.toml` e `sam3_runtime/uv.lock`
como dado inerte, só com `tomllib` da biblioteca padrão — sem tocar em
modelo, runtime, checkpoint ou GPU — e falha se qualquer uma destas duas
condições não se sustentar isoladamente: (1) uma cláusula de
`constraint-dependencies` identifica a distribuição `setuptools` (por nome
normalizado conforme a PEP 503, então `setuptools-scm` e afins não contam)
com um teto superior estritamente abaixo de `82`; e (2) todo stanza
`setuptools` do lock commitado resolve abaixo de `82`. A checagem falha
fechada diante de arquivo ausente ou malformado. Ela garante apenas que os
dois arquivos declarados são consistentes entre si — não que o worker real
sobe nem que as máscaras produzidas são válidas.

### Dependências adicionadas por lacuna de empacotamento upstream

Após o teto de `setuptools` acima resolver o bloqueio de `pkg_resources`, um
segundo smoke test real do worker avançou até falhar com
`ModuleNotFoundError: No module named 'einops'`, ainda antes de qualquer
construção de modelo. `sam3_runtime/pyproject.toml` agora declara três
dependências adicionais para fechar essa lacuna:

- **`einops` (resolvido em `0.8.2`)** — `sam3/sam/rope.py:15` do commit
  fixado importa `from einops import rearrange, repeat`
  incondicionalmente, mas o `[project] dependencies` upstream não declara
  `einops`; ele só aparece em um extra de notebooks e no README do
  repositório upstream. É uma lacuna de empacotamento do upstream que este
  sub-projeto isolado precisa fechar para importar `sam3` de qualquer forma.
- **`pycocotools` (resolvido em `2.0.11`)** e
  **`psutil` (resolvido em `7.2.2`)** — exigidos pela mesma causa
  estrutural: `sam3/__init__.py` importa `model_builder`, que carrega
  módulos de treino e de predição de vídeo no momento da importação, antes
  de qualquer uso da API de imagem. As cadeias de importação verificadas
  são:
  ```text
  pycocotools: sam3/__init__.py:5 -> model_builder.py:40 ->
    sam1_task_predictor.py:16 -> sam3_tracker_base.py:14 ->
    train/data/collator.py:16 -> train/data/sam3_image_dataset.py:26 ->
    train/data/coco_json_loaders.py:10 (`from pycocotools import mask as mask_util`)
  psutil: sam3/__init__.py:5 -> model_builder.py:44 ->
    model/sam3_video_predictor.py:17 (`import psutil`)
  ```

Essas duas últimas dependências existem **só** por causa do grafo de
importação eager do upstream, não porque este projeto adota treino ou
tracking de vídeo. O pipeline continua congelado, offline, monocular RGB e
quadro a quadro, sem tracking de vídeo — nenhum desses invariantes muda com
esta atualização.

Com `sam3_runtime/uv.lock` regenerado a partir dessas três novas entradas de
dependência, um **preflight só de importação** — sem download de checkpoint,
sem construção de modelo e sem GPU — foi verificado com sucesso a partir do
lock committado:

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

Ambas as importações têm sucesso. Isso prova apenas que o grafo de módulos
Python resolve; **não é** uma inicialização de modelo bem-sucedida nem uma
extração real do Le2i. A construção do modelo e a extração real foram
validadas depois (ver
[validação manual em hardware real](#validacao-manual-em-hardware-real)).

### Política de precisão de inferência (autocast)

No commit fixado `2345a4ad109ac29c569da749c91d84f10dc08c40`, o `Mlp.forward()`
de `sam3/model/vitdet.py` chama `sam3.perflib.fused.addmm_act()`, que converte
entrada, peso e viés da **primeira** projeção para `torch.bfloat16`
incondicionalmente e devolve BF16 para um `fc2` que continua FP32 — daí o
`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and
Float`. Os exemplos oficiais do upstream rodam a inferência sob autocast, que
é exatamente a metade que faltava aqui.

A correção fica **só no worker isolado** (`sam3_runtime/run_sam3.py`):
`processor.set_image(...)` e `processor.set_text_prompt(...)` rodam dentro de
um `torch.autocast(device_type=..., dtype=...)` explícito. Nenhum tensor do
modelo ou do checkpoint é convertido de forma permanente (`.half()`,
`model.to(torch.bfloat16)`, `set_default_dtype`), e o `perflib` do upstream
não é corrigido, vendorizado nem monkey-patched — o backbone continua
congelado e o pacote `sam3` continua exatamente como publicado. As conversões
`.to(torch.bool)`/`.to(torch.float32)` das máscaras e scores ficam **fora** do
bloco de autocast, então o protocolo de fio permanece byte a byte idêntico.
`set_image`/`set_text_prompt` já são `@torch.inference_mode()` upstream, então
nenhum `no_grad`/`inference_mode` adicional é aplicado.

O dtype escolhido segue esta tabela:

| Dispositivo | Condição | `sam3_inference_autocast_dtype` |
| --- | --- | --- |
| CUDA | BF16 nativo | `bfloat16` |
| CUDA | sem BF16 nativo | `float16` |
| CPU | — | `bfloat16` |

O suporte a BF16 é consultado com
`torch.cuda.is_bf16_supported(including_emulation=False)`. O argumento é
obrigatório: no torch 2.14 o padrão `including_emulation=True` responde `True`
em `sm_75` (GTX 1650) apenas porque um tensor bfloat16 pode ser alocado, o que
anularia em silêncio o recuo para FP16 em hardware pré-Ampere.

O recuo FP16 não depende de um despacho FP16 do próprio
`_addmm_activation`: no runtime fixado (torch 2.14.0) esse op está
registrado no autocast de **CPU**, mas **não** está na lista de precisão
reduzida do autocast **CUDA**. Sob um autocast FP16 em CUDA, portanto, a
primeira projeção fundida do upstream pode continuar saindo em BF16. O que
reconcilia a divergência é o `linear`/`fc2` **seguinte**, esse sim elegível
ao autocast CUDA: sob o contexto FP16 ele leva a ativação BF16 e os
parâmetros FP32 a um dtype comum.

O ramo FP16 foi exercitado em hardware real: a GTX 1650 e a Tesla T4
(ambas `sm_75`, sem BF16 nativo) concluíram a inferência sob
`sam3_inference_autocast_dtype=float16` sem erro de dtype, incluindo o lote
completo na T4 (ver
[validação manual em hardware real](#validacao-manual-em-hardware-real)).
Isso comprova o comportamento de ponta a ponta, não uma instrumentação dos
dtypes intermediários. O ramo BF16 em CUDA continua **não exercitado em
hardware real**.

BF16 e FP16 **não** produzem artefatos intercambiáveis: a precisão reduzida
desloca os scores comparados com o `confidence_threshold = 0.5` de filtragem
de instâncias dentro do processador, logo `n_instances` e a própria
composição das máscaras podem mudar entre os dois ramos — por isso o dtype é
gravado como proveniência e por isso misturá-lo em um mesmo conjunto de
artefatos é falha de gate.

A política é **duplicada** nos dois lados — `select_inference_autocast_dtype_name`
em `gatefall.sam3.runtime` e `_select_inference_autocast_dtype_name` no worker
— pelo mesmo motivo do protocolo de fio: nenhum dos dois lados pode importar o
outro. O `extract selftest` só consegue cruzar as duas metades
**textualmente** (procura os fragmentos obrigatórios e proibidos no fonte do
worker); não há prova de equivalência semântica entre elas.

O ramo de CPU é uma política explícita, **não exercitada em hardware real**:
nenhuma extração do Le2i foi concluída em CPU até aqui.

### Setup único e isolado

```bash
uv sync --project sam3_runtime --locked
```

`sam3_runtime/uv.lock` é commitado: `uv sync --project sam3_runtime --locked`
materializa o ambiente exatamente a partir desse lock, falhando em vez de
regravá-lo caso ele esteja desatualizado. `ensure_sam3_runtime_available`
falha alto (`FileNotFoundError`) se `sam3_runtime/uv.lock` não existir no
diretório do runtime, e o worker falha na inicialização
(`Sam3SourceRevisionError`, saída não-zero) se não conseguir resolver
`sam3_source_revision` a partir do `direct_url.json` da distribuição `sam3`
instalada — nenhum dos dois caminhos reais grava mais um placeholder vazio
de proveniência. O sentinela de string vazia para
`sam3_checkpoint_sha256`/`sam3_runtime_lock_sha256`/`sam3_inference_autocast_dtype`
permanece **só** no
caminho sintético do `selftest`, que injeta um segmentador falso e nunca
sobe o runtime real — ver a limitação residual 1 abaixo para o caso real
restante (lock desatualizado, não lock ausente).

## Limitações residuais conhecidas

1. **`extract` (vídeo único) pode reportar "já existe e é válido" com base
   em um `uv.lock` desatualizado em relação a um `pyproject.toml` editado.**
   Commitar `sam3_runtime/uv.lock` resolve o caso de o arquivo estar
   totalmente ausente (agora `ensure_sam3_runtime_available` falha alto
   nesse caso). O que resta: o caminho de `extract` para um único vídeo
   ainda lê `sam3_runtime/uv.lock` do disco e faz seu hash para checar o
   `.h5` existente **antes** de o subprocesso do runtime sincronizar o
   sub-projeto isolado. Se `sam3_runtime/pyproject.toml` for editado
   localmente e `uv lock`/`uv sync` ainda não tiverem sido rodados, o hash é
   calculado sobre o lock desatualizado como está no disco, e isso pode
   fazer o comando pular uma reextração necessária. `extract-all` não tem
   esse problema: ele resolve os hashes depois de entrar no contexto do
   runtime e os repassa para cada vídeo. O mesmo vale para
   `sam3_inference_autocast_dtype`: contra um runtime vivo, `extract` de um
   vídeo único valida o `.h5` existente **antes** de o worker subir e
   reportar seu manifesto, então essa decisão de pular é cega ao dtype de
   autocast e não detecta um `.h5` antigo extraído em outra precisão.
   `extract-all` (que já conhece o dtype ao chamar cada vídeo) e
   `sam3 report` (que confere formato e homogeneidade entre todos os `.h5`)
   não compartilham esse ponto cego. Contorno: rode `uv lock`/`uv sync`
   em `sam3_runtime/` antes de extrair um vídeo único, ou passe `--force`.
   `--force` **não** introduz uma segunda precisão no conjunto: quando o
   `.h5` existente já traz um `sam3_inference_autocast_dtype` e o worker
   reporta outro, a reextração falha com `Sam3ExtractError` nomeando o vídeo
   e os dois dtypes, em vez de sobrescrever o artefato — o conjunto inteiro
   precisa ser reextraído, não apenas aquele vídeo. O ponto cego que resta é
   só o da decisão de pular, que continua deliberadamente sem subir o worker.
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
protocolo `cs`, das fixtures de alinhamento de quadro, da construção do
comando/caminhos do worker (`build_worker_invocation`,
`resolve_runtime_project_dir`, `resolve_checkpoint_path`), das rejeições do
gate de `sam3 report` (proveniência ausente/vazia/malformada, `.h5`
estruturalmente inválido), do
[teto de `setuptools`](#teto-de-setuptools-no-sam3_runtime) declarado em
`sam3_runtime/pyproject.toml`/`uv.lock` e da
[política de precisão de inferência](#politica-de-precisao-de-inferencia-autocast)
— a tabela pura `dispositivo × suporte a BF16 → dtype`, a rejeição de
`sam3_inference_autocast_dtype` ausente, vazio ou fora do vocabulário, a
divergência entre dois vídeos extraídos com dtypes diferentes, e um
cruzamento **textual** com o fonte do worker isolado (que o `gatefall` nunca
importa) —, tudo contra um segmentador falso
injetado ou lendo arquivos como dado inerte — não toca no checkpoint do SAM
3, no sub-projeto `sam3_runtime/` nem no dataset real.

```bash
uv run python -m gatefall.sam3.extract extract --video-id <ENV/VIDEO> [--runtime-dir DIR] [--checkpoint PATH] [--force] [--dataset le2i]
```

Extrai `V_t` de um único vídeo e grava o `.h5`. Requer `sam3_runtime/`
sincronizado (`uv sync --project sam3_runtime --locked`) e o checkpoint do
SAM 3 disponível. Um `.h5`
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

Gate de integridade e capacidade pré-C0 sobre os 190 vídeos: valida a
cobertura dos `.h5` de SAM 3 já extraídos contra `frames.parquet`, a
homogeneidade de proveniência entre eles, a estrutura de cada `.h5`
(`storage.validate_existing_file` — presença, shape e dtype corretos de
`v_t`, `sam_score`, `n_instances`) e que os atributos de proveniência
obrigatórios (`sam3_checkpoint_sha256`, `sam3_runtime_lock_sha256`,
`sam3_source_revision`, `sam3_inference_autocast_dtype`) não estão ausentes,
vazios nem malformados (`sam3_source_revision` deve ser um SHA de commit git
hexadecimal minúsculo de 40 caracteres; os dois hashes SHA-256, hexadecimal
minúsculo de 64 caracteres; `sam3_inference_autocast_dtype` deve ser
exatamente `bfloat16` ou `float16`). Qualquer falha resulta em saída
não-zero.

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
pela Meta/`facebookresearch/sam3` — **não verificada nem afirmada por esta
documentação**. Quem for rodar a extração real deve consultar os termos
oficiais diretamente na fonte do modelo antes de baixar, redistribuir ou
publicar resultados obtidos com ele; a licença MIT deste repositório não se
estende a esses materiais.
