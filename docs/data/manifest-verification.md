# Manifesto e verificação

`data/manifest.parquet` contém uma linha por vídeo e relaciona o arquivo local à
anotação do OmniFall, aos metadados medidos e ao estado das futuras etapas de
features. O manifesto é gerado localmente e não é versionado.

## Construção

Depois de preparar as [anotações](omnifall.md) e os [vídeos do
Le2i](le2i.md), execute:

```bash
uv run python -m gatefall.data.ingest ingest
```

A ingestão:

1. reúne os segmentos dos três splits em um índice com uma linha por path de
   vídeo e exige `subject`, `cam` e `split` consistentes entre seus segmentos;
2. descobre os arquivos `.avi` e exige a bijeção descrita na página do Le2i;
3. usa `ffprobe` para ler metadados e contar os quadros;
4. calcula o SHA-256 de cada vídeo;
5. aplica o schema e a ordem de colunas definidos pelo projeto;
6. ordena as linhas por `video_id` e grava o Parquet.

Se `data/manifest.parquet` já existir, o comando o preserva e termina sem erro.
Para reconstruí-lo:

```bash
uv run python -m gatefall.data.ingest ingest --force
```

A gravação usa primeiro `data/manifest.parquet.tmp` e só então substitui o
destino. Assim, uma falha antes da substituição não publica um manifesto
parcial.

## Schema

As colunas são persistidas nesta ordem:

| Colunas | Tipo | Conteúdo |
| --- | --- | --- |
| `video_id`, `dataset` | `string` | Chave normalizada e nome do dataset |
| `relative_path`, `absolute_path` | `string` | Caminhos relativo e absoluto do vídeo |
| `env` | `string` | Ambiente obtido do path anotado |
| `subject`, `cam` | `int64` | Identificadores publicados na anotação |
| `split` | `string` | `train`, `val` ou `test` |
| `fps` | `float64` | Taxa de quadros resolvida pelo `ffprobe` |
| `fps_source` | `string` | Campo usado: `avg_frame_rate` ou `r_frame_rate` |
| `n_frames_header` | `Int64` | Contagem declarada no header, se disponível |
| `n_frames_counted` | `int64` | Contagem obtida com `ffprobe -count_frames` |
| `duration_s` | `float64` | Duração em segundos |
| `width`, `height` | `int64` | Resolução do vídeo |
| `codec` | `string` | Nome do codec |
| `sha256` | `string` | Hash do arquivo de vídeo |
| `pose_status`, `dino_status`, `sam_status` | `string` | Estado de cada branch de features |

Na ingestão atual, os três status começam como `pending`; isso não indica que a
extração de features tenha sido implementada.

O FPS nunca é assumido como constante. A resolução prefere
`avg_frame_rate` quando o valor é válido e maior que zero; caso contrário, usa
`r_frame_rate`. Nos arquivos avaliados, `Home_01` e `Home_02` ficam em torno de
23,9997 fps e os demais ambientes em 25 fps, por isso o valor é sempre medido
por vídeo.

## Verificação

Execute a verificação sobre o manifesto já construído:

```bash
uv run python -m gatefall.data.ingest verify
```

O comando recalcula propriedades a partir dos vídeos, das anotações e do
manifesto. As saídas têm dois papéis distintos.

### Verificações críticas

Qualquer falha abaixo faz o comando terminar com código diferente de zero:

- bijeção entre os vídeos locais e os paths anotados;
- disjunção de paths de vídeo entre `train`, `val` e `test`;
- presença de cada vídeo referenciado pelo manifesto e igualdade de seu
  SHA-256 com o valor persistido.

Todas as verificações são executadas antes do resumo final, permitindo relatar
mais de um problema na mesma execução.

### Relatórios informativos

Os relatórios abaixo descrevem o conjunto, mas não determinam sucesso ou falha:

- sobreposição de `subject` entre os splits;
- distribuição de resolução e vídeos fora da resolução modal;
- distribuição de FPS por ambiente;
- tabela cruzada entre `cam` e `env` e avaliação da relação entre ambos;
- estatísticas de duração dos segmentos por classe;
- quantidade de segmentos por classe e split, com avisos para combinações
  vazias;
- duração total e projeção de quadros a 10, 12,5 e 25 fps.

Em particular, a disjunção por sujeito é informativa. O identificador upstream
`le2i-cs` não substitui essa medição e o nome da configuração não deve ser
interpretado, isoladamente, como garantia de um split cross-subject.

## Auditoria de cobertura

Execute a auditoria de cobertura das anotações sobre o manifesto já construído:

```bash
uv run python -m gatefall.data.coverage audit
```

O comando mede o quanto os segmentos anotados (`train.csv`, `val.csv` e
`test.csv`, unidos) cobrem a duração de cada vídeo do Le2i. A duração canônica
de um vídeo é `n_frames_counted / fps`, lida do manifesto — não o
`duration_s` bruto do container. A auditoria é somente leitura: não grava no
manifesto, nas anotações nem em `data/raw/`.

Para cada vídeo, calcula:

| Métrica | Significado |
| --- | --- |
| `n_segments` | Quantidade de segmentos anotados |
| `segments_total_s` | Soma da duração dos segmentos |
| `gap_s` | Tempo do vídeo não coberto por nenhum segmento |
| `overlap_s` | Soma das sobreposições entre pares de segmentos |
| `overhang_s` | Tempo de segmentos que ultrapassa o fim do vídeo |
| `duration_delta_s` | `abs(duration_s - n_frames_counted / fps)` |

### Relatórios

- totais agregados de `segments_total_s`, `gap_s`, `overlap_s` e `overhang_s`,
  com `gap_s` também como percentual da duração total;
- os 10 piores vídeos por `gap_s`, `overlap_s`, `overhang_s` e
  `duration_delta_s`;
- quantis de `gap_s` (mínimo, p25, mediana, p75, máximo);
- contagem de vídeos perfeitamente cobertos (`tiled`): `gap_s`, `overlap_s` e
  `overhang_s` todos abaixo da duração de um quadro. A tolerância é medida por
  vídeo (`1 / fps`), pois o fps não é constante entre ambientes — ver
  distribuição de fps acima;
- `gap_s` somado e como percentual, agrupado por `env` e por `split`;
- decomposição do `gap_s` em `leading_gap_s` (antes do primeiro segmento),
  `trailing_gap_s` (depois do último segmento) e `interior_gap_s` (entre
  segmentos), agregada no total e por `env`;
- tabela de `trailing_gap_s` por vídeo em `Home_01`/`Home_02` (fps ≈
  24,000384, diferente dos demais ambientes), com a razão
  `trailing_gap_s / video_duration_s`, e a correlação de Pearson entre
  `trailing_gap_s` e `video_duration_s` por `env` — para expor a assinatura de
  um possível desvio sistemático de conversão de fps nesses dois ambientes;
- distribuição (contagem, mínimo, p25, mediana, p75, máximo) da duração dos
  gaps interiores individuais, agrupados de todos os vídeos;
- lista de vídeos sem nenhum segmento anotado.

### Cross-check entre fontes de anotação

O comando também compara `data/labels/omnifall/le2i.csv` (configuração
`labels`, filtrada por Le2i) com a união de `train.csv` + `val.csv` +
`test.csv` (configuração `le2i-cs`), tupla a tupla por `(path, start, end,
label)`. Divergências são impressas em até 20 exemplos por lado.

### Falhas críticas

Apenas três problemas fazem o comando terminar com código diferente de zero:

- `interior_gap_s` negativo em algum vídeo (violação do invariante interno da
  decomposição do `gap_s`);
- algum vídeo sem nenhum segmento anotado;
- divergência entre `le2i.csv` e a união dos splits.

`gap_s`, `overlap_s`, `overhang_s` e `duration_delta_s` são achados, não
falhas: a auditoria os relata mesmo quando o comando termina com sucesso.
