# Auditoria de caixas manuais do Le2i

A distribuição original do Le2i inclui arquivos `Annotation_files/video (N).txt` para 108 dos 190 vídeos do corpus GateFall, em três ambientes. A auditoria usa apenas os arquivos presentes e relata a cobertura observada por ambiente. Os demais vídeos não entram nos denominadores de detecção e IoU.

Esta análise é um diagnóstico **pós-hoc** da detecção e da localização automática do YOLO-Pose. As caixas manuais não entram em `frames.parquet`, nos HDF5 de pose, nos datasets de treino ou nos resultados de referência. Elas não constituem novo ground truth para seleção de modelos nem um classificador de oclusão.

## Execução

Depois de preparar o manifesto, a grade de 10 Hz, os vídeos originais com `Annotation_files/` e os HDF5 de pose, execute na raiz do repositório:

```bash
uv run python -m gatefall.pose.manual_bbox_audit report
uv run python -m gatefall.pose.manual_bbox_audit report --output data/scratch/le2i-manual-bbox-audit.json
uv run python -m gatefall.pose.manual_bbox_audit selftest
```

O JSON é opcional e só é gravado no caminho indicado. O comando recusa sobrescrever um arquivo existente ou escrever sob `runs/reference/`. Os testes sintéticos não exigem os dados reais.

## Leitura dos resultados

Cada linha com um inteiro é metadado de limite de queda; cada linha de quadro contém seis inteiros separados por vírgula, por exemplo `1,1,292,152,311,240`, com espaços opcionais ao redor dos campos. As colunas representam quadro, valor auxiliar e `x1,y1,x2,y2`. O valor auxiliar é preservado pelo parser sem receber semântica presumida. Apenas a caixa exata `(0,0,0,0)` é contada como sentinela zero; linhas com coordenadas negativas são contadas separadamente. Outras caixas com área zero, caixas invertidas, não finitas ou fora da resolução do manifesto e estruturas malformadas causam erro. A quantidade de linhas de anotação não é usada como contagem de quadros do vídeo: cada número de quadro é validado contra `n_frames_counted` do manifesto.

O número do quadro manual começa em 1. A auditoria procura correspondência exata com `src_index + 1` de `frames.parquet`; não interpola nem reamostra caixas. Nos quadros amostrados que têm pelo menos uma caixa manual válida, relata recall de `person_found`, quantidade e fração de falhas, e distribuição do maior IoU com a caixa de pose quando ela existe. Registra também `manual_box_count` por quadro, a frequência de múltiplas caixas e sequências consecutivas de falhas na grade amostrada. Quadros sem caixa manual válida interrompem uma sequência.

Os resumos incluem ambiente, split e rótulo de quadro. A comparação rotulada separa `fall`/`fallen` dos outros rótulos válidos; quadros com `IGNORE_LABEL` aparecem em um grupo `ignored` próprio e continuam no resumo geral. O maior IoU entre várias caixas mede somente a localização: **não comprova identidade correta**, pois GateFall não usa semântica documentada para a coluna auxiliar da anotação.
