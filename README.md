# GateFall

**Fusão Adaptativa por Confiança entre Pose e Informação Visual de Modelos de
Fundação para Detecção Robusta de Quedas Humanas**

Trabalho de Conclusão de Curso (TCC) em Visão Computacional.
**Autor:** Arthur Ghizi · **Orientador:** Rodrigo Ramos Silva.

## Pergunta de pesquisa

> Em que medida acrescentar informação visual de modelos de fundação congelados
> (DINOv3, SAM 3) a um detector de quedas baseado em pose (YOLO-Pose) aumenta a
> robustez da detecção em vídeo RGB monocular — e como a fusão deve ponderar
> pose e informação visual em função da confiança de cada fonte, nos regimes em
> que a estimativa de pose se degrada (oclusão, truncamento, poses atípicas do
> corpo caído)?

## Configurações experimentais

| Config | Conteúdo de cada linha da janela | Codificador temporal |
| --- | --- | --- |
| **A** | Pose (YOLO-Pose) | TCN |
| **B** | Pose + embedding visual (DINOv3) | TCN |
| **C** | Pose + DINOv3 + descritor de máscara (SAM 3) | TCN |

**As três configurações são mantidas rigorosamente idênticas em tudo exceto no
conteúdo de cada linha da janela.** Tamanho de janela, split treino/teste, seed,
codificador temporal e número de épocas são os mesmos em A, B e C; a única
variável entre elas é o vetor de features por timestep. Qualquer alteração que
afete apenas uma das configurações invalida a comparação.

## Instalação

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12 (o uv baixa a versão
automaticamente a partir do `.python-version`).

```bash
git clone https://github.com/Arthu085/gate-fall-engine.git
cd gate-fall-engine
uv sync
```

Documentação local:

```bash
uv run mkdocs serve
```

## Dados

Os datasets **não são redistribuídos neste repositório** e devem ser baixados
diretamente das fontes originais para `data/` (diretório versionado vazio, com
conteúdo ignorado pelo Git):

| Dataset | Fonte | Licença |
| --- | --- | --- |
| UR Fall Detection (URFD) | http://fenix.ur.edu.pl/~mkepski/ds/uf.html | CC BY-NC-SA 4.0 |
| OmniFall | https://huggingface.co/datasets/simplexsigil2/omnifall | CC BY-NC 4.0 |

Ambas as licenças são **não comerciais** e incompatíveis com a redistribuição a
partir deste repositório. O uso é restrito aos termos de cada licença original;
consulte-os antes de qualquer reaproveitamento.

O pipeline é **RGB monocular**: nenhum descritor derivado de mapa de
profundidade é utilizado — incluindo as colunas D–K do CSV de features do URFD,
calculadas a partir do sensor de profundidade do Kinect.

### URFD: download e validação

```bash
scripts/download_urfd.sh
```

Baixa os 70 vídeos câmera 0 do URFD (30 quedas + 40 ADLs) para
`data/urfd/videos/{fall,adl}/`. É idempotente: reexecutar pula os arquivos já
presentes e válidos, sem nova requisição de rede.

Os CSVs de labels (`urfall-cam0-falls.csv` e `urfall-cam0-adls.csv`) **não são
baixados por este script** — baixe-os manualmente da fonte original e
coloque-os em `data/urfd/labels/` antes de validar.

```bash
uv run scripts/validate_urfd.py
```

Cruza os CSVs de labels contra os frames de cada vídeo já baixado, detecta
lacunas de `frame_idx` e anomalias de rótulo, e grava um relatório em
`data/urfd/validation_report.csv`.

Layout em disco, schema dos CSVs e as conclusões da validação (convenção de
rótulo, base de indexação do frame, lacunas encontradas) estão documentados em
`docs/urfd.md`.

## Licença

O **código** deste repositório é distribuído sob a licença [MIT](LICENSE).
Os **datasets** não são cobertos por ela e permanecem sob suas licenças
originais (URFD: CC BY-NC-SA 4.0; OmniFall: CC BY-NC 4.0).
