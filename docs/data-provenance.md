# Proveniência dos dados

Esta página documenta de onde vêm as anotações do Le2i usadas neste projeto,
sob qual licença, e como reproduzir o download com garantia de integridade.
Ela existe porque os CSVs de anotação **não são versionados** neste
repositório (ver `.gitignore` e a seção "Dados" do README) — a
reprodutibilidade vem da revisão fixada do dataset upstream e dos checksums
abaixo, não de um arquivo commitado.

## Fonte e revisão fixada

As anotações são obtidas do dataset agregado
[`simplexsigil2/omnifall`](https://huggingface.co/datasets/simplexsigil2/omnifall)
no HuggingFace Hub, que resolve a correspondência entre vídeos, splits e
rótulos de queda entre os datasets originais que compõem o OmniFall.

`scripts/fetch_labels.py` fixa a revisão do dataset em um SHA de commit
específico do repositório HuggingFace, em vez de seguir a `main` (`HEAD`)
móvel:

| Campo                | Valor                                      |
| --------------------- | ------------------------------------------- |
| `dataset_repo_id`     | `simplexsigil2/omnifall`                    |
| `OMNIFALL_REVISION`   | `68e5cee56a4bad38cca4aea791cac248f96e79a0`  |
| Configs usados        | `le2i-cs`, `labels`                         |

Os nomes desses configs já foram reestruturados uma vez no histórico do
dataset upstream. Fixar a revisão garante que uma futura reestruturação não
altere silenciosamente as colunas, os splits ou o conteúdo baixado por este
repositório.

## Contagens esperadas

Para a revisão fixada acima, o config `le2i-cs` (splits oficiais por sujeito)
e o config `labels` (filtrado para `dataset == "le2i"`) produzem:

| Arquivo      | Origem                          | Linhas |
| ------------ | -------------------------------- | -----: |
| `train.csv`  | `le2i-cs`, split `train`         |    670 |
| `val.csv`    | `le2i-cs`, split `validation`    |     94 |
| `test.csv`   | `le2i-cs`, split `test`          |    203 |
| `le2i.csv`   | `labels`, filtrado para Le2i     |    967 |

`670 + 94 + 203 = 967`, ou seja, `le2i.csv` cobre exatamente a união dos três
splits de `le2i-cs` — é usado para conferência cruzada, não como split
adicional.

## Reprodução e verificação de integridade

Baixar as anotações (fixadas em `OMNIFALL_REVISION`):

```bash
uv run python scripts/fetch_labels.py
```

O script grava, junto dos CSVs, um `data/labels/omnifall/PROVENANCE.json`
com `dataset_repo_id`, `revision`, `configs`, o timestamp UTC do fetch e,
para cada CSV, seu nome, hash SHA-256 e número de linhas.

Para conferir que os arquivos em disco continuam batendo com o que foi
registrado em `PROVENANCE.json` (sem baixar nada de novo):

```bash
uv run python scripts/fetch_labels.py --verify
```

O comando termina com código de saída diferente de zero e lista cada arquivo
ausente ou com hash divergente, caso encontre alguma inconsistência.

## Licenças

Este repositório mistura três escopos de licença **distintos**, sobre
artefatos diferentes — é exatamente por isso que os CSVs de anotação não
podem ser versionados junto do código:

| Artefato                          | Licença                                                                          |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| Código deste repositório           | MIT                                                                                |
| Anotações do OmniFall (CSVs)       | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — uso não comercial |
| Vídeos originais do Le2i           | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) — uso não comercial |

CC BY-NC-SA é incompatível com a distribuição junto de código MIT em um
repositório público: exige atribuição, proíbe uso comercial e obriga
compartilhamento pela mesma licença para qualquer redistribuição — condições
que não fazem sentido aplicadas ao código do projeto. Por isso as anotações
são sempre baixadas sob demanda a partir da fonte original, nunca commitadas.

## Citações

**OmniFall** (dataset agregado, fonte das anotações baixadas por este
repositório):

> Schneider, D., Marinov, Z., Mistol, M., Zhong, Z., Jaus, A., Düger, R.,
> Baur, R., Sarfraz, M. S., & Stiefelhagen, R. (2025). *OmniFall: From
> Staged Through Synthetic to Wild, A Unified Multi-Domain Dataset for
> Robust Fall Detection*. arXiv:2505.19889.
> https://arxiv.org/abs/2505.19889

**Le2i** (dataset original dos vídeos, cujas anotações o OmniFall
reorganiza):

> Dubois, J., & Miteran, J. (2014). *Fall Detection Dataset (Le2i)*.
> Laboratoire Electronique, Informatique et Image (Le2i), Université
> Bourgogne Franche-Comté. DOI: 10.25666/DATAUBFC-2024-04-09.
