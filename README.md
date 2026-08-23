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

| Config | Conteúdo de cada linha da janela    | Codificador temporal |
| ------ | ----------------------------------- | -------------------- |
| **A**  | Pose (YOLO-Pose)                    | TCN                  |
| **B**  | Pose + embedding visual (DINOv3)    | TCN                  |
| **C**  | Pose + descritor de máscara (SAM 3) | TCN                  |

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

## Licença

O **código** deste repositório é distribuído sob a licença [MIT](LICENSE).
