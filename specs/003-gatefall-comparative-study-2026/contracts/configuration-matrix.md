# Contrato: matriz de configurações

## Métodos primários (`gatefall.study.methods.PRIMARY_METHODS`)

| Método | `--arm` | `--fusion` | Pacote | Diretório |
| --- | --- | --- | --- | --- |
| A | A | pose | none | `a-pose` |
| B0 | B0 | concat | dinov3 | `b0-concat` |
| B1 learned | B1 | learned | dinov3 | `b1-learned` |
| C0 | C0 | concat | sam3 | `c0-concat` |
| C1 learned | C1 | learned | sam3 | `c1-learned` |
| B1 physics | B1 | physics | dinov3 | `b1-physics` |
| C1 physics | C1 | physics | sam3 | `c1-physics` |
| TCNTE adaptado | TCNTE | tcnte | none | `tcnte` |

Todos os métodos usam pose138 + evidence8. As cinco configurações históricas (pose134) são
referência congelada e não fazem parte desta tabela.

## Ablações de gate (fora da tabela primária)

O gate varia em `constant`, `quality_concat`, `learned`, `monotonic` e `physics`, para B1
(DINOv3) e C1 (SAM 3). `learned` e `physics` reaproveitam os runs primários do UP-Fall, porque
têm a mesma identidade e o mesmo diretório. O contraste de referência de cada variante é a
variante `constant`.

## Contrastes primários (tratamento − referência)

| Tratamento | Referência | Pergunta |
| --- | --- | --- |
| B1 physics | B1 learned | efeito do gate físico (DINO) |
| C1 physics | C1 learned | efeito do gate físico (SAM) |
| B0 | A | contexto DINOv3 concatenado |
| C0 | A | descritor SAM 3 concatenado |
| B1 learned | B0 | gate aprendido sobre concatenação (DINO) |
| C1 learned | C0 | gate aprendido sobre concatenação (SAM) |
| TCNTE | A | referência externa adaptada × TCN do projeto |

Movimento: `motion_physics − motion` (primário) e `motion − baseline` (efeito auxiliar), por braço.

## Matrizes

| Matriz | Composição | Treinos | Avaliações |
| --- | --- | --- | --- |
| `le2i` | 8 métodos × 5 sementes × {Le2i-CS, Le2i-CV} | 80 | 80 de teste |
| `upfall` | 8 × 5 no UP-Fall | 40 | 40 de teste |
| `transfer` | runs da `upfall` → GMDCSA-24 | 0 novos | 40 externas |
| `motion` | 5 braços (A pose/SAM, B0, B1 physics, C0, C1 physics) × 3 receitas × 5 sementes × 2 protocolos | 150 | 150 de teste |
| `ablations` | 5 gates × {B1, C1} × 5 sementes no UP-Fall | 50 (30 novos) | 50 de teste |
| **Total único** | | **300** | **300 + 40** |

Sem reuso, a estimativa é de 300 treinos de 30 épocas, 300 avaliações de teste em CPU e 40
avaliações externas. A extração cobre UP-Fall (cerca de 1118 vídeos de câmera) e GMDCSA-24
(160 clipes) com quatro extratores; o Le2i já está extraído no checkout principal. O reuso
compatível dos 150 runs de movimento reduziria os treinos únicos para 150.
