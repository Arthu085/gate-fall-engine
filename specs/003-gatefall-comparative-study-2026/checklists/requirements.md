# Checklist de requisitos

## Qualidade da especificação

- [x] O objetivo e o escopo declaram o que é deslocado (reexecução histórica pose134; CAUCAFall
  só como fallback).
- [x] Os critérios de aceitação são verificáveis por selftest sintético.
- [x] Os resultados científicos estão marcados como pendentes, sem números inventados.
- [x] As fontes externas aparecem com revisão e licença.

## Invariantes

- [x] Os métodos diferem só pelo vetor por passo e pela fusão; protocolo, janela, partição,
  sementes, épocas e alarme são compartilhados.
- [x] As partições são por sujeito; vistas sincronizadas e gravações nunca são divididas.
- [x] A entrada é só RGB monocular.
- [x] Os backbones ficam congelados, com features offline por vídeo.
- [x] O HDF5 é organizado por vídeo.

## Cobertura de regressão

- [x] Isolamento de partição: staged selftest (vazamento, vistas, duplicatas, split OmniFall).
- [x] Alinhamento temporal: staged selftest (último quadro ≤ t com amostragem irregular, empates, quadros perdidos e vídeo; Le2i histórico; rótulo após a fonte).
- [x] Reuso seguro: staged e study selftests (preparação parcial do UP-Fall, artefatos corrompidos, anotações alteradas, pacote incompleto, retomada).
- [x] Bootstrap por sujeito: study selftest (várias gravações por sujeito, sujeito ambíguo, teste incompleto).
- [x] Mapeamento de rótulos: staged selftest (classe 12 → `IGNORE`, rótulos adulterados).
- [x] Vazamento de normalização: study selftest (estatísticas idênticas com val/test perturbados).
- [x] Causalidade de inferência: study selftest (9 classificadores).
- [x] Proveniência de checkpoint: study selftest (identidade e checkpoint adulterados).
- [x] Correção da avaliação: study selftest (durante/após a queda, FA/h, latência com perdas).
- [x] Compatibilidade de artefatos: TCNTE selftest (transferência) e study selftest (relatório,
  reuso).
