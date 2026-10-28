"""Métodos do estudo sob a mesma interface de treino (`gatefall.experiments.physics`)."""

from dataclasses import dataclass

from gatefall.experiments.tcnte import ARM as TCNTE_ARM

GATE_ABLATION_FUSIONS = ("constant", "quality_concat", "learned", "monotonic", "physics")


@dataclass(frozen=True)
class Method:
    name: str
    arm: str
    fusion: str
    visual: str

    @property
    def slug(self) -> str:
        """Diretório do run: idêntico para o mesmo (braço, fusão), então ablações reaproveitam primários."""
        return "tcnte" if self.arm == TCNTE_ARM else f"{self.arm}-{self.fusion}".lower()

    @property
    def family(self) -> str:
        return {"none": "pose", "dinov3": "dino", "sam3": "sam"}[self.visual]


# Todos os oito usam o backend pose138 + evidence8; a diferença entre B1/C1
# learned e physics é só o gate, nunca a representação.
PRIMARY_METHODS = (
    Method("A", "A", "pose", "none"),
    Method("B0", "B0", "concat", "dinov3"),
    Method("B1-learned", "B1", "learned", "dinov3"),
    Method("C0", "C0", "concat", "sam3"),
    Method("C1-learned", "C1", "learned", "sam3"),
    Method("B1-physics", "B1", "physics", "dinov3"),
    Method("C1-physics", "C1", "physics", "sam3"),
    Method("TCNTE", TCNTE_ARM, "tcnte", "none"),
)

GATE_ABLATIONS = tuple(
    Method(f"{arm}-{fusion}", arm, fusion, visual)
    for arm, visual in (("B1", "dinov3"), ("C1", "sam3"))
    for fusion in GATE_ABLATION_FUSIONS
)

# Braços do experimento de movimento, como no módulo importado (5f6e2ae):
# A usa o pacote SAM com o visual ignorado.
MOTION_METHODS = (
    Method("A", "A", "pose", "sam3"),
    Method("B0", "B0", "concat", "dinov3"),
    Method("B1-physics", "B1", "physics", "dinov3"),
    Method("C0", "C0", "concat", "sam3"),
    Method("C1-physics", "C1", "physics", "sam3"),
)

# (tratamento, referência): diferença pareada por semente = tratamento − referência.
PRIMARY_CONTRASTS = (
    ("B1-physics", "B1-learned"),
    ("C1-physics", "C1-learned"),
    ("B0", "A"),
    ("C0", "A"),
    ("B1-learned", "B0"),
    ("C1-learned", "C0"),
    ("TCNTE", "A"),
)
MOTION_CONTRASTS = (("motion_physics", "motion"), ("motion", "baseline"))

METHODS_BY_NAME = {method.name: method for method in PRIMARY_METHODS + GATE_ABLATIONS}
