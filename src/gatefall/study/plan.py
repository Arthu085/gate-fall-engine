"""Matrizes do estudo como passos ordenados, com saídas, dependências e identidade de run."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from gatefall.data.staged.build import staged_paths
from gatefall.data.staged.specs import STAGED_SPECS
from gatefall.datasets import get_dataset
from gatefall.experiments.motion import RECIPES
from gatefall.runs import LOCAL_RUN_ROOTS
from gatefall.study import STUDY_DIR_NAME
from gatefall.study.methods import GATE_ABLATIONS, MOTION_METHODS, PRIMARY_METHODS, Method
from gatefall.study.protocol import BATCH_SIZE, CHECKPOINT_POLICY, EPOCHS, SEEDS, canonical_json, protocol_sha256

MATRICES = ("le2i", "upfall", "transfer", "motion", "ablations")
STAGES = ("data", "extract", "bundle", "targets", "train", "evaluate", "transfer")
LE2I_PROTOCOLS = ("le2i", "le2i-cv")
PRIMARY_DATASET = "upfall"
EXTERNAL_DATASET = "gmdcsa24"
UPFALL_ARCHIVES = Path("data/scratch/upfall_archives")
UV = ("uv", "run", "python", "-m")
VISUAL_EXTRACTORS = {"none": (), "dinov3": ("quality", "dinov3"), "sam3": ("sam3",)}
EXTRACTORS = {
    "pose": ("gatefall.pose.extract", "gpu"),
    "quality": ("gatefall.features.quality_extract", "gpu"),
    "dinov3": ("gatefall.dinov3.extract", "gpu"),
    "sam3": ("gatefall.sam3.extract", "gpu"),
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    method: Method
    seed: int
    recipe: str | None
    run_dir: Path
    bundle: Path
    targets: Path | None = None

    def identity(self) -> dict:
        return {
            "protocol_sha256": protocol_sha256(), "dataset": self.dataset, "arm": self.method.arm,
            "fusion": self.method.fusion, "visual_kind": self.method.visual, "seed": self.seed,
            "motion_recipe": self.recipe, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "checkpoint_policy": CHECKPOINT_POLICY, "augment": False, "order_weight": 0.0,
        }

    def identity_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.identity())).hexdigest()


@dataclass(frozen=True)
class Step:
    key: str
    stage: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    requires: tuple[str, ...]
    resources: str
    run: RunSpec | None = None
    matrices: frozenset[str] = field(default_factory=frozenset)

    @property
    def manual(self) -> bool:
        return not self.command


def study_root(dataset: str) -> Path:
    return LOCAL_RUN_ROOTS[dataset] / STUDY_DIR_NAME


def bundle_dir(dataset: str, visual: str) -> Path:
    return study_root(dataset) / "bundles" / visual


def targets_dir(dataset: str, visual: str) -> Path:
    return study_root(dataset) / "motion-targets" / visual


def run_dir(dataset: str, method: Method, seed: int) -> Path:
    return study_root(dataset) / method.slug / f"seed-{seed}"


def motion_run_dir(dataset: str, method: Method, seed: int, recipe: str) -> Path:
    # Mesmo layout do experimento importado, para auditar runs externos no lugar.
    return LOCAL_RUN_ROOTS[dataset] / f"motion-{method.arm.lower()}" / f"seed-{seed}" / recipe


def pose_root(dataset: str) -> Path:
    return get_dataset(_feature_source(dataset)).pose_root


def _feature_source(dataset: str) -> str:
    return "le2i" if dataset == "le2i-cv" else dataset


class _Builder:
    def __init__(self) -> None:
        self.steps: dict[str, Step] = {}

    def add(self, step: Step, matrix: str) -> str:
        existing = self.steps.get(step.key)
        if existing is None:
            self.steps[step.key] = Step(**{**step.__dict__, "matrices": frozenset({matrix})})
        else:
            if (existing.command, existing.outputs, existing.run) != (step.command, step.outputs, step.run):
                raise ValueError(f"passo {step.key} definido de formas divergentes")
            self.steps[step.key] = Step(**{**existing.__dict__, "matrices": existing.matrices | {matrix}})
        return step.key

    def data(self, dataset: str, matrix: str) -> str:
        if dataset not in STAGED_SPECS:
            frames = get_dataset(dataset).frames_path
            return self.add(Step(f"data:{dataset}", "data", (), (frames,), (), "pré-requisito existente"), matrix)
        spec = STAGED_SPECS[dataset]
        paths = staged_paths(spec)
        previous: tuple[str, ...] = ()
        if dataset == "upfall":
            previous = (self.add(Step(
                "data:upfall:stage", "data",
                (*UV, "gatefall.data.staged", "stage-archives", "--archives", str(UPFALL_ARCHIVES)),
                (spec.raw_dir,), (), "download manual + disco"), matrix),)
        else:
            previous = (self.add(Step(f"data:{dataset}:raw", "data", (), (spec.raw_dir,), (),
                                      "download manual"), matrix),)
        sequence = (
            ("fetch-labels", (spec.labels_dir / "PROVENANCE.json",), "rede (Hugging Face)"),
            ("verify-labels", (), "cpu"),
            ("manifest", (paths.manifest, paths.provenance), "cpu + leitura completa das fontes"),
            ("timegrid", (paths.frames, paths.alignment), "cpu"),
            ("audit", (paths.coverage,), "cpu"),
        )
        for command, outputs, resources in sequence:
            key = self.add(Step(f"data:{dataset}:{command}", "data",
                                (*UV, "gatefall.data.staged", command, "--dataset", dataset),
                                outputs, previous, resources), matrix)
            previous = (key,)
        return previous[0]

    def extraction(self, dataset: str, visual: str, matrix: str) -> tuple[str, ...]:
        source = _feature_source(dataset)
        data_keys = tuple(dict.fromkeys((self.data(source, matrix), self.data(dataset, matrix))))
        names = ("pose", *VISUAL_EXTRACTORS[visual])
        keys = []
        previous = data_keys
        for name in names:
            module, resources = EXTRACTORS[name]
            key = self.add(Step(f"extract:{source}:{name}", "extract",
                                (*UV, module, "extract-all", "--dataset", source), (), previous,
                                f"{resources}; extract-all pula vídeos já extraídos"), matrix)
            keys.append(key)
            if name == "pose":
                previous = (key,)
        return tuple(keys)

    def bundle(self, dataset: str, visual: str, matrix: str) -> str:
        requires = self.extraction(dataset, visual, matrix)
        output = bundle_dir(dataset, visual)
        return self.add(Step(f"bundle:{dataset}:{visual}", "bundle",
                             (*UV, "gatefall.experiments.physics", "prepare", "--dataset", dataset,
                              "--visual", visual, "--output", str(output)),
                             (output / "manifest.json",), requires, "cpu + leitura de features"), matrix)

    def train(self, run: RunSpec, requires: tuple[str, ...], matrix: str) -> str:
        method = run.method
        common = ("--bundle", str(run.bundle), "--run-dir", str(run.run_dir), "--arm", method.arm,
                  "--fusion", method.fusion, "--seed", str(run.seed), "--epochs", str(EPOCHS),
                  "--batch-size", str(BATCH_SIZE), "--checkpoint-policy", CHECKPOINT_POLICY)
        if run.recipe is None:
            command = (*UV, "gatefall.experiments.physics", "train", *common)
            key = f"train:{run.dataset}:{method.slug}:seed-{run.seed}"
        else:
            command = (*UV, "gatefall.experiments.motion", "train", *common,
                       "--targets", str(run.targets), "--recipe", run.recipe)
            key = f"train:{run.dataset}:motion-{method.arm.lower()}:seed-{run.seed}:{run.recipe}"
        return self.add(Step(key, "train", command, (run.run_dir / "metrics.json",), requires,
                             "gpu opcional (cpu viável)", run), matrix)

    def evaluate(self, run: RunSpec, train_key: str, matrix: str) -> str:
        module = "gatefall.experiments.physics" if run.recipe is None else "gatefall.experiments.motion"
        key = train_key.replace("train:", "evaluate:", 1)
        return self.add(Step(key, "evaluate",
                             (*UV, module, "evaluate", "--bundle", str(run.bundle), "--run-dir",
                              str(run.run_dir), "--split", "test"),
                             (run.run_dir / "evaluation-test.json",), (train_key,), "cpu", run), matrix)

    def transfer(self, run: RunSpec, train_key: str, matrix: str) -> str:
        target = self.bundle(EXTERNAL_DATASET, run.method.visual, matrix)
        key = train_key.replace("train:", f"transfer:{EXTERNAL_DATASET}:", 1)
        return self.add(Step(key, "transfer",
                             (*UV, "gatefall.experiments.physics", "evaluate-external", "--bundle",
                              str(run.bundle), "--run-dir", str(run.run_dir), "--target-bundle",
                              str(bundle_dir(EXTERNAL_DATASET, run.method.visual))),
                             (run.run_dir / f"evaluation-external-{EXTERNAL_DATASET}.json",),
                             (train_key, target), "cpu", run), matrix)


def primary_runs(dataset: str, methods: tuple[Method, ...] = PRIMARY_METHODS) -> list[RunSpec]:
    return [RunSpec(dataset, method, seed, None, run_dir(dataset, method, seed),
                    bundle_dir(dataset, method.visual))
            for method in methods for seed in SEEDS]


def motion_runs(dataset: str) -> list[RunSpec]:
    return [RunSpec(dataset, method, seed, recipe, motion_run_dir(dataset, method, seed, recipe),
                    bundle_dir(dataset, method.visual), targets_dir(dataset, method.visual))
            for method in MOTION_METHODS for seed in SEEDS for recipe in RECIPES]


def build_plan(matrices: tuple[str, ...] = MATRICES) -> list[Step]:
    unknown = set(matrices) - set(MATRICES)
    if unknown:
        raise ValueError(f"matriz desconhecida: {sorted(unknown)}")
    builder = _Builder()
    for matrix in matrices:
        if matrix in ("le2i", "upfall", "ablations"):
            datasets = LE2I_PROTOCOLS if matrix == "le2i" else (PRIMARY_DATASET,)
            methods = GATE_ABLATIONS if matrix == "ablations" else PRIMARY_METHODS
            for dataset in datasets:
                runs = primary_runs(dataset, methods)
                trained = [(run, builder.train(run, (builder.bundle(dataset, run.method.visual, matrix),), matrix))
                           for run in runs]
                for run, key in trained:
                    builder.evaluate(run, key, matrix)
        elif matrix == "transfer":
            for run in primary_runs(PRIMARY_DATASET):
                key = builder.train(run, (builder.bundle(PRIMARY_DATASET, run.method.visual, matrix),), matrix)
                builder.transfer(run, key, matrix)
        else:
            for dataset in LE2I_PROTOCOLS:
                trained = []
                for run in motion_runs(dataset):
                    bundle = builder.bundle(dataset, run.method.visual, matrix)
                    output = targets_dir(dataset, run.method.visual)
                    targets = builder.add(Step(
                        f"targets:{dataset}:{run.method.visual}", "targets",
                        (*UV, "gatefall.experiments.motion", "prepare", "--bundle", str(run.bundle),
                         "--pose-root", str(pose_root(dataset)), "--output", str(output)),
                        (output / "motion-manifest.json",), (bundle,), "cpu"), matrix)
                    trained.append((run, builder.train(run, (bundle, targets), matrix)))
                for run, key in trained:
                    builder.evaluate(run, key, matrix)
    order = {stage: index for index, stage in enumerate(STAGES)}
    steps = list(builder.steps.values())
    # Estável por estágio: todo treino da matriz termina antes de qualquer avaliação de teste.
    return sorted(steps, key=lambda step: order[step.stage])


def matrix_summary(steps: list[Step]) -> dict:
    summary: dict = {}
    for matrix in MATRICES:
        members = [step for step in steps if matrix in step.matrices]
        if not members:
            continue
        summary[matrix] = {stage: sum(step.stage == stage for step in members) for stage in STAGES}
    summary["unique"] = {stage: sum(step.stage == stage for step in steps) for stage in STAGES}
    return summary
