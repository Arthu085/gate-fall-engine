"""Arquivos CMDFall do OmniFall fixados por revisão e SHA-256, sem nenhum vídeo."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from gatefall.hashing import sha256_file

DATASET_REPO_ID = "simplexsigil2/omnifall"
# Mesma revisão das anotações Le2i (gatefall.data.le2i.annotations). Os oito
# arquivos abaixo têm SHA-256 idêntico na head 83572a37 (2026-08-27).
OMNIFALL_REVISION = "68e5cee56a4bad38cca4aea791cac248f96e79a0"
SCRATCH_ROOT = Path("data/scratch/cmdfall")
DEFAULT_SOURCE_DIR = SCRATCH_ROOT / "omnifall"
DEFAULT_REPORT_PATH = SCRATCH_ROOT / "audit.json"
PROTECTED_PATHS = (
    Path(".git"),
    Path(".github"),
    Path("src"),
    Path("scripts"),
    Path("docs"),
    Path("runs/reference"),
    Path("data/raw"),
    Path("data/labels"),
    Path("data/processed"),
    Path("data/features"),
)

SEGMENTS_FILE = "labels/cmdfall.csv"
LABEL2ID_FILE = "labels/label2id.csv"
CS_SPLIT_FILES = {
    "train": "splits/cs/cmdfall/train.csv",
    "val": "splits/cs/cmdfall/val.csv",
    "test": "splits/cs/cmdfall/test.csv",
}
CV_SPLIT_FILES = {
    "train": "splits/cv/cmdfall/train.csv",
    "val": "splits/cv/cmdfall/val.csv",
    "test": "splits/cv/cmdfall/test.csv",
}

EXPECTED_SHA256 = {
    SEGMENTS_FILE: "6244c5ca0069df788632ac313f9c44d6a120465a4940e98e02c1daf1975eda28",
    LABEL2ID_FILE: "e5e2de05f00818378ad36b3b310f4c9fc28d2300bc80aa25599de7e9a4ca3b7e",
    CS_SPLIT_FILES["train"]: "8974ebdb1c85b87bd3267cd423ef5e3a5f17e380eb5b86cab76b1513f8361e6b",
    CS_SPLIT_FILES["val"]: "8f7ed61cee7b2f347eeb1bc2cf223b6ec150a740f82b8167e04841a277ec506d",
    CS_SPLIT_FILES["test"]: "b6a89c1335ef1732318ca73cb745ba88c14a7e4736d7dafb8b6b33a984da2205",
    CV_SPLIT_FILES["train"]: "35c9875fc0e3192cc9f4e8004436db989a61dab0ea71d73a87f9917d997867ef",
    CV_SPLIT_FILES["val"]: "fd215d53b2008cdc1b33f30c8f48b029b32fa6177295272cdf930424493b9ade",
    CV_SPLIT_FILES["test"]: "d104088dca92eec2141c4005a399a117258e1d349bba4ee6b9282999e15ab1e6",
}


def destination_problems(destination: Path, repository_root: Path) -> list[str]:
    """Caminhos são comparados já resolvidos, para que `..` e links simbólicos
    não levem a escrita para fora de SCRATCH_ROOT nem para áreas protegidas."""
    repository = repository_root.resolve()
    allowed = (repository_root / SCRATCH_ROOT).resolve()
    target = (repository_root / destination).resolve()
    problems: list[str] = []
    if not allowed.is_relative_to(repository):
        problems.append(f"{SCRATCH_ROOT} resolve para {allowed}, fora do repositório")
    if target == allowed or not target.is_relative_to(allowed):
        problems.append(f"{destination}: resolve para {target}, fora de {SCRATCH_ROOT}/")
    problems += [
        f"{destination}: resolve para {target}, dentro da área protegida {protected}/"
        for protected in PROTECTED_PATHS
        if target.is_relative_to((repository_root / protected).resolve())
    ]
    return problems


def fetch_sources(source_dir: Path) -> None:
    for filename in EXPECTED_SHA256:
        hf_hub_download(
            DATASET_REPO_ID,
            filename,
            repo_type="dataset",
            revision=OMNIFALL_REVISION,
            local_dir=source_dir,
        )
        print(f"{source_dir / filename}: baixado de {DATASET_REPO_ID}@{OMNIFALL_REVISION}")


def source_problems(
    source_dir: Path, expected_sha256: dict[str, str] = EXPECTED_SHA256
) -> list[str]:
    problems: list[str] = []
    for filename, expected in expected_sha256.items():
        path = source_dir / filename
        if not path.is_file():
            problems.append(f"{path}: arquivo ausente")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{path}: sha256 divergente (esperado {expected}, encontrado {actual})")
    return problems
