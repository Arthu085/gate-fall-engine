"""Arquivos CMDFall do OmniFall fixados por revisão e SHA-256, sem nenhum vídeo."""

import contextlib
import os
import tempfile
from collections.abc import Callable
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
    if allowed != repository / SCRATCH_ROOT:
        problems.append(
            f"{SCRATCH_ROOT} resolve para {allowed}, não para o diretório físico "
            f"{repository / SCRATCH_ROOT}"
        )
    if target == allowed or not target.is_relative_to(allowed):
        problems.append(f"{destination}: resolve para {target}, fora de {SCRATCH_ROOT}/")
    problems += [
        f"{destination}: resolve para {target}, dentro da área protegida {protected}/"
        for protected in PROTECTED_PATHS
        if target.is_relative_to((repository_root / protected).resolve())
    ]
    return problems


def publish_file(destination: Path, payload: bytes) -> None:
    """`mkstemp` cria o temporário com O_CREAT | O_EXCL e nome imprevisível no
    diretório do destino, então um link simbólico preexistente nunca é seguido;
    `os.replace` troca o destino de uma vez. Em falha, o temporário é removido e
    o destino anterior fica intacto."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def fetch_destination_problems(
    source_dir: Path, repository_root: Path, filenames: list[str]
) -> list[str]:
    problems = destination_problems(source_dir, repository_root)
    for filename in filenames:
        destination = source_dir / filename
        problems += destination_problems(destination, repository_root)
        if (repository_root / destination).is_dir():
            problems.append(f"{destination}: é um diretório")
    return list(dict.fromkeys(problems))


def download_from_hub(filename: str, staging_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            DATASET_REPO_ID,
            filename,
            repo_type="dataset",
            revision=OMNIFALL_REVISION,
            local_dir=staging_dir,
        )
    )


def fetch_sources(
    source_dir: Path,
    repository_root: Path,
    expected_sha256: dict[str, str] = EXPECTED_SHA256,
    download: Callable[[str, Path], Path] = download_from_hub,
) -> list[str]:
    """Baixa num diretório temporário fora do repositório, confere o SHA-256 e
    só então publica cada CSV no destino, revalidado logo antes da escrita."""
    filenames = list(expected_sha256)
    problems = fetch_destination_problems(source_dir, repository_root, filenames)
    if problems:
        return [f"destino: {problem}" for problem in problems]

    with tempfile.TemporaryDirectory(prefix="gatefall-cmdfall-") as staging:
        payloads: dict[str, bytes] = {}
        for filename, expected in expected_sha256.items():
            downloaded = download(filename, Path(staging))
            actual = sha256_file(downloaded)
            if actual != expected:
                problems.append(
                    f"fontes: {filename}: sha256 divergente (esperado {expected}, encontrado {actual})"
                )
            payloads[filename] = downloaded.read_bytes()
        if problems:
            return problems

        problems = fetch_destination_problems(source_dir, repository_root, filenames)
        if problems:
            return [f"destino: {problem}" for problem in problems]
        for filename, payload in payloads.items():
            publish_file(repository_root / source_dir / filename, payload)
            print(f"{source_dir / filename}: baixado de {DATASET_REPO_ID}@{OMNIFALL_REVISION}")
    return []


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
