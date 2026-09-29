"""Checagens sintéticas da compatibilidade do runtime isolado SAM 3."""

import re
import tempfile
import tomllib
from pathlib import Path

from gatefall.sam3 import runtime
from gatefall.sam3.selftests.fixtures import _check


_SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION = (82, 0, 0)


def _parse_leading_version_tuple(text: str) -> tuple[int, ...] | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _parse_whole_version_tuple(text: str) -> tuple[int, ...] | None:
    # O texto da versão precisa ocupar o restante inteiro da cláusula (só
    # espaços são tolerados nas bordas): um sufixo não numérico como
    # `<82junk` ou `<=81.5.2rc1` é entrada malformada e reprova.
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)*)\s*", text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _pad_version_tuples(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = max(len(left), len(right))
    return (
        left + (0,) * (length - len(left)),
        right + (0,) * (length - len(right)),
    )


def _version_lt(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    padded_left, padded_right = _pad_version_tuples(left, right)
    return padded_left < padded_right


def _version_le(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    padded_left, padded_right = _pad_version_tuples(left, right)
    return padded_left <= padded_right


def _split_leading_distribution_name(clause: str) -> tuple[str, str]:
    # O PEP 508 só anexa o nome da distribuição à primeira cláusula separada
    # por vírgula; separamos esse prefixo (e extras opcionais, ex.: `[foo]`)
    # do restante para identificar a distribuição e testar o operador em
    # qualquer posição da cláusula.
    if clause[:1] in "<>=!~":
        return "", clause
    match = re.match(r"^([A-Za-z0-9_.-]+)(?:\s*\[[^\]]*\])?\s*", clause)
    if match is None:
        return "", clause
    return match.group(1), clause[match.end():]


def _normalize_distribution_name(name: str) -> str:
    # PEP 503: nomes de distribuição são comparados em minúsculas com
    # sequências de `-`, `_` e `.` colapsadas em um único `-`.
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
    pyproject_path: Path,
) -> bool:
    try:
        with pyproject_path.open("rb") as pyproject_file:
            pyproject_data = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    tool_table = pyproject_data.get("tool", {})
    if not isinstance(tool_table, dict):
        return False
    uv_table = tool_table.get("uv", {})
    if not isinstance(uv_table, dict):
        return False
    constraints = uv_table.get("constraint-dependencies", [])
    if not isinstance(constraints, list):
        return False
    for constraint in constraints:
        if not isinstance(constraint, str):
            continue
        if ";" in constraint:
            # Um marcador de ambiente torna o teto condicional: no ambiente em
            # que o marcador é falso não há teto e o setuptools pode resolver
            # >= 82, removendo pkg_resources.
            continue
        clauses = [part.strip() for part in constraint.split(",")]
        if not clauses:
            continue
        distribution_name, remainder = _split_leading_distribution_name(clauses[0])
        if _normalize_distribution_name(distribution_name) != "setuptools":
            continue
        clauses[0] = remainder
        for clause in clauses:
            if clause.startswith("<="):
                version = _parse_whole_version_tuple(clause[2:])
                if version is not None and _version_lt(
                    version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION
                ):
                    return True
            elif clause.startswith("<"):
                version = _parse_whole_version_tuple(clause[1:])
                if version is not None and _version_le(
                    version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION
                ):
                    return True
    return False


def _lock_resolves_setuptools_below_pkg_resources_removal(lock_path: Path) -> bool:
    try:
        with lock_path.open("rb") as lock_file:
            lock_data = tomllib.load(lock_file)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    packages = lock_data.get("package", [])
    if not isinstance(packages, list):
        return False
    # `uv lock` pode gerar mais de um stanza `setuptools` quando a resolução
    # se bifurca por `resolution-markers` (versões de Python/plataformas
    # diferentes); avaliar só o primeiro stanza encontrado deixaria passar um
    # lock em que um stanza fica abaixo do teto e outro acima.
    found_setuptools_stanza = False
    for package in packages:
        if not isinstance(package, dict):
            continue
        if package.get("name") != "setuptools":
            continue
        found_setuptools_stanza = True
        version = _parse_leading_version_tuple(str(package.get("version", "")))
        if version is None:
            return False
        if not _version_lt(version, _SETUPTOOLS_PKG_RESOURCES_REMOVAL_VERSION):
            return False
    return found_setuptools_stanza


def _write_pyproject_with_constraint(root: Path, constraint: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text(
        "[tool.uv]\n"
        f'constraint-dependencies = ["{constraint}"]\n'
    )
    return path


def _write_lock_with_setuptools_version(root: Path, version: str) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{version}"\n'
    )
    return path


def _write_lock_with_two_setuptools_versions(
    root: Path, first_version: str, second_version: str
) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{first_version}"\n'
        "[[package]]\n"
        'name = "setuptools"\n'
        f'version = "{second_version}"\n'
    )
    return path


def _write_lock_with_no_setuptools_stanza(root: Path) -> Path:
    path = root / "uv.lock"
    path.write_text(
        "[[package]]\n"
        'name = "torch"\n'
        'version = "2.0.0"\n'
    )
    return path


def _check_setuptools_ceiling_matches_distribution_name_not_substring() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        lookalikes_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools-scm<82",
                "setuptools_scm<82",
                "setuptools.scm<82",
                "mysetuptools<2",
                "setuptools--scm<82",
                "setuptools__scm<82",
                "setuptools-_.scm<82",
            )
        )
        # Regressão do bug de indexação de cláusula: só a cláusula 0 carrega o
        # nome da distribuição (PEP 508); um token de outra distribuição nas
        # cláusulas seguintes não pode ser lido como teto do setuptools.
        clause_indexing_false_positives_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools>=200,foopkg<82",
                "setuptools>=77.0.3,otherpkg<82",
            )
        )
        positives_accepted = all(
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools>=77.0.3,<82",
                "setuptools<82,>=77.0.3",
                "setuptools<82",
                "setuptools >= 77.0.3 , < 82",
                "setuptools[foo]<82",
                "SetupTools<82",
                "SETUPTOOLS<82",
                "  setuptools<82  ",
                "  setuptools>=77.0.3,<82  ",
            )
        )
        unsafe_ceiling_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools>=77.0.3,<=82")
            )
        )
        absent_ceiling_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools>=77.0.3")
            )
        )
        direct_reference_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(
                    root, "setuptools @ https://example.invalid/setuptools.whl"
                )
            )
        )
        no_version_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "setuptools[foo]")
            )
        )
        empty_string_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, "")
            )
        )
        non_string_entry_path = root / "pyproject.toml"
        non_string_entry_path.write_text(
            "[tool.uv]\nconstraint-dependencies = [1]\n"
        )
        non_string_entry_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                non_string_entry_path
            )
        )
        lookalikes_only_list_path = root / "pyproject.toml"
        lookalikes_only_list_path.write_text(
            "[tool.uv]\n"
            'constraint-dependencies = ["setuptools-scm<82", "mysetuptools<2"]\n'
        )
        lookalikes_only_list_rejected = not (
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                lookalikes_only_list_path
            )
        )
    ok = (
        lookalikes_rejected
        and clause_indexing_false_positives_rejected
        and positives_accepted
        and unsafe_ceiling_rejected
        and absent_ceiling_rejected
        and direct_reference_rejected
        and no_version_rejected
        and empty_string_rejected
        and non_string_entry_rejected
        and lookalikes_only_list_rejected
    )
    return _check(
        "_pyproject_declares_setuptools_ceiling_below_pkg_resources_removal: "
        "identifica a distribuição pelo nome normalizado (PEP 503), não por "
        "substring — nomes parecidos (setuptools-scm, setuptools_scm, "
        "setuptools.scm, mysetuptools, e variantes com separadores repetidos "
        "como setuptools--scm/setuptools__scm/setuptools-_.scm) não "
        "satisfazem o teto, uma cláusula 2+ com nome de outra distribuição "
        "(setuptools>=200,foopkg<82 / setuptools>=77.0.3,otherpkg<82) não é "
        "lida como teto do setuptools, as variantes reais (reordenada, "
        "cláusula única, com espaços, com extras, com maiúsculas/minúsculas "
        "trocadas, com espaços nas bordas) continuam passando, e um teto "
        "ausente, <=82, uma referência direta (`@ url`), extras sem versão, "
        "string vazia, entrada não textual e uma lista só com nomes "
        "parecidos continuam reprovando", ok
    )


def _check_setuptools_ceiling_rejects_environment_markers_and_trailing_garbage() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        # Um marcador de ambiente torna o teto condicional: no ambiente em que
        # o marcador é falso não há teto algum e o setuptools pode subir para
        # >=82, removendo pkg_resources e quebrando o worker isolado.
        markers_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools<82; python_version<'3.12'",
                "setuptools<82; sys_platform=='win32'",
                # O marcador viaja na última cláusula de um conjunto múltiplo.
                "setuptools>=77.0.3,<82; python_version<'3.12'",
            )
        )
        trailing_garbage_rejected = all(
            not _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                "setuptools<82junk",
                "setuptools<=81.5.2rc1",
            )
        )
        well_formed_accepted = all(
            _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
                _write_pyproject_with_constraint(root, constraint)
            )
            for constraint in (
                # Restrição literal de sam3_runtime/pyproject.toml.
                "setuptools>=77.0.3,<82",
                "setuptools<82",
                "setuptools <= 81",
            )
        )
    ok = markers_rejected and trailing_garbage_rejected and well_formed_accepted
    return _check(
        "_pyproject_declares_setuptools_ceiling_below_pkg_resources_removal: "
        "o texto da versão precisa ser o restante inteiro da cláusula "
        "(apenas espaços à direita) — marcadores de ambiente "
        "(`; python_version<'3.12'`, `; sys_platform=='win32'`, inclusive "
        "sobre a última cláusula de setuptools>=77.0.3,<82) tornam o teto "
        "condicional e reprovam, sufixo não numérico reprova nos dois "
        "operadores (<82junk, <=81.5.2rc1), e as formas bem formadas "
        "(setuptools>=77.0.3,<82, setuptools<82, setuptools <= 81) "
        "continuam passando", ok
    )


def _check_setuptools_lock_version_boundary_and_corruption() -> bool:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        version_82_rejected = not _lock_resolves_setuptools_below_pkg_resources_removal(
            _write_lock_with_setuptools_version(root, "82.0.0")
        )
        version_81_accepted = _lock_resolves_setuptools_below_pkg_resources_removal(
            _write_lock_with_setuptools_version(root, "81.0.0")
        )
        corrupted_lock_path = root / "uv.lock"
        corrupted_lock_path.write_text('package = "oops"\n')
        corrupted_rejected = not _lock_resolves_setuptools_below_pkg_resources_removal(
            corrupted_lock_path
        )
        no_setuptools_stanza_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_no_setuptools_stanza(root)
            )
        )
        # Regressão do bug de "primeiro stanza só": `uv lock` pode gerar mais
        # de um stanza `setuptools` quando a resolução se bifurca por
        # `resolution-markers` (Python/plataforma diferentes). Um stanza
        # abaixo do teto e outro acima deve reprovar, em qualquer ordem.
        two_stanzas_low_then_high_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_two_setuptools_versions(root, "81.0.0", "84.0.0")
            )
        )
        two_stanzas_high_then_low_rejected = (
            not _lock_resolves_setuptools_below_pkg_resources_removal(
                _write_lock_with_two_setuptools_versions(root, "84.0.0", "81.0.0")
            )
        )
    ok = (
        version_82_rejected
        and version_81_accepted
        and corrupted_rejected
        and no_setuptools_stanza_rejected
        and two_stanzas_low_then_high_rejected
        and two_stanzas_high_then_low_rejected
    )
    return _check(
        "_lock_resolves_setuptools_below_pkg_resources_removal: versão "
        "82.0.0 reprova, 81.0.0 passa, lock corrompido (`package` fora do "
        "formato de lista de tabelas) reprova sem levantar exceção, lock sem "
        "nenhum stanza setuptools reprova, e dois stanzas setuptools "
        "(bifurcação por resolution-markers) com um deles >= 82 reprovam em "
        "qualquer ordem — não basta o primeiro stanza encontrado passar", ok
    )


def _check_normalize_distribution_name_collapses_separator_runs() -> bool:
    # PEP 503 colapsa QUALQUER sequência de `-`, `_` e `.` em um único `-`;
    # esta asserção fixa esse comportamento (e não apenas separadores
    # isolados), pegando uma regressão de `[-_.]+` para `[-_.]`.
    ok = (
        _normalize_distribution_name("setuptools__scm")
        == _normalize_distribution_name("setuptools-scm")
        == _normalize_distribution_name("setuptools-_.scm")
        == "setuptools-scm"
    )
    return _check(
        "_normalize_distribution_name: sequências de separadores repetidos "
        "ou mistos (setuptools__scm, setuptools-_.scm) colapsam para o "
        "mesmo nome normalizado que setuptools-scm", ok
    )


def _check_sam3_runtime_lock_pins_setuptools_below_pkg_resources_removal() -> bool:
    root = Path(__file__).resolve().parents[4]
    pyproject_path = root / "sam3_runtime" / "pyproject.toml"
    lock_path = root / "sam3_runtime" / "uv.lock"

    pyproject_ok = _pyproject_declares_setuptools_ceiling_below_pkg_resources_removal(
        pyproject_path
    )
    lock_ok = _lock_resolves_setuptools_below_pkg_resources_removal(lock_path)

    return _check(
        "sam3_runtime: uv.lock resolve setuptools abaixo de 82 (pkg_resources "
        "ainda existe) e sam3_runtime/pyproject.toml declara esse teto em "
        "[tool.uv].constraint-dependencies",
        pyproject_ok and lock_ok,
    )


def _check_sam3_runtime_worker_declares_same_inference_autocast_policy() -> bool:
    worker_path = Path(__file__).resolve().parents[4] / "sam3_runtime" / "run_sam3.py"
    try:
        worker_source = worker_path.read_text(encoding="utf-8")
    except OSError:
        # Fail-closed: sem o arquivo do worker não há o que comparar.
        return _check(
            "sam3_runtime/run_sam3.py: worker ilegível ou ausente", False
        )

    # Fragmentos entre aspas: `float16` é substring de `bfloat16`, então a
    # forma nua não distinguiria um worker que perdeu o recuo FP16.
    required_fragments = (
        "sam3_inference_autocast_dtype",
        *(f'"{name}"' for name in runtime.SAM3_INFERENCE_AUTOCAST_DTYPE_NAMES),
        "torch.autocast(",
        "including_emulation=False",
    )
    forbidden_fragments = (".half()", "model.to(torch.bfloat16)", "set_default_dtype")

    required_ok = all(fragment in worker_source for fragment in required_fragments)
    forbidden_ok = all(
        fragment not in worker_source for fragment in forbidden_fragments
    )

    ok = required_ok and forbidden_ok
    return _check(
        "sam3_runtime/run_sam3.py: o worker isolado declara "
        "sam3_inference_autocast_dtype no manifesto, usa torch.autocast com "
        "is_bf16_supported(including_emulation=False) e não converte pesos "
        "globalmente (.half(), model.to(torch.bfloat16), set_default_dtype) — "
        "é uma checagem textual entre dois lados deliberadamente duplicados "
        "(o worker nunca é importado pelo gatefall), não uma prova de "
        "equivalência semântica com select_inference_autocast_dtype_name", ok
    )
