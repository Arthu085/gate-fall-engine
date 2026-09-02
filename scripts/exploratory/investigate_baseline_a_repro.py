"""Comparação field-by-field entre runs/reference/le2i/baseline_a e runs/local/le2i/baseline_a
para diagnosticar a divergência de sensibilidade de evento no val (12/13 vs 13/13) descrita em
docs/tasks/8e-v1.md, seguindo o método do passo 1: diff de config.yaml e metrics.json.
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml

REFERENCE_DIR = Path("runs/reference/le2i/baseline_a")
LOCAL_DIR = Path("runs/local/le2i/baseline_a")

FIELDS_THAT_SETTLE_DIVERGENCE = (
    "seed",
    "epochs",
    "epochs_trained",
    "standardization_stats_sha256",
    "torch_version",
    "device",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(f"erro: {path} não contém um mapeamento YAML no topo", file=sys.stderr)
        sys.exit(1)
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"erro: {path} não contém um objeto JSON no topo", file=sys.stderr)
        sys.exit(1)
    return data


def diff_top_level_keys(
    reference: dict[str, Any], local: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    all_keys = sorted(set(reference) | set(local))
    diffs: list[tuple[str, Any, Any]] = []
    for key in all_keys:
        ref_value = reference.get(key, "<ausente>")
        local_value = local.get(key, "<ausente>")
        if ref_value != local_value:
            diffs.append((key, ref_value, local_value))
    return diffs


MAX_INLINE_VALUE_LENGTH = 200


def _format_value(value: Any) -> str:
    text = repr(value)
    if len(text) <= MAX_INLINE_VALUE_LENGTH:
        return text
    return f"{text[:MAX_INLINE_VALUE_LENGTH]}... (truncado, {len(text)} chars)"


def print_diff_report(title: str, diffs: list[tuple[str, Any, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not diffs:
        print("nenhuma diferença em chaves de topo")
        return
    for key, ref_value, local_value in diffs:
        print(f"{key}:")
        print(f"  reference = {_format_value(ref_value)}")
        print(f"  local     = {_format_value(local_value)}")


def find_sha256_or_checkpoint_keys(data: dict[str, Any]) -> list[str]:
    return sorted(k for k in data if "sha256" in k or "checkpoint" in k)


def print_settling_fields(
    reference_config: dict[str, Any],
    local_config: dict[str, Any],
    reference_metrics: dict[str, Any],
    local_metrics: dict[str, Any],
) -> bool:
    print("\n=== campos que, por regra do passo 1 da tarefa, encerram a investigação ===")
    merged_reference: dict[str, Any] = {**reference_config, **reference_metrics}
    merged_local: dict[str, Any] = {**local_config, **local_metrics}

    any_diff = False
    for field in FIELDS_THAT_SETTLE_DIVERGENCE:
        ref_value = merged_reference.get(field, "<ausente>")
        local_value = merged_local.get(field, "<ausente>")
        status = "DIVERGE" if ref_value != local_value else "igual"
        if ref_value != local_value:
            any_diff = True
        print(f"{field}: {status} (reference={ref_value!r}, local={local_value!r})")

    reference_hash_keys = set(find_sha256_or_checkpoint_keys(reference_metrics))
    local_hash_keys = set(find_sha256_or_checkpoint_keys(local_metrics))
    only_in_local = sorted(local_hash_keys - reference_hash_keys)
    only_in_reference = sorted(reference_hash_keys - local_hash_keys)
    print(f"\nchaves *sha256*/checkpoint presentes só em local: {only_in_local or 'nenhuma'}")
    print(
        f"chaves *sha256*/checkpoint presentes só em reference: {only_in_reference or 'nenhuma'}"
    )
    if only_in_local or only_in_reference:
        print(
            "nota: assimetria de chaves de hash é esperada — reference é anterior à mudança "
            "de código que passou a gravar config_sha256/checkpoint_sha256/"
            "training_metrics_sha256/alarm_protocol_sha256; não é, por si só, causa de "
            "divergência de métrica."
        )
    return any_diff


def print_verdict(settling_field_diverged: bool) -> None:
    print("\n=== veredito ===")
    if settling_field_diverged:
        print(
            "pelo menos um dos campos de metadado listados no passo 1 diverge entre "
            "reference e local. Pela regra de parada da própria tarefa, isso sozinho já "
            "explica a divergência de métricas (12/13 vs 13/13 de sensibilidade de evento "
            "no val) — não é necessário retreinar. Não prosseguir para o passo 2 (evaluate "
            "--force) nem para o passo 3 (train --force)."
        )
    else:
        print(
            "nenhum dos campos de metadado do passo 1 diverge. Prosseguir para o passo 2: "
            "rodar evaluate --force duas vezes contra o mesmo checkpoint para isolar "
            "não-determinismo de inferência."
        )


def main() -> None:
    reference_config = _load_yaml(REFERENCE_DIR / "config.yaml")
    local_config = _load_yaml(LOCAL_DIR / "config.yaml")
    reference_metrics = _load_json(REFERENCE_DIR / "metrics.json")
    local_metrics = _load_json(LOCAL_DIR / "metrics.json")

    config_diffs = diff_top_level_keys(reference_config, local_config)
    metrics_diffs = diff_top_level_keys(reference_metrics, local_metrics)

    print_diff_report("diff de config.yaml (chaves de topo)", config_diffs)
    print_diff_report("diff de metrics.json (chaves de topo)", metrics_diffs)

    settling_field_diverged = print_settling_fields(
        reference_config, local_config, reference_metrics, local_metrics
    )
    print_verdict(settling_field_diverged)


if __name__ == "__main__":
    main()
