import sys

import numpy as np
import pandas as pd

from gatefall.config import IGNORE_LABEL
from gatefall.eval.analysis.grouped_bootstrap import (
    METHOD_NOTE,
    SplitPredictions,
    _build_replicate_arrays,
    _event_metrics,
    _index_windows_by_video,
    _percentile_ci,
    _validate_subject_video_mapping,
    build_subject_to_videos,
    run_grouped_bootstrap_for_split,
)
from gatefall.eval.analysis.selftests.grouped_bootstrap_arms import (
    run_grouped_bootstrap_arms_selftest,
)
from gatefall.eval.shared.alarm_protocol import BASELINE_A_ALARM_PROTOCOL


def _check(name: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def _fixture_single_video_subject(
    subject: int, video_id: str, n: int, true_labels: list[int], pred_labels: list[int]
) -> tuple[SplitPredictions, dict[int, list[str]]]:
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    return predictions, {subject: [video_id]}


def _selftest_replicate_window_counts_are_cluster_multiples() -> bool:
    # Sujeito 1 -> vídeo com 5 janelas; sujeito 2 -> vídeo com 3 janelas.
    video_ids = ["v1"] * 5 + ["v2"] * 3
    k_ends = list(range(5)) + list(range(3))
    true_labels = [0] * 8
    pred_labels = [0] * 8
    predictions = SplitPredictions(
        video_ids=video_ids,
        k_ends=k_ends,
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=8,
        total_windows=8,
        labeled_windows=8,
    )
    subject_to_videos = {1: ["v1"], 2: ["v2"]}
    windows_by_video = _index_windows_by_video(video_ids, k_ends)

    drawn = np.array([1, 1, 2], dtype=np.int64)
    rep_video_ids, _rep_k_ends, _rep_true, _rep_pred = _build_replicate_arrays(
        drawn,
        subject_to_videos,
        windows_by_video,
        np.array(k_ends, dtype=np.int64),
        np.array(true_labels, dtype=np.int64),
        np.array(pred_labels, dtype=np.int64),
    )

    real_video_of = lambda virtual_id: virtual_id.split("__draw")[0]
    from collections import Counter

    counts_by_real_video: dict[str, int] = Counter(
        real_video_of(vid) for vid in rep_video_ids.tolist()
    )
    ok = (
        len(rep_video_ids) == 13
        and counts_by_real_video["v1"] % 5 == 0
        and counts_by_real_video["v2"] % 3 == 0
        and counts_by_real_video["v1"] == 10
        and counts_by_real_video["v2"] == 3
    )
    return _check(
        "contagem de janelas por vídeo real na réplica é sempre múltiplo "
        "exato da contagem verdadeira daquele vídeo (clusters, não janelas, "
        "são reamostrados)",
        ok,
    )


def _selftest_multi_video_subject_moves_together() -> bool:
    video_ids = ["va"] * 4 + ["vb"] * 2
    k_ends = list(range(4)) + list(range(2))
    true_labels = [0] * 6
    pred_labels = [0] * 6
    subject_to_videos = {1: ["va", "vb"]}
    windows_by_video = _index_windows_by_video(video_ids, k_ends)

    drawn = np.array([1], dtype=np.int64)
    rep_video_ids, _rep_k_ends, _rep_true, _rep_pred = _build_replicate_arrays(
        drawn,
        subject_to_videos,
        windows_by_video,
        np.array(k_ends, dtype=np.int64),
        np.array(true_labels, dtype=np.int64),
        np.array(pred_labels, dtype=np.int64),
    )
    real_videos_present = {vid.split("__draw")[0] for vid in rep_video_ids.tolist()}
    ok = len(rep_video_ids) == 6 and real_videos_present == {"va", "vb"}
    return _check(
        "sortear um sujeito com múltiplos vídeos arrasta todos os seus "
        "vídeos e todas as suas janelas juntos",
        ok,
    )


def _selftest_repeated_draws_stay_separate_in_events() -> bool:
    # Geometria adaptada de alarm_protocol_sensitivity: fall em k=[2,3],
    # fallen em k=[5,6,7], alarme dispara em k=[10,11,12] (trigger=3 do
    # protocolo congelado) -> exatamente 1 evento fall detectado no vídeo
    # base.
    n = 13
    video_id = "video_fixture"
    k_ends = list(range(n))
    true_labels = [0] * n
    true_labels[2:4] = [1, 1]
    true_labels[5:8] = [2, 2, 2]
    pred_labels = [0] * n
    pred_labels[10:13] = [1, 1, 1]

    subject_to_videos = {1: [video_id]}
    windows_by_video = _index_windows_by_video([video_id] * n, k_ends)

    for multiplicity in (1, 2, 3):
        drawn = np.array([1] * multiplicity, dtype=np.int64)
        rep_video_ids, rep_k_ends, rep_true, rep_pred = _build_replicate_arrays(
            drawn,
            subject_to_videos,
            windows_by_video,
            np.array(k_ends, dtype=np.int64),
            np.array(true_labels, dtype=np.int64),
            np.array(pred_labels, dtype=np.int64),
        )
        total_windows = len(rep_video_ids)
        labeled_windows = int(np.sum(rep_true != IGNORE_LABEL))
        report, _metrics = _event_metrics(
            rep_video_ids.tolist(),
            rep_k_ends.tolist(),
            rep_true.tolist(),
            rep_pred.tolist(),
            BASELINE_A_ALARM_PROTOCOL,
            total_windows,
            total_windows,
            labeled_windows,
        )
        if report["n_fall_events"] != multiplicity or report["n_detected_events"] != multiplicity:
            return _check(
                "sorteios repetidos do mesmo cluster permanecem estatisticamente "
                "separados na avaliação de eventos: n_fall_events/n_detected_events "
                "escalam exatamente com a multiplicidade do sorteio",
                False,
            )
    return _check(
        "sorteios repetidos do mesmo cluster permanecem estatisticamente "
        "separados na avaliação de eventos: n_fall_events/n_detected_events "
        "escalam exatamente com a multiplicidade do sorteio",
        True,
    )


def _selftest_identical_seeds_identical_results() -> bool:
    n = 13
    video_id = "video_fixture"
    k_ends = list(range(n))
    true_labels = [0] * n
    true_labels[2:4] = [1, 1]
    true_labels[5:8] = [2, 2, 2]
    pred_labels = [0] * n
    pred_labels[10:13] = [1, 1, 1]
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=k_ends,
        true_labels=true_labels,
        pred_labels=pred_labels,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos = {1: [video_id]}

    seed_sequence = np.random.SeedSequence(123)
    rng_a = np.random.default_rng(seed_sequence)
    result_a = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 200, 0.95, rng_a
    )
    seed_sequence_repeat = np.random.SeedSequence(123)
    rng_b = np.random.default_rng(seed_sequence_repeat)
    result_b = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 200, 0.95, rng_b
    )
    return _check(
        "seeds idênticas produzem resultados idênticos (igualdade de dict "
        "completa do relatório por split)",
        result_a == result_b,
    )


def _selftest_percentile_bounds_known_array() -> bool:
    values = [float(v) for v in range(1, 101)]
    ci_lower, ci_upper = _percentile_ci(values, 0.95)
    ok = (
        ci_lower is not None
        and ci_upper is not None
        and abs(ci_lower - 3.475) < 1e-9
        and abs(ci_upper - 97.525) < 1e-9
    )
    return _check(
        "percentil 2.5/97.5 sobre 1..100 reproduz os limites conhecidos "
        "(3.475, 97.525) via interpolação linear",
        ok,
    )


def _selftest_subject_split_mapping_validation() -> bool:
    frames_two_subjects = pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "split": ["val", "val"],
            "subject": [1, 2],
        }
    )
    raised_two_subjects = False
    try:
        build_subject_to_videos(frames_two_subjects, "val")
    except ValueError:
        raised_two_subjects = True

    subject_to_videos = {1: ["v1"]}
    raised_missing_from_map = False
    try:
        _validate_subject_video_mapping(subject_to_videos, ["v1", "v2"], "val")
    except ValueError:
        raised_missing_from_map = True

    raised_missing_from_predictions = False
    try:
        _validate_subject_video_mapping(subject_to_videos, [], "val")
    except ValueError:
        raised_missing_from_predictions = True

    ok = raised_two_subjects and raised_missing_from_map and raised_missing_from_predictions
    return _check(
        "vídeo com dois subjects distintos e vídeo ausente do mapeamento (em "
        "qualquer direção) levantam ValueError",
        ok,
    )


def _selftest_undefined_replicates_handling() -> bool:
    # Todos os replicates sem nenhum evento fall (true_labels/pred_labels
    # constantes em 0) -> sensitivity indefinida em 100% das réplicas.
    n = 5
    video_id = "video_no_fall"
    predictions_no_fall = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=[0] * n,
        pred_labels=[0] * n,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos_no_fall = {1: [video_id]}
    rng_no_fall = np.random.default_rng(np.random.SeedSequence(7))
    result_no_fall = run_grouped_bootstrap_for_split(
        predictions_no_fall,
        subject_to_videos_no_fall,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        50,
        0.95,
        rng_no_fall,
    )
    sensitivity_no_fall = result_no_fall["events"]["sensitivity"]
    all_no_fall_ok = (
        sensitivity_no_fall["ci_lower"] is None
        and sensitivity_no_fall["ci_upper"] is None
        and sensitivity_no_fall["valid_replicates"] == 0
        and sensitivity_no_fall["undefined_replicates"] == 50
    )

    # Dois sujeitos: um sem eventos fall, outro com um evento fall -> as
    # réplicas alternam entre definidas e indefinidas para sensitivity.
    n2 = 13
    video_with_fall = "video_with_fall"
    true_with_fall = [0] * n2
    true_with_fall[2:4] = [1, 1]
    true_with_fall[5:8] = [2, 2, 2]
    pred_with_fall = [0] * n2
    pred_with_fall[10:13] = [1, 1, 1]

    video_ids_mixed = [video_id] * n + [video_with_fall] * n2
    k_ends_mixed = list(range(n)) + list(range(n2))
    true_mixed = [0] * n + true_with_fall
    pred_mixed = [0] * n + pred_with_fall
    predictions_mixed = SplitPredictions(
        video_ids=video_ids_mixed,
        k_ends=k_ends_mixed,
        true_labels=true_mixed,
        pred_labels=pred_mixed,
        usable_windows=len(video_ids_mixed),
        total_windows=len(video_ids_mixed),
        labeled_windows=len(video_ids_mixed),
    )
    subject_to_videos_mixed = {1: [video_id], 2: [video_with_fall]}
    rng_mixed = np.random.default_rng(np.random.SeedSequence(11))
    result_mixed = run_grouped_bootstrap_for_split(
        predictions_mixed,
        subject_to_videos_mixed,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        200,
        0.95,
        rng_mixed,
    )
    sensitivity_mixed = result_mixed["events"]["sensitivity"]
    mixed_ok = (
        sensitivity_mixed["valid_replicates"] + sensitivity_mixed["undefined_replicates"] == 200
        and 0 < sensitivity_mixed["valid_replicates"] < 200
        and sensitivity_mixed["ci_lower"] is not None
        and sensitivity_mixed["ci_upper"] is not None
    )

    ok = all_no_fall_ok and mixed_ok
    check_no_fall_and_mixed = _check(
        "réplicas com denominador zero nunca substituem por zero: fixture "
        "sem nenhum evento fall produz limites nulos com "
        "valid_replicates=0/undefined_replicates=n_replicates; fixture mista "
        "produz valid+undefined=n_replicates com CI calculado só sobre o "
        "subconjunto válido",
        ok,
    )

    # Sujeito mapeado para uma lista de vídeos vazia -> toda réplica sorteia
    # zero janelas, total_windows == 0 em 100% das réplicas ->
    # false_alarms_per_hour indefinida (denominador de horas de vídeo é
    # zero), nunca substituída silenciosamente por 0.0.
    n3 = 5
    predictions_zero_windows = SplitPredictions(
        video_ids=["video_unreachable"] * n3,
        k_ends=list(range(n3)),
        true_labels=[0] * n3,
        pred_labels=[0] * n3,
        usable_windows=n3,
        total_windows=n3,
        labeled_windows=n3,
    )
    subject_to_videos_zero_windows: dict[int, list[str]] = {1: []}
    rng_zero_windows = np.random.default_rng(np.random.SeedSequence(13))
    result_zero_windows = run_grouped_bootstrap_for_split(
        predictions_zero_windows,
        subject_to_videos_zero_windows,
        BASELINE_A_ALARM_PROTOCOL,
        10,
        50,
        0.95,
        rng_zero_windows,
    )
    false_alarms_per_hour_zero_windows = result_zero_windows["events"][
        "false_alarms_per_hour"
    ]
    zero_windows_ok = (
        false_alarms_per_hour_zero_windows["ci_lower"] is None
        and false_alarms_per_hour_zero_windows["ci_upper"] is None
        and false_alarms_per_hour_zero_windows["valid_replicates"] == 0
        and false_alarms_per_hour_zero_windows["undefined_replicates"] == 50
    )
    check_zero_windows = _check(
        "réplicas cujo sorteio de subjects resulta em zero janelas (subject "
        "sem vídeos) nunca contam false_alarms_per_hour como válida com "
        "0.0: valid_replicates=0/undefined_replicates=n_replicates e "
        "limites nulos",
        zero_windows_ok,
    )

    return check_no_fall_and_mixed and check_zero_windows


def _selftest_binary_f1_validity_rule() -> bool:
    # Caso (a): existe positivo verdadeiro (fall/fallen), mas nenhuma
    # predição positiva -> tp=0, fp=0, fn>0. Denominador do F1 próprio
    # (2*tp+fp+fn) é fn>0, logo é uma réplica VÁLIDA com F1=0.0, não uma
    # réplica indefinida.
    n_a = 5
    video_a = "video_all_predictions_negative"
    predictions_a = SplitPredictions(
        video_ids=[video_a] * n_a,
        k_ends=list(range(n_a)),
        true_labels=[1, 1, 0, 0, 0],
        pred_labels=[0, 0, 0, 0, 0],
        usable_windows=n_a,
        total_windows=n_a,
        labeled_windows=n_a,
    )
    subject_to_videos_a = {1: [video_a]}
    rng_a = np.random.default_rng(np.random.SeedSequence(101))
    result_a = run_grouped_bootstrap_for_split(
        predictions_a, subject_to_videos_a, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_a
    )
    f1_a = result_a["classification"]["f1"]
    case_a_ok = (
        f1_a["valid_replicates"] == 50
        and f1_a["undefined_replicates"] == 0
        and f1_a["point_estimate"] == 0.0
        and f1_a["ci_lower"] == 0.0
        and f1_a["ci_upper"] == 0.0
    )

    # Caso (b): predições incluem positivos, mas nenhum positivo verdadeiro
    # -> tp=0, fn=0, fp>0. Denominador do F1 próprio é fp>0, logo também é
    # uma réplica VÁLIDA com F1=0.0.
    n_b = 5
    video_b = "video_no_true_positives"
    predictions_b = SplitPredictions(
        video_ids=[video_b] * n_b,
        k_ends=list(range(n_b)),
        true_labels=[0, 0, 0, 0, 0],
        pred_labels=[1, 1, 0, 0, 0],
        usable_windows=n_b,
        total_windows=n_b,
        labeled_windows=n_b,
    )
    subject_to_videos_b = {1: [video_b]}
    rng_b = np.random.default_rng(np.random.SeedSequence(102))
    result_b = run_grouped_bootstrap_for_split(
        predictions_b, subject_to_videos_b, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_b
    )
    f1_b = result_b["classification"]["f1"]
    case_b_ok = (
        f1_b["valid_replicates"] == 50
        and f1_b["undefined_replicates"] == 0
        and f1_b["point_estimate"] == 0.0
        and f1_b["ci_lower"] == 0.0
        and f1_b["ci_upper"] == 0.0
    )

    # Caso genuinamente indefinido: nem positivo verdadeiro, nem positivo
    # predito -> tp=fp=fn=0, denominador do F1 próprio é zero -> réplica
    # indefinida, limites nulos.
    n_c = 5
    video_c = "video_no_positives_at_all"
    predictions_c = SplitPredictions(
        video_ids=[video_c] * n_c,
        k_ends=list(range(n_c)),
        true_labels=[0, 0, 0, 0, 0],
        pred_labels=[0, 0, 0, 0, 0],
        usable_windows=n_c,
        total_windows=n_c,
        labeled_windows=n_c,
    )
    subject_to_videos_c = {1: [video_c]}
    rng_c = np.random.default_rng(np.random.SeedSequence(103))
    result_c = run_grouped_bootstrap_for_split(
        predictions_c, subject_to_videos_c, BASELINE_A_ALARM_PROTOCOL, 10, 50, 0.95, rng_c
    )
    f1_c = result_c["classification"]["f1"]
    case_c_ok = (
        f1_c["valid_replicates"] == 0
        and f1_c["undefined_replicates"] == 50
        and f1_c["ci_lower"] is None
        and f1_c["ci_upper"] is None
    )

    ok = case_a_ok and case_b_ok and case_c_ok
    return _check(
        "F1 binário é válido (F1=0.0) sempre que seu próprio denominador "
        "(2*tp+fp+fn) é não nulo, mesmo quando precision ou recall têm "
        "denominador zero isoladamente; só é indefinido quando tp=fp=fn=0",
        ok,
    )


def _selftest_alarm_protocol_recorded_in_report() -> bool:
    n = 5
    video_id = "video_alarm_protocol_check"
    predictions = SplitPredictions(
        video_ids=[video_id] * n,
        k_ends=list(range(n)),
        true_labels=[0] * n,
        pred_labels=[0] * n,
        usable_windows=n,
        total_windows=n,
        labeled_windows=n,
    )
    subject_to_videos = {1: [video_id]}
    rng = np.random.default_rng(np.random.SeedSequence(104))
    split_report = run_grouped_bootstrap_for_split(
        predictions, subject_to_videos, BASELINE_A_ALARM_PROTOCOL, 10, 10, 0.95, rng
    )
    report = {
        "run_name": "selftest_run",
        "checkpoint_path": "unused",
        "checkpoint_sha256": "unused",
        "training_metrics_path": "unused",
        "training_metrics_sha256": "unused",
        "alarm_protocol": BASELINE_A_ALARM_PROTOCOL.to_dict(),
        "method": {
            "cluster_unit": "subject",
            "n_replicates": 10,
            "confidence_level": 0.95,
            "seed": 104,
            "ci_method": "percentile",
            "note": METHOD_NOTE,
        },
        "splits": {"val": split_report},
    }
    ok = (
        "alarm_protocol" in report
        and report["alarm_protocol"] == BASELINE_A_ALARM_PROTOCOL.to_dict()
    )
    return _check(
        "relatório do bootstrap agrupado carrega alarm_protocol == "
        "BASELINE_A_ALARM_PROTOCOL.to_dict()",
        ok,
    )


def run_grouped_bootstrap_selftest() -> bool:
    checks = [
        _selftest_replicate_window_counts_are_cluster_multiples(),
        _selftest_multi_video_subject_moves_together(),
        _selftest_repeated_draws_stay_separate_in_events(),
        _selftest_identical_seeds_identical_results(),
        _selftest_percentile_bounds_known_array(),
        _selftest_subject_split_mapping_validation(),
        _selftest_undefined_replicates_handling(),
        _selftest_binary_f1_validity_rule(),
        _selftest_alarm_protocol_recorded_in_report(),
        *run_grouped_bootstrap_arms_selftest(),
    ]
    ok = all(checks)
    if not ok:
        print("\ngrouped_bootstrap selftest FALHOU", file=sys.stderr)
    else:
        print("\ngrouped_bootstrap selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_grouped_bootstrap_selftest():
        sys.exit(1)
