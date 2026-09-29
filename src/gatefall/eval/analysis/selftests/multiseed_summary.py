import sys

from gatefall.eval.analysis.selftests.aggregation import (
    _selftest_aggregate_stats_known_array,
    _selftest_binary_fall_fallen_matches_independent_recomputation,
    _selftest_seed_blocks_round_trip_and_new_aggregates_exist,
    _selftest_two_valid_seed_runs_aggregate,
)
from gatefall.eval.analysis.selftests.guards import (
    _selftest_arm_and_config_guards,
    _selftest_duplicate_seed_raises,
    _selftest_fewer_than_min_seeds_raises,
    _selftest_malformed_run_dir_raises,
    _selftest_missing_diagnostics_raises,
    _selftest_non_seed_config_divergence_raises,
)
from gatefall.eval.analysis.selftests.multi_arm import _selftest_all_arms
from gatefall.eval.analysis.selftests.outputs import (
    _selftest_arm_a_output_compatibility,
    _selftest_csv_row_inventory_matches_schema,
    _selftest_writer_honors_force,
)


def run_multiseed_summary_selftest() -> bool:
    checks = [
        _selftest_aggregate_stats_known_array(),
        _selftest_two_valid_seed_runs_aggregate(),
        _selftest_seed_blocks_round_trip_and_new_aggregates_exist(),
        _selftest_binary_fall_fallen_matches_independent_recomputation(),
        _selftest_missing_diagnostics_raises(),
        _selftest_csv_row_inventory_matches_schema(),
        _selftest_duplicate_seed_raises(),
        _selftest_non_seed_config_divergence_raises(),
        _selftest_malformed_run_dir_raises(),
        _selftest_fewer_than_min_seeds_raises(),
        _selftest_writer_honors_force(),
        _selftest_all_arms(),
        _selftest_arm_and_config_guards(),
        _selftest_arm_a_output_compatibility(),
    ]
    ok = all(checks)
    if not ok:
        print("\nmultiseed_summary selftest FALHOU", file=sys.stderr)
    else:
        print("\nmultiseed_summary selftest OK: todas as checagens passaram")
    return ok


def run_selftest() -> None:
    if not run_multiseed_summary_selftest():
        sys.exit(1)
