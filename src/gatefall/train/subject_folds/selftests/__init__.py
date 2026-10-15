from gatefall.train.subject_folds.selftests.folds import run_folds_selftest
from gatefall.train.subject_folds.selftests.training import run_training_selftest


def run_selftest() -> bool:
    results = [run_folds_selftest(), run_training_selftest()]
    return all(results)
