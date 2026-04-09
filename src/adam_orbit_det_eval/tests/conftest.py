from collections import defaultdict

import numpy as np
import pytest

# Storage for quality tracker
_quality_results = defaultdict(list)  # { metric_name: [{"test": ..., "actual": ...}] }


class QualityTracker:
    """Collects quality comparisons during a test"""

    def __init__(self, node_name: str):
        self._node_name = node_name

    def check(
        self,
        name: str,
        actual,
        baseline,
    ):
        """
        Record how far *actual* is from *baseline*.

        Parameters
        ----------
        name      : human-readable metric label
        actual    : scalar from the fitter under test
        baseline  : scalar from MPC
        """
        delta = np.abs(actual - baseline)
        with np.errstate(divide="ignore", invalid="ignore"):
            percent = np.where(baseline != 0, 100.0 * delta / np.abs(baseline), np.nan)
        entry = {
            "test": self._node_name,
            "metric": name,
            "actual": actual,
            "baseline": baseline,
            "delta": delta,
            "percent": percent,
        }
        _quality_results[self._node_name].append(entry)
        return entry  # caller can still assert on it if desired


@pytest.fixture
def quality_tracker(request):
    tracker = QualityTracker(request.node.name)
    yield tracker
    # entries are already pushed to _quality_results inside .check()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Terminal hook reporter for quality tracker"""
    if not _quality_results:
        return

    tr = terminalreporter
    tr.write_sep("=", "quality tracker")

    col_test = max(len(e) for e in _quality_results.keys())
    col_metric = max(len(row["metric"]) for e in _quality_results.values() for row in e)
    col_test = max(col_test, 4)  # "Test"
    col_metric = max(col_metric, 5)  # "Param"
    num_w = 12  # width of each numeric column

    header = (
        f"{'Test':<{col_test}}  "
        f"{'Param':<{col_metric}}  "
        f"{'Fitted':>{num_w}}  "
        f"{'MPC':>{num_w}}  "
        f"{'Delta':>{num_w}}  "
        f"{'Percent delta':>{num_w}}"
    )
    tr.write_line(header)
    tr.write_line("-" * len(header))

    def fmt(v):
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    for test in _quality_results.values():
        first = True
        for entry in test:
            line = f"{entry['test']:<{col_test}}  " if first else " " * (col_test + 2)
            line += (
                f"{entry['metric']:<{col_metric}}  "
                f"{fmt(entry['actual']):>{num_w}}  "
                f"{fmt(entry['baseline']):>{num_w}}  "
                f"{fmt(entry['delta']):>{num_w}}  "
                f"{fmt(entry['percent']):>{num_w}}"
            )
            first = False
            tr.write_line(line)
