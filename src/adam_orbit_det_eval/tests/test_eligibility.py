"""Tests for the LOOO eligibility module."""

import numpy as np
import pytest

from adam_orbit_det_eval.looo.core import LOOOConfig
from adam_orbit_det_eval.looo.eligibility import (
    EligibilityResult,
    ExclusionStats,
    check_pair_eligibility,
    is_comet,
)


class TestIsComet:
    def test_comet_c_slash(self):
        assert is_comet("C/2020 F3") is True

    def test_comet_p_slash(self):
        assert is_comet("P/2021 A2") is True

    def test_asteroid_numbered(self):
        assert is_comet("433") is False

    def test_asteroid_provisional(self):
        assert is_comet("2020 AV2") is False


class TestExclusionStats:
    def test_record_eligible(self):
        stats = ExclusionStats()
        result = EligibilityResult(eligible=True, reason="eligible", stats={})
        stats.record(result)
        assert stats.total_checked == 1
        assert stats.total_eligible == 1
        assert stats.total_excluded == 0

    def test_record_excluded_reasons(self):
        stats = ExclusionStats()
        stats.record(EligibilityResult(eligible=False, reason="only 0 held-out obs", stats={}))
        stats.record(EligibilityResult(eligible=False, reason="only 3 remaining obs", stats={}))
        stats.record(EligibilityResult(eligible=False, reason="held-out fraction 0.90 > 0.80", stats={}))
        stats.record(EligibilityResult(eligible=False, reason="arc length 2.0d < 7.0d", stats={}))
        stats.record(EligibilityResult(eligible=False, reason="comet excluded", stats={}))
        assert stats.total_checked == 5
        assert stats.excluded_min_obs_held_out == 1
        assert stats.excluded_min_obs_remaining == 1
        assert stats.excluded_max_held_out_fraction == 1
        assert stats.excluded_min_arc_length == 1
        assert stats.excluded_comet == 1

    def test_summary(self):
        stats = ExclusionStats()
        stats.record(EligibilityResult(eligible=True, reason="eligible", stats={}))
        stats.record(EligibilityResult(eligible=False, reason="only 3 remaining obs", stats={}))
        summary = stats.summary()
        assert summary["total_checked"] == 2
        assert summary["total_eligible"] == 1
        assert summary["total_excluded"] == 1
