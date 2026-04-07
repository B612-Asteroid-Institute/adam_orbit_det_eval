"""
Observatory mapping for synthetic observation generation.

``FakeObservatory`` pairs a fake MPC station code (used in synthetic output)
with a real MPC code (used for physical site coordinates / parallax).  It
carries the injected noise sigmas and composable bias models.

``ObservatoryMap`` manages a list of ``FakeObservatory`` assignments and
provides lookup utilities.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .bias_models import BiasModel, CompoundBias

logger = logging.getLogger(__name__)


@dataclass
class FakeObservatory:
    """
    Describes a single synthetic observatory in a simulation scenario.

    Parameters
    ----------
    fake_code : str
        Synthetic MPC station code used in output observations (e.g. "AA01").
    real_code : str
        Real MPC station code whose physical site coordinates and parallax
        constants are used for ephemeris generation.
    noise_sigma_ra : float
        1-sigma Gaussian noise in arcseconds applied to RA*cos(dec).
    noise_sigma_dec : float
        1-sigma Gaussian noise in arcseconds applied to Dec.
    biases : list of BiasModel
        Ordered list of bias models to apply.  An empty list means the station
        is a clean reference (noise only, no systematic bias).
    astcat : str
        Astrometric catalog code reported in the synthetic ADES output.
    """

    fake_code: str
    real_code: str
    noise_sigma_ra: float
    noise_sigma_dec: float
    biases: List[BiasModel] = field(default_factory=list)
    astcat: str = "Gaia3E"

    @property
    def compound_bias(self) -> BiasModel:
        """Return a single BiasModel that is the sum of all assigned biases."""
        if not self.biases:
            return _ZeroBias()
        if len(self.biases) == 1:
            return self.biases[0]
        return CompoundBias(self.biases)

    @property
    def bias_type(self) -> str:
        """Human-readable bias type string for truth table."""
        if not self.biases:
            return "None"
        names = [b.bias_type for b in self.biases]
        return "+".join(names)

    def bias_params_dict(self) -> dict:
        """JSON-serialisable dict of bias parameters for truth_biases.csv."""
        if not self.biases:
            return {}
        if len(self.biases) == 1:
            return self.biases[0].params_dict()
        return {"biases": [{"type": b.bias_type, "params": b.params_dict()} for b in self.biases]}


class _ZeroBias(BiasModel):
    """Internal no-op bias used when biases list is empty."""

    def apply(self, ra, dec, obstime, mag=None, zenith_angle=None,
              parallactic_angle=None, object_rate=None, field_ra=None,
              field_dec=None, velocity_ra_unit=None, velocity_dec_unit=None):
        import numpy as np
        return np.zeros_like(ra), np.zeros_like(dec)

    def params_dict(self) -> dict:
        return {}


class ObservatoryMap:
    """
    Manages the mapping from real MPC codes to fake observatory assignments.

    Parameters
    ----------
    assignments : list of FakeObservatory
        All fake observatories in this simulation scenario.  Each assignment
        maps one real code to one fake code (many-to-one is not supported;
        each real code should appear at most once).
    """

    def __init__(self, assignments: List[FakeObservatory]) -> None:
        self.assignments = assignments
        self._by_fake: Dict[str, FakeObservatory] = {a.fake_code: a for a in assignments}
        self._by_real: Dict[str, FakeObservatory] = {}
        for a in assignments:
            if a.real_code in self._by_real:
                logger.warning(
                    f"Real code {a.real_code!r} assigned to multiple fake codes "
                    f"({self._by_real[a.real_code].fake_code!r} and {a.fake_code!r}). "
                    "Only the last assignment will be used for real→fake lookup."
                )
            self._by_real[a.real_code] = a

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def fake_for_real(self, real_code: str) -> Optional[FakeObservatory]:
        """Return the FakeObservatory assigned to *real_code*, or None."""
        return self._by_real.get(real_code)

    def real_for_fake(self, fake_code: str) -> str:
        """Return the real MPC code for a given fake code. Raises KeyError if missing."""
        return self._by_fake[fake_code].real_code

    def get_by_fake(self, fake_code: str) -> FakeObservatory:
        """Return the FakeObservatory for *fake_code*. Raises KeyError if missing."""
        return self._by_fake[fake_code]

    def all_real_codes(self) -> List[str]:
        """Sorted list of all real MPC codes in this map."""
        return sorted(self._by_real.keys())

    def all_fake_codes(self) -> List[str]:
        """Sorted list of all fake codes in this map."""
        return sorted(self._by_fake.keys())

    # ------------------------------------------------------------------
    # Truth table
    # ------------------------------------------------------------------

    def truth_table(
        self,
        empirical_means: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per fake observatory.

        Columns
        -------
        fake_code, real_code, astcat, noise_sigma_ra_arcsec,
        noise_sigma_dec_arcsec, bias_type, bias_params_json,
        expected_mean_ra_arcsec, expected_mean_dec_arcsec

        Parameters
        ----------
        empirical_means : dict, optional
            Maps ``real_code -> (mean_ra_arcsec, mean_dec_arcsec)``.  When
            provided, overrides the analytically-derived expected mean for any
            station whose real code appears in the dict.  Pass this for
            sample-dependent biases (``TrailingBias``, ``DCRBias``, etc.)
            whose mean cannot be predicted without knowing the actual
            observation geometry.
        """
        rows = []
        for a in self.assignments:
            if empirical_means is not None and a.real_code in empirical_means:
                exp_ra, exp_dec = empirical_means[a.real_code]
            else:
                exp_ra = _expected_mean_ra(a)
                exp_dec = _expected_mean_dec(a)
            rows.append(
                {
                    "fake_code": a.fake_code,
                    "real_code": a.real_code,
                    "astcat": a.astcat,
                    "noise_sigma_ra_arcsec": a.noise_sigma_ra,
                    "noise_sigma_dec_arcsec": a.noise_sigma_dec,
                    "bias_type": a.bias_type,
                    "bias_params_json": json.dumps(a.bias_params_dict()),
                    "expected_mean_ra_arcsec": exp_ra,
                    "expected_mean_dec_arcsec": exp_dec,
                }
            )
        return pd.DataFrame(rows)


def _expected_mean_ra(fobs: FakeObservatory) -> float:
    """
    Compute the expected mean RA bias that LOOO should recover for this station.

    For ConstantBias this is simply delta_ra.  For time-varying biases the
    mean over a typical observation arc is 0 or near 0.  We use a simple
    heuristic: for ConstantBias-like models return their delta_ra; for all
    others return 0 (the conservative / worst-case expectation for detection).
    """
    from .bias_models import (
        ConstantBias, CatalogEpochBias, WrongSiteBias,
        CTEBias, PlateScaleBias, FieldRotationBias,
    )
    total = 0.0
    for b in fobs.biases:
        if isinstance(b, ConstantBias):
            total += b.delta_ra
        elif isinstance(b, CatalogEpochBias):
            total += b.pm_ra_median_masyr * b.epoch_error_years / 1000.0
        elif isinstance(b, WrongSiteBias):
            import numpy as np
            dist_m = b.nominal_distance_au * 1.495978707e11
            total += float(np.degrees(b.coord_error_east_m / dist_m)) * 3600.0
        elif isinstance(b, CTEBias):
            import numpy as np
            total += b.amplitude_arcsec * np.sin(np.deg2rad(b.readout_direction_deg))
        # For SeasonalBias, TimingBias, etc. the mean over a long arc is ~0
    return total


def _expected_mean_dec(fobs: FakeObservatory) -> float:
    """Expected mean Dec bias that LOOO should recover."""
    from .bias_models import (
        ConstantBias, CatalogEpochBias, WrongSiteBias,
        CTEBias,
    )
    total = 0.0
    for b in fobs.biases:
        if isinstance(b, ConstantBias):
            total += b.delta_dec
        elif isinstance(b, CatalogEpochBias):
            total += b.pm_dec_median_masyr * b.epoch_error_years / 1000.0
        elif isinstance(b, WrongSiteBias):
            import numpy as np
            dist_m = b.nominal_distance_au * 1.495978707e11
            total += float(np.degrees(b.coord_error_north_m / dist_m)) * 3600.0
        elif isinstance(b, CTEBias):
            import numpy as np
            total += b.amplitude_arcsec * np.cos(np.deg2rad(b.readout_direction_deg))
    return total
