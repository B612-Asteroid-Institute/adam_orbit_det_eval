"""
Gaussian noise model for synthetic observation generation.

``NoiseModel`` provides per-(station, catalog) noise sigma lookup and
vectorised Gaussian draws.  It can be initialised with hard-coded defaults
or loaded from an empirical sigma table produced by ``03_analyze.py``.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default fallback sigma (arcsec) when a station is not in any table
_DEFAULT_SIGMA = 0.5


class NoiseModel:
    """
    Per-observation Gaussian noise model.

    Supports loading empirical per-(stn, astcat) sigmas from the LOOO
    ``sigma_table.csv`` produced by ``03_analyze.py``.  If a (stn, astcat)
    pair is not found, falls back to the default sigma provided at
    construction time.

    Parameters
    ----------
    default_sigma_ra : float
        Fallback 1-sigma noise in arcseconds for RA*cos(dec).
    default_sigma_dec : float
        Fallback 1-sigma noise in arcseconds for Dec.
    """

    def __init__(
        self,
        default_sigma_ra: float = _DEFAULT_SIGMA,
        default_sigma_dec: float = _DEFAULT_SIGMA,
    ) -> None:
        self.default_sigma_ra = float(default_sigma_ra)
        self.default_sigma_dec = float(default_sigma_dec)
        # Dict[(stn, astcat)] -> (sigma_ra, sigma_dec)
        self._table: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_sigma_table(cls, sigma_table_csv: Path) -> "NoiseModel":
        """
        Load empirical per-(stn, astcat) sigmas from a sigma_table CSV.

        The CSV is expected to have at least the columns:
          stn, astcat, rms_ra_arcsec, rms_dec_arcsec

        These are typically produced by running ``03_analyze.py`` with
        ``--object-weighted`` on the full LOOO run.

        Parameters
        ----------
        sigma_table_csv : Path
            Path to the sigma table CSV file.

        Returns
        -------
        NoiseModel
        """
        df = pd.read_csv(sigma_table_csv)

        required = {"stn", "astcat", "rms_ra_arcsec", "rms_dec_arcsec"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"sigma_table_csv is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        model = cls()
        for _, row in df.iterrows():
            stn = str(row["stn"])
            astcat = str(row["astcat"]) if pd.notna(row["astcat"]) else ""
            sigma_ra = float(row["rms_ra_arcsec"])
            sigma_dec = float(row["rms_dec_arcsec"])
            if np.isfinite(sigma_ra) and np.isfinite(sigma_dec):
                model._table[(stn, astcat)] = (sigma_ra, sigma_dec)

        logger.info(
            f"NoiseModel loaded {len(model._table)} (stn, astcat) sigma entries "
            f"from {sigma_table_csv}"
        )
        return model

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_sigma(
        self, stn: str, astcat: str = "Gaia3E"
    ) -> Tuple[float, float]:
        """
        Return (sigma_ra_arcsec, sigma_dec_arcsec) for this station/catalog pair.

        Lookup order:
          1. Exact (stn, astcat) match in the loaded table.
          2. (stn, "") — stn-only entry (if the CSV has no catalog column).
          3. Default sigma values supplied at construction.

        Parameters
        ----------
        stn : str
            MPC station code (real or fake).
        astcat : str
            Astrometric catalog code.

        Returns
        -------
        (sigma_ra_arcsec, sigma_dec_arcsec)
        """
        key = (stn, astcat)
        if key in self._table:
            return self._table[key]
        # Try stn-only fallback
        key_stn = (stn, "")
        if key_stn in self._table:
            return self._table[key_stn]
        return self.default_sigma_ra, self.default_sigma_dec

    # ------------------------------------------------------------------
    # Noise draw
    # ------------------------------------------------------------------

    def draw(
        self,
        sigma_ra: np.ndarray,
        sigma_dec: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw independent Gaussian noise samples.

        Parameters
        ----------
        sigma_ra : np.ndarray, shape (N,)
            Per-observation 1-sigma noise in RA*cos(dec) (arcseconds).
        sigma_dec : np.ndarray, shape (N,)
            Per-observation 1-sigma noise in Dec (arcseconds).
        rng : np.random.Generator
            NumPy random generator (use ``np.random.default_rng(seed)``).

        Returns
        -------
        (delta_ra_cosdec_arcsec, delta_dec_arcsec) : tuple of np.ndarray
            Independent Gaussian draws with the given sigmas, in arcseconds.
        """
        sigma_ra = np.asarray(sigma_ra, dtype=float)
        sigma_dec = np.asarray(sigma_dec, dtype=float)
        n = len(sigma_ra)
        noise_ra = rng.normal(0.0, sigma_ra, size=n)
        noise_dec = rng.normal(0.0, sigma_dec, size=n)
        return noise_ra, noise_dec
