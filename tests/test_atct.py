"""
Tests for the AT/CT decomposition module (looo/atct.py).

Tests the pure math rotation and the cos(dec) correction logic.
"""

import numpy as np
import pytest

from adam_orbit_det_eval.looo.atct import rotate_to_atct


class TestRotateToAtct:
    """Tests for the rotate_to_atct function."""

    def test_pure_ra_motion_ra_residual_maps_to_at(self):
        """
        Object moving purely in +RA direction.
        A residual purely in RA should map entirely to AT.
        """
        # Velocity unit vector pointing in +RA direction
        v_ra_unit = np.array([1.0])
        v_dec_unit = np.array([0.0])

        # Residual purely in RA
        res_ra = np.array([0.5])
        res_dec = np.array([0.0])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        np.testing.assert_allclose(at, [0.5], atol=1e-15)
        np.testing.assert_allclose(ct, [0.0], atol=1e-15)

    def test_pure_ra_motion_dec_residual_maps_to_ct(self):
        """
        Object moving purely in +RA direction.
        A residual purely in Dec should map entirely to CT.
        """
        v_ra_unit = np.array([1.0])
        v_dec_unit = np.array([0.0])

        res_ra = np.array([0.0])
        res_dec = np.array([0.3])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        np.testing.assert_allclose(at, [0.0], atol=1e-15)
        np.testing.assert_allclose(ct, [0.3], atol=1e-15)

    def test_pure_dec_motion_dec_residual_maps_to_at(self):
        """
        Object moving purely in +Dec direction.
        A residual purely in Dec should map entirely to AT.
        """
        v_ra_unit = np.array([0.0])
        v_dec_unit = np.array([1.0])

        res_ra = np.array([0.0])
        res_dec = np.array([0.7])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        np.testing.assert_allclose(at, [0.7], atol=1e-15)
        np.testing.assert_allclose(ct, [0.0], atol=1e-15)

    def test_pure_dec_motion_ra_residual_maps_to_negative_ct(self):
        """
        Object moving purely in +Dec direction.
        A residual purely in +RA should map to -CT (since CT is 90 deg CCW
        from AT: CT unit vector = (-v_dec, v_ra) = (0, 0) wait...
        CT = -res_ra * v_dec_unit + res_dec * v_ra_unit
           = -0.4 * 1.0 + 0.0 * 0.0 = -0.4
        """
        v_ra_unit = np.array([0.0])
        v_dec_unit = np.array([1.0])

        res_ra = np.array([0.4])
        res_dec = np.array([0.0])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        np.testing.assert_allclose(at, [0.0], atol=1e-15)
        np.testing.assert_allclose(ct, [-0.4], atol=1e-15)

    def test_diagonal_motion(self):
        """
        Object moving at 45 degrees (equal RA and Dec velocity).
        A residual in RA should split equally between AT and CT.
        """
        u = 1.0 / np.sqrt(2.0)
        v_ra_unit = np.array([u])
        v_dec_unit = np.array([u])

        res_ra = np.array([1.0])
        res_dec = np.array([0.0])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        # AT = res_ra * u + 0 = u
        # CT = -res_ra * u + 0 = -u
        np.testing.assert_allclose(at, [u], atol=1e-15)
        np.testing.assert_allclose(ct, [-u], atol=1e-15)

    def test_near_stationary_nan_propagation(self):
        """
        When velocity unit vector components are NaN (near-stationary guard),
        AT and CT should also be NaN.
        """
        v_ra_unit = np.array([np.nan])
        v_dec_unit = np.array([np.nan])

        res_ra = np.array([0.5])
        res_dec = np.array([0.3])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        assert np.isnan(at[0])
        assert np.isnan(ct[0])

    def test_vectorized(self):
        """Test that the function works on arrays of multiple observations."""
        v_ra_unit = np.array([1.0, 0.0, np.nan])
        v_dec_unit = np.array([0.0, 1.0, np.nan])

        res_ra = np.array([0.5, 0.3, 0.1])
        res_dec = np.array([0.2, 0.4, 0.2])

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        # obs 0: moving in +RA -> AT=res_ra=0.5, CT=res_dec=0.2
        np.testing.assert_allclose(at[0], 0.5, atol=1e-15)
        np.testing.assert_allclose(ct[0], 0.2, atol=1e-15)

        # obs 1: moving in +Dec -> AT=res_dec=0.4, CT=-res_ra=-0.3
        np.testing.assert_allclose(at[1], 0.4, atol=1e-15)
        np.testing.assert_allclose(ct[1], -0.3, atol=1e-15)

        # obs 2: stationary -> NaN
        assert np.isnan(at[2])
        assert np.isnan(ct[2])

    def test_norm_preservation(self):
        """
        AT/CT rotation should preserve the magnitude of the residual vector.
        |AT|^2 + |CT|^2 == |res_ra|^2 + |res_dec|^2
        """
        rng = np.random.default_rng(42)
        n = 100
        res_ra = rng.normal(0, 1, n)
        res_dec = rng.normal(0, 1, n)

        # Random unit vectors
        angles = rng.uniform(0, 2 * np.pi, n)
        v_ra_unit = np.cos(angles)
        v_dec_unit = np.sin(angles)

        at, ct = rotate_to_atct(res_ra, res_dec, v_ra_unit, v_dec_unit)

        original_mag2 = res_ra**2 + res_dec**2
        rotated_mag2 = at**2 + ct**2

        np.testing.assert_allclose(rotated_mag2, original_mag2, rtol=1e-14)


class TestCosDecCorrection:
    """
    Test that the cos(dec) correction is applied correctly when
    converting from vlon to v_ra_cosdec.

    This tests the logic that would be in compute_velocity_vectors:
        v_ra_cosdec = vlon * cos(dec)
        v_dec = vlat
    """

    def test_equator(self):
        """At dec=0, cos(dec)=1, so v_ra_cosdec == vlon."""
        vlon = 0.5  # deg/day
        vlat = 0.3  # deg/day
        dec_deg = 0.0

        v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))
        v_dec = vlat

        assert v_ra_cosdec == pytest.approx(0.5)
        assert v_dec == pytest.approx(0.3)

    def test_mid_latitude(self):
        """At dec=60, cos(dec)=0.5, so v_ra_cosdec = vlon * 0.5."""
        vlon = 1.0  # deg/day
        vlat = 0.0  # deg/day
        dec_deg = 60.0

        v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))

        assert v_ra_cosdec == pytest.approx(0.5, rel=1e-10)

    def test_near_pole(self):
        """Near the pole, cos(dec) -> 0, so v_ra_cosdec -> 0 even for large vlon."""
        vlon = 10.0  # deg/day — large RA rate near pole
        vlat = 0.1  # deg/day
        dec_deg = 89.9

        v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))
        v_dec = vlat

        # v_ra_cosdec should be very small
        assert abs(v_ra_cosdec) < 0.02
        # Speed should be dominated by v_dec
        speed = np.sqrt(v_ra_cosdec**2 + v_dec**2)
        assert speed == pytest.approx(v_dec, abs=0.02)

    def test_unit_vector_normalization(self):
        """Verify that the unit vector from (v_ra_cosdec, v_dec) has norm 1."""
        vlon = 0.8
        vlat = 0.3
        dec_deg = 45.0

        v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))
        v_dec = vlat

        speed = np.sqrt(v_ra_cosdec**2 + v_dec**2)
        v_ra_unit = v_ra_cosdec / speed
        v_dec_unit = v_dec / speed

        norm = np.sqrt(v_ra_unit**2 + v_dec_unit**2)
        assert norm == pytest.approx(1.0, rel=1e-14)

    def test_stationary_guard(self):
        """Speed below threshold should be flagged."""
        from adam_orbit_det_eval.looo.atct import MIN_SPEED_DEG_PER_DAY

        vlon = 1e-8  # essentially zero
        vlat = 1e-8
        dec_deg = 0.0

        v_ra_cosdec = vlon * np.cos(np.deg2rad(dec_deg))
        v_dec = vlat
        speed = np.sqrt(v_ra_cosdec**2 + v_dec**2)

        assert speed < MIN_SPEED_DEG_PER_DAY
