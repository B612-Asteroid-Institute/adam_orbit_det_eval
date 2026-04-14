"""Tests for comet detection in looo/eligibility.py."""

from adam_orbit_det_eval.looo.eligibility import is_comet


class TestIsComet:
    """Tests for is_comet() function."""

    def test_classic_comet_prefixes(self):
        assert is_comet("C/2023 A1") is True
        assert is_comet("P/2024 B2") is True

    def test_defunct_comet(self):
        assert is_comet("D/1993 F2") is True  # Shoemaker-Levy 9

    def test_reclassified_comet(self):
        assert is_comet("A/2018 V3") is True

    def test_interstellar_objects(self):
        assert is_comet("I/2017 U1") is True  # 'Oumuamua
        assert is_comet("2I/Borisov") is True  # numbered interstellar

    def test_numbered_periodic_comets_without_name(self):
        assert is_comet("1P") is True  # Halley
        assert is_comet("109P") is True  # Swift-Tuttle

    def test_numbered_periodic_comets_with_name(self):
        assert is_comet("1P/Halley") is True
        assert is_comet("109P/Swift-Tuttle") is True
        assert is_comet("153P/Ikeya-Zhang") is True

    def test_asteroids_not_comets(self):
        assert is_comet("(12345)") is False
        assert is_comet("(1P)") is False  # parenthesized — asteroid provisional
        assert is_comet("2024 PD1") is False  # asteroid provisional with P in it
        assert is_comet("12345") is False  # numbered asteroid
        assert is_comet("(433)") is False  # Eros
