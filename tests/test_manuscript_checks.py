"""Numerical checks of the manuscript (Appendix B and Sections 4.4, 4.5), taken from
scripts/reproduce_manuscript_results.py: every value is compared with its tolerance there."""
import pytest

import reproduce_manuscript_results as rmr


@pytest.mark.parametrize("check", [
    rmr.appendix_b1,                    # C.1 isotropic solution: Williams, Stroh limit, energy release rate
    rmr.appendix_b2_and_section_4_1,    # C.2 / 4.1 single crystal: K normalisation, opening, finite differences
    rmr.appendix_b3_and_section_4_2,    # C.3 / 4.2 interface crack: Brunner Eq. (24), interpenetration, Griffith
    rmr.appendix_c4,                    # C.4 isotropic limits, exact crack-face evaluation (regression test)
], ids=["C1_isotropic", "C2_single_crystal", "C3_interface", "C4_isotropic_limits"])
def test_manuscript_checks(check):
    rmr.FAILED.clear()
    check()
    assert rmr.FAILED == [], "failed checks: " + "; ".join(rmr.FAILED)
