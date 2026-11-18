"""Closed-form interface stresses for six grain pairs (beta up to 0.071) and all loading modes (tests/validate_stress_interface.py):
Hwu's solution on the interface, finite differences, continuity across the interface, equilibrium."""
import validate_stress_interface as vsi


def test_interface_stresses():
    assert vsi.main() == 0
