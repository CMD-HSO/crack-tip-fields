# crack_tip_fields – documentation

`crack_tip_fields` computes near-tip displacement and stress fields of cracks in isotropic material,
anisotropic single crystals (Stroh formalism) and along interfaces between two anisotropic grains, and
exports them as displacement boundary conditions for Abaqus boundary-layer (K-field) models.

| Topic | Where |
|---|---|
| Installation, conventions, app workflow, API | [README](../README.md) |
| Theory, verification and FE case study | accompanying article (submitted) and its supplementary material (1: usage instructions, 2: derivations and verification) |
| Worked examples and validation | [`notebooks/crack_tip_fields_validation_and_examples.ipynb`](../notebooks/crack_tip_fields_validation_and_examples.ipynb) |
| Interactive boundary-condition app | [`notebooks/abaqus_crack_tip_bc_app.ipynb`](../notebooks/abaqus_crack_tip_bc_app.ipynb) |
| Generalised plane strain (3D slice) | [`scripts/make_gpe_model.py`](../scripts/make_gpe_model.py), README section "2D or 3D model?" |
| Evaluation of Abaqus results | [`scripts/evaluate_kfield_abaqus.py`](../scripts/evaluate_kfield_abaqus.py), [`scripts/compare_stress_circle.py`](../scripts/compare_stress_circle.py) |
| Abaqus example models | [`examples/fe_case_study/`](../examples/fe_case_study/README.md) |
| Verification scripts | [`tests/validate_stress_interface.py`](../tests/validate_stress_interface.py), [`scripts/reproduce_manuscript_results.py`](../scripts/reproduce_manuscript_results.py) |

All public functions and classes of `src/crack_tip_fields.py` have docstrings, e.g.

```python
import sys; sys.path.insert(0, "src")
import crack_tip_fields as ctf
help(ctf.InterfaceCrackModel)
```
