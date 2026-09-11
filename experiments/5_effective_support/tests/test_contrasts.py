"""Unit tests for contrasts, LaTeX table generation, and plotting."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from breadth import BREADTH_LADDER, DEPTH_LADDER, GRID_CELLS
from breadth.analyze.report import (
    _build_contrast_table,
    _build_grid_table,
    _build_icc_table,
    _build_surface_table,
    generate_figures,
    generate_latex_tables,
)


def test_contrast_identities():
    """Contrast definitions hold numerically."""
    cells = GRID_CELLS
    accs = {c: float(c[0] * 2 + c[1] * 0.5) for c in cells}

    delta_m = {g: accs[(g, 32)] - accs[(g, 8)] for g in BREADTH_LADDER}
    delta_g = {m: accs[(20, m)] - accs[(5, m)] for m in DEPTH_LADDER}
    x_val = accs[(20, 8)] - accs[(5, 32)]

    for g in BREADTH_LADDER:
        assert np.isclose(delta_m[g], (32 - 8) * 0.5)

    for m in DEPTH_LADDER:
        assert np.isclose(delta_g[m], (20 - 5) * 2.0)

    assert np.isclose(x_val, (20 * 2 + 8 * 0.5) - (5 * 2 + 32 * 0.5))


def test_latex_table_builders():
    """Table generation functions produce valid LaTeX code."""
    mock_cell_accs = {
        f"G{g}_m{m}": {
            "point": 75.0,
            "ci_2_5": 72.0,
            "ci_97_5": 78.0,
            "draw_dispersion": 0.5,
            "n_eff": 50.0,
        }
        for g, m in GRID_CELLS
    }

    grid_tex = _build_grid_table(mock_cell_accs)
    assert r"\begin{tabular}{lrrr}" in grid_tex
    assert "Patients per class $G$" in grid_tex

    mock_contrasts = {
        "delta_m": {str(g): {"point": 2.5, "ci_2_5": 1.0, "ci_97_5": 4.0} for g in BREADTH_LADDER},
        "delta_g": {str(m): {"point": 4.0, "ci_2_5": 2.5, "ci_97_5": 5.5} for m in DEPTH_LADDER},
        "equal_budget_advantage_X": {"point": 3.0, "ci_2_5": 1.5, "ci_97_5": 4.5},
    }
    contrast_tex = _build_contrast_table(mock_contrasts)
    assert r"Equal-budget advantage $X" in contrast_tex

    mock_surface_params = {
        "beta_n": {"point": 2.1, "ci_2_5": 1.5, "ci_97_5": 2.7},
        "res_std_n": {"point": 1.2, "ci_2_5": 0.9, "ci_97_5": 1.5},
        "beta_g": {"point": 3.4, "ci_2_5": 2.8, "ci_97_5": 4.0},
        "res_std_g": {"point": 0.8, "ci_2_5": 0.6, "ci_97_5": 1.1},
        "beta_neff": {"point": 3.0, "ci_2_5": 2.4, "ci_97_5": 3.6},
        "res_std_neff": {"point": 0.5, "ci_2_5": 0.3, "ci_97_5": 0.7},
        "gamma_n": {"point": 1.8, "ci_2_5": 1.0, "ci_97_5": 2.6},
        "res_std_aug_nom": {"point": 0.4, "ci_2_5": 0.3, "ci_97_5": 0.6},
        "gamma_e": {"point": 0.3, "ci_2_5": -0.4, "ci_97_5": 1.0},
        "res_std_aug_eff": {"point": 0.4, "ci_2_5": 0.3, "ci_97_5": 0.6},
    }
    surface_tex = _build_surface_table(mock_surface_params)
    assert r"Augmented effective" in surface_tex

    mock_cohort_iccs = {"Class_A": 0.08, "Class_B": 0.15}
    icc_tex = _build_icc_table(mock_cohort_iccs)
    assert "Class A" in icc_tex
    assert "Mean" in icc_tex


def test_generate_latex_tables_and_figures(tmp_path: Path):
    """generate_latex_tables and generate_figures write files to target directory."""
    mock_results = {
        "cell_accuracies": {
            f"G{g}_m{m}": {
                "point": 70.0 + g * 0.5 + m * 0.2,
                "ci_2_5": 70.0 + g * 0.5 + m * 0.2 - 2.0,
                "ci_97_5": 70.0 + g * 0.5 + m * 0.2 + 2.0,
                "draw_dispersion": 0.3,
                "n_eff": float(g * m),
            }
            for g, m in GRID_CELLS
        },
        "contrasts": {
            "delta_m": {str(g): {"point": 2.0, "ci_2_5": 1.0, "ci_97_5": 3.0} for g in BREADTH_LADDER},
            "delta_g": {str(m): {"point": 3.0, "ci_2_5": 2.0, "ci_97_5": 4.0} for m in DEPTH_LADDER},
            "equal_budget_advantage_X": {"point": 1.5, "ci_2_5": 0.5, "ci_97_5": 2.5},
        },
        "surface_parameters": {
            "beta_n": {"point": 2.0, "ci_2_5": 1.5, "ci_97_5": 2.5},
            "res_std_n": {"point": 1.0, "ci_2_5": 0.8, "ci_97_5": 1.2},
            "beta_g": {"point": 3.0, "ci_2_5": 2.5, "ci_97_5": 3.5},
            "res_std_g": {"point": 0.8, "ci_2_5": 0.6, "ci_97_5": 1.0},
            "beta_neff": {"point": 2.8, "ci_2_5": 2.3, "ci_97_5": 3.3},
            "res_std_neff": {"point": 0.5, "ci_2_5": 0.3, "ci_97_5": 0.7},
            "gamma_n": {"point": 1.5, "ci_2_5": 0.8, "ci_97_5": 2.2},
            "res_std_aug_nom": {"point": 0.4, "ci_2_5": 0.3, "ci_97_5": 0.5},
            "gamma_e": {"point": 0.2, "ci_2_5": -0.3, "ci_97_5": 0.8},
            "res_std_aug_eff": {"point": 0.4, "ci_2_5": 0.3, "ci_97_5": 0.5},
        },
        "secondary": {
            "cells": {
                f"G{g}_m{m}": {
                    key: {"point": 1.0, "ci_2_5": 0.5, "ci_97_5": 1.5}
                    for key in (
                        "macro_nll",
                        "expected_calibration_error",
                        "patch_micro_balanced_accuracy",
                    )
                }
                for g, m in GRID_CELLS
            },
            "equal_budget_X": {
                key: {"point": -0.2, "ci_2_5": -0.5, "ci_97_5": 0.1}
                for key in (
                    "macro_nll",
                    "expected_calibration_error",
                    "patch_micro_balanced_accuracy",
                )
            },
            "class_recalls": {
                "Class_1": {
                    "G5_m32": {"point": 50.0, "ci_2_5": 45.0, "ci_97_5": 55.0},
                    "G20_m8": {"point": 55.0, "ci_2_5": 50.0, "ci_97_5": 60.0},
                    "equal_budget_X": {"point": 5.0, "ci_2_5": 1.0, "ci_97_5": 9.0},
                }
            },
        },
    }
    mock_iccs = {"Class_1": 0.1}

    out_tables = tmp_path / "tables"
    generate_latex_tables(mock_results, mock_iccs, out_tables)
    assert (out_tables / "grid_accuracies.tex").exists()
    assert (out_tables / "contrasts.tex").exists()
    assert (out_tables / "surface_fits.tex").exists()
    assert (out_tables / "class_iccs.tex").exists()
    assert (out_tables / "secondary_endpoints.tex").exists()
    assert (out_tables / "class_recalls.tex").exists()

    out_figures = tmp_path / "figures"
    generate_figures(mock_results, out_figures)
    assert (out_figures / "support_surface_nominal.png").exists()
