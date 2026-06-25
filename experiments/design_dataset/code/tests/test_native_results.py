from analysis.plotting.results import (
    _result_colspec_headers,
    _result_method_row,
    native_results,
)

import pandas as pd


def test_native_results_parse_and_map() -> None:
    native = native_results()
    # Companion labels are mapped to this report's display names.
    assert native["patch"]["OKO"]["balanced_accuracy"] == (0.791, 0.005)
    assert native["patch"]["CFAL"]["macro_f1"] == (0.798, 0.009)
    # "Weighted MIL" maps onto the WSI "Weighted CE" row.
    assert native["wsi_bag"]["Weighted CE"]["balanced_accuracy"] == (0.833, 0.019)
    assert native["wsi_bag"]["RankMix"]["macro_f1"] == (0.854, 0.008)


def test_native_column_appended_to_method_row() -> None:
    native = {"OKO": {"balanced_accuracy": (0.79, 0.01), "macro_f1": (0.78, 0.01)}}
    part = pd.DataFrame(
        [
            {
                "parameter": p,
                "balanced_accuracy_mean": 0.7,
                "balanced_accuracy_std": 0.0,
                "macro_f1_mean": 0.7,
                "macro_f1_std": 0.0,
            }
            for p in (0.5, 1.0, 1.5)
        ]
    )
    _, headers = _result_colspec_headers([0.5, 1.0, 1.5], with_native=True)
    assert "Native" in headers[0]
    row = _result_method_row("patch_feature_oko", part, [0.5, 1.0, 1.5], native)
    # Three lambda blocks + one native block => four BAcc/F1 pairs.
    assert row.count("$\\pm$") == 8
    assert row.endswith("\\num{0.780} $\\pm$ \\num{0.010}\\\\")
