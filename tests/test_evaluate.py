import pandas as pd

from metaextract.evaluate import evaluate


def _row(depth, mean):
    return {
        "first_author": "A",
        "year": 2020,
        "variable_name": "SOC",
        "treatment_group": "T",
        "control_group": "C",
        "sampling_depth_cm": depth,
        "mean_t": mean,
        "sd_t": 1,
        "n_t": 3,
        "mean_c": mean,
        "sd_c": 1,
        "n_c": 3,
    }


def test_matching_is_one_to_one_and_keeps_depth_identity(tmp_path):
    truth_path = tmp_path / "truth.csv"
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame([_row("0-10", 1), _row("10-20", 2)]).to_csv(truth_path, index=False)
    pd.DataFrame([_row("0-10", 1)]).to_csv(pred_path, index=False)

    report = evaluate(pred_path, truth_path)

    assert report["rows"] == {
        "ground_truth": 2,
        "predicted": 1,
        "matched": 1,
        "row_recall": 0.5,
        "row_precision": 1.0,
    }
    assert report["overall_field_recovery"] == 0.5
