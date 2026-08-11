import numpy as np

from mihic_repro.metrics import expected_calibration_error, metric_bundle


def test_perfect_predictions_have_zero_ece_and_nll():
    labels = np.array([0, 1, 2])
    probabilities = np.eye(3)
    metrics = metric_bundle(labels, probabilities, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["ece_15bin"] == 0.0
    assert np.isclose(metrics["nll"], 0.0)


def test_ece_matches_manual_single_bin_example():
    labels = np.array([0, 1])
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4]])
    # One correct, one incorrect; mean confidence 0.7, accuracy 0.5.
    assert np.isclose(expected_calibration_error(probabilities, labels, bins=1), 0.2)
