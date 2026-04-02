import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of test days where predicted direction matches actual direction.
    Uses actual[i-1] as the 'today' reference; first test day is skipped.
    Random-guess baseline = 50%.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    actual_dir = np.sign(actual[1:] - actual[:-1])
    pred_dir   = np.sign(predicted[1:] - actual[:-1])
    return float(np.mean(actual_dir == pred_dir) * 100)


def compute_all(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mape": mape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae":  mae(actual, predicted),
        "da":   directional_accuracy(actual, predicted),
    }
