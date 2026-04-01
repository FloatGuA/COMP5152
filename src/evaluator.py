import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def compute_all(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mape": mape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mae":  mae(actual, predicted),
    }
