from typing import List
import statistics
import math


def normality_statistics(xs: List[float]) -> float:
    """Вычислить статистику критерия Эппса-Палли для нормального закона.

    :param xs: выборка
    :return: значение статистики критерия
    """
    xs_length = len(xs)
    mean = statistics.mean(xs)
    variance = statistics.pvariance(xs)

    a = 0
    variance_x4 = 4 * variance
    for x in xs:
        first = (x - mean) ** 2
        a += math.exp(- first / variance_x4)
    a *= math.sqrt(2)

    b = 0
    variance_x2 = 2 * variance
    for k in range(1, xs_length):
        exponents_sum = 0
        for j in range(k):
            first = ((xs[j] - xs[k]) ** 2)
            exponents_sum += math.exp(- first / variance_x2)
        b += exponents_sum
    b *= (2 / xs_length)
    return 1 + (xs_length / math.sqrt(3)) + b - a


def exponentiality_statistics(xs: List[float]) -> float:
    """Вычислить статистику критерия Эппса-Палли для экспоненциального закона.

    :param xs: выборка
    :return: значение статистики критерия
    """
    mean = statistics.mean(xs)
    ys = map(lambda x: x / mean, xs)

    exponents_sum = 0
    for y in ys:
        exponents_sum += math.exp(-y)
    exponents_sum *= 1 / len(xs)
    exponents_sum -= 0.5
    return exponents_sum * math.sqrt(48 * len(xs))
