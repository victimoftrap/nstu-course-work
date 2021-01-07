from typing import List
import scipy.stats as stat


def pearson_chi_squared(data: List[float], intervals: int = 7):
    """Нахождение статистики с помощью критерия хи-квадрат Пирсона.

    :param data: выборка
    :param intervals: количество интервалов для группирования выборки
    :return: значение статистики.
    """
    sorted_data = sorted(data)
    max_value, min_value = max(sorted_data), min(sorted_data)
    value_length = max_value - min_value
    step = value_length / intervals

    # to intervals
    interval_shift = step
    intervals_frequency = [0 for i in range(intervals)]
    interval_index = 0
    for x in sorted_data:
        if x <= (sorted_data[0] + interval_shift):
            intervals_frequency[interval_index] += 1
        else:
            interval_shift += step
            interval_index += 1

    # probability to enter the interval
    intervals_probability = [0 for i in range(intervals)]
    for i in range(intervals):
        intervals_probability[i] = intervals_frequency[i] / len(sorted_data)

    # boundary points o interval
    boundary_points = [0 for i in range(intervals)]
    boundary_points[0] = sorted_data[0] + step
    for i in range(1, intervals - 1):
        boundary_points[i] = boundary_points[i - 1] + step

    # Pearson chi-squared
    estimated_interval_entering = [0 for i in range(intervals)]
    sum = 0

    estimated_interval_entering[0] = stat.expon.pdf(boundary_points[0])
    sum += estimated_interval_entering[0]
    for i in range(1, intervals - 1):
        estimated_interval_entering[i] = stat.expon.pdf(boundary_points[i]) - sum
        sum += estimated_interval_entering[i]
    estimated_interval_entering[intervals - 1] = 1 - sum

    p_square = 0
    for i in range(intervals):
        p_square += ((intervals_probability[i] - estimated_interval_entering[i]) ** 2) / estimated_interval_entering[i]
    return p_square * len(sorted_data)
