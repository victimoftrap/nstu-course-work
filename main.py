import isw_file
import epps_pulley
import chi_squared

import math
import argparse

import numpy


def alternative(size):
    xs = numpy.random.random(size)
    return list(map(lambda x: 1 - math.exp(-x), xs))


def monte_carlo_method(xs, iterations):
    xs_statistic = epps_pulley.exponentiality_statistics(xs)
    m = 0
    for i in range(iterations):
        ys = numpy.random.exponential(1, size=len(xs))
        ys_statistic = chi_squared.pearson_chi_squared(ys)
        if ys_statistic > xs_statistic:
            m += 1
    return m / len(xs)


def main(n: float, scale: int, alpha: float, mciters: int):
    sample_data = numpy.random.exponential(scale=scale, size=n)
    sample_data = sample_data.tolist()
    isw_file.save_to_isw_file(sample_data, 'exp2.dat', f'exponential with {scale} scale')

    # ex = isw_file.read_from_isw_file('exp.dat')
    stat = epps_pulley.exponentiality_statistics(sample_data)
    print('статистика криетрия: ', stat)

    p_value = monte_carlo_method(sample_data, mciters)
    print('p-value:', p_value)

    if p_value > alpha:
        print('гипотеза не отклоняется')
    else:
        print('гипотеза отвергается')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Epps-Pulley exponentiality test")
    parser.add_argument('-n', dest='n', type=int, help="объём выборки")
    parser.add_argument('--scale', dest='scale', type=float, help="параметр масштаба")
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, help="заданный уровень значимости")
    parser.add_argument('-m', '--mciters', dest='mciters', type=int, help="количество итераций метода Монте-Карло")
    parsed_args = parser.parse_args()
    # main(parsed_args.n, parsed_args.scale, parsed_args.alpha, parsed_args.mciters)

    size = 100
    scale = 1
    alpha = 0.05
    monte_carlo_iterations = 100
    main(size, scale, alpha, monte_carlo_iterations)
