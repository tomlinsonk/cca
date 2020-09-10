import time
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

N = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
S = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
E = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
W = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
NE = np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]])
SE = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])
SW = np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]])
NW = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

VN_NEIGHBORS = (N, S, E, W)
MOORE_NEIGHBORS = (N, S, E, W, NE, SE, SW, NW)

TRIALS_PER_SET = 32
TRIAL_SETS = 32
STEPS = 500
THREADS = 20
TYPE_RANGE = range(7, 20)


def cca_diff_sums(types, grid_size, steps, neighborhood):
    diff_sums = []
    grid = np.random.randint(types, size=(grid_size, grid_size))
    prev_grid = np.copy(grid)

    for step in range(steps):
        to_increment = np.where(
            np.logical_or.reduce(
                [ndimage.convolve(grid, neighbor, mode='wrap') % types == 1 for neighbor in neighborhood]
            )
        )

        grid[to_increment] += 1
        grid[to_increment] %= types

        diff = (grid - prev_grid) % types
        diff_sums.append(np.sum(diff))

        prev_grid[:] = grid

    return diff_sums


def format_data(trials, types_range):
    out = 'types, trial, data...\n'
    for types in types_range:
        for i, trial in enumerate(trials[types]):
            out += '{}, {}, {}\n'.format(types, i, ','.join(map(str, trial)))

    return out


def run_trial_set(args):
    grid_size, number = args
    np.random.seed()

    # Von Neumann Neighbor Runs
    vn_trials = {}
    for types in TYPE_RANGE:
        print(grid_size, 'VN', number, ':', types)
        vn_trials[types] = []
        for trial in range(TRIALS_PER_SET):
            vn_trials[types].append(cca_diff_sums(types, grid_size, STEPS, VN_NEIGHBORS))

    with open('{}_vn_neighbor_diff_data_{}.csv'.format(grid_size, number), 'w') as f:
        f.write(format_data(vn_trials, TYPE_RANGE))

    # Moore Neighbor Runs
    moore_trials = {}
    for types in TYPE_RANGE:
        print(grid_size, 'MOORE', number, ':', types)
        moore_trials[types] = []
        for trial in range(TRIALS_PER_SET):
            moore_trials[types].append(cca_diff_sums(types, grid_size, STEPS, MOORE_NEIGHBORS))

    with open('{}_moore_neighbor_diff_data_{}.csv'.format(grid_size, number), 'w') as f:
        f.write(format_data(moore_trials, TYPE_RANGE))


if __name__ == '__main__':
    start = time.time()

    pool = Pool(THREADS)
    pool.map(run_trial_set, product([256, 512, 1024], range(TRIAL_SETS)))
    pool.close()
    pool.join()

    print('Runtime:', time.time() - start)
