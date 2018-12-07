import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from itertools import product


N = np.array([[0,1,0],[0,-1,0],[0,0,0]])
S = np.array([[0,0,0],[0,-1,0],[0,1,0]])
E = np.array([[0,0,0],[0,-1,1],[0,0,0]])
W = np.array([[0,0,0],[1,-1,0],[0,0,0]])
NE = np.array([[0,0,1],[0,-1,0],[0,0,0]])
SE = np.array([[0,0,0],[0,-1,0],[0,0,1]])
SW = np.array([[0,0,0],[0,-1,0],[1,0,0]])
NW = np.array([[1,0,0],[0,-1,0],[0,0,0]])

vn_neighbors = (N, S, E, W)
moore_neighbors = (N, S, E, W, NE, SE, SW, NW)

trials = 32
trial_sets = 32
steps = 500
types_range = range(7, 22)


def cca_diff_sums(types, grid_size, steps, neighborhood):
    diff_sums = []
    grid = np.random.randint(types, size=(grid_size, grid_size))
    prev_grid = np.copy(grid)

    for step in range(steps):
        to_increment = np.where(np.logical_or.reduce([ndimage.convolve(grid, neighbor, mode='wrap') % types == 1 for neighbor in neighborhood]))

        grid[to_increment] += 1
        grid[to_increment] %= types
        
        diff = (grid - prev_grid) % types
        diff_sums.append(np.sum(diff))

        prev_grid[:] = grid

    return diff_sums

def cca_inert_bonds(types, grid_size, steps, neighborhood):
    inerts = []
    grid = np.random.randint(types, size=(grid_size, grid_size))
    prev_grid = np.copy(grid)

    for step in range(steps):
        to_increment = np.where(np.logical_or.reduce([ndimage.convolve(grid, neighbor, mode='wrap') % types == 1 for neighbor in neighborhood]))

        grid[to_increment] += 1
        grid[to_increment] %= types
        
        inerts.append(np.sum(np.logical_or.reduce([ndimage.convolve(grid, neighbor, mode='wrap') == 0 for neighbor in neighborhood])))

        prev_grid[:] = grid

    return inerts

def format_data(trials, types_range):
    out = 'types, trial, data...\n'
    for types in types_range:
        for i, trial in enumerate(trials[types]):
            out += '{}, {}, {}\n'.format(types, i,','.join(map(str, trial)))

    return out



def run_trial_set(args):
    grid_size, number = args
    np.random.seed()
    
    # Von Neumann Neighbor Runs
    vn_trials = {}
    for types in types_range:
        print(grid_size, 'VN', number, ':', types)
        vn_trials[types] = []
        for trial in range(trials):
            vn_trials[types].append(cca_diff_sums(types, grid_size, steps, vn_neighbors))

    with open('{}_vn_neighbor_diff_data_{}.csv'.format(grid_size, number), 'w') as f:
        f.write(format_data(vn_trials, types_range))

    # Moore Neighbor Runs
    moore_trials = {}
    for types in types_range:
        print(grid_size, 'MOORE', number, ':', types)
        moore_trials[types] = []
        for trial in range(trials):
            moore_trials[types].append(cca_diff_sums(types, grid_size, steps, moore_neighbors))

    with open('{}_moore_neighbor_diff_data_{}.csv'.format(grid_size, number), 'w') as f:
        f.write(format_data(moore_trials, types_range))


if __name__ == '__main__':
    pool = Pool(16)
    pool.map(run_trial_set, product([128, 256, 512], range(trial_sets)))
    pool.close()
    pool.join()
