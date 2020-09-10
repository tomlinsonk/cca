import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from scipy.interpolate import interp1d
import scipy.signal
import random
import pickle
import itertools
import sys
import os
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

cca_dir = '/Users/tomlinsonk/Projects/Research/cca/'


def odd_round(x):
    return max(5, 2 * int(x / 2) + 1)


def main(neighborhood, size):
    vn = neighborhood == 'vn'

    print('Processing {0}x{0} {1} data...'.format(size, 'von Neumann' if vn else 'Moore'))
    k_range = range(7, 16) if vn else range(11, 20)

    if size == 128:
        if vn:
            min_step = {7:8, 8:8, 9:13, 10:18, 11:26, 12:44, 13:79, 14:128, 15: 160}
            widths = {7:9, 8:11, 9:17, 10:21, 11:27, 12:51, 13:71, 14:61, 15: 101}
        else:
            min_step = {11: 11, 12: 14, 13: 19, 14: 22, 15: 27, 16: 36, 17: 43, 18: 55, 19: 67, 20: 72, 21: 86, 22: 104, 23: 125, 24:161, 25: 197, 26: 217}
            widths = {11: 5, 12: 9, 13: 15, 14: 15, 15: 15, 16: 25, 17: 25, 18: 31, 19: 41, 20:51, 21: 61, 22: 81, 23: 91, 24:101, 25: 101, 26: 101}
    elif size == 256:
        if vn:
            min_step = {7:8, 8:8, 9:13, 10:18, 11:26, 12:44, 13:79, 14:128, 15: 160}
            widths = {7:9, 8:7, 9:17, 10:23, 11:31, 12:41, 13:51, 14:61, 15: 81}
        else:
            min_step = {11: 10, 12: 14, 13: 19, 14: 22, 15: 27, 16: 36, 17: 43, 18: 55, 19: 67, 20: 72, 21: 86, 22: 104, 23: 125, 24:161, 25: 197, 26: 217}
            widths = {11: 9, 12: 9, 13: 13, 14: 17, 15: 25, 16: 25, 17: 35, 18: 41, 19: 51, 20: 37, 21: 41, 22: 49, 23: 51, 24:71, 25: 91, 26: 101}
    elif size == 512:
        if vn:
            min_step = {7:6, 8:8, 9:13, 10:18, 11:26, 12:44, 13:79, 14:128, 15: 160}
            widths = {7:5, 8:11, 9:17, 10:21, 11:27, 12:51, 13:71, 14:61, 15: 101}
        else:
            min_step = {11: 10, 12: 14, 13: 19, 14: 22, 15: 27, 16: 36, 17: 43, 18: 55, 19: 67, 20: 72, 21: 86, 22: 104, 23: 125, 24:161, 25: 197, 26: 217}
            widths = {11: 7, 12: 9, 13: 15, 14: 17, 15: 25, 16: 25, 17: 31, 18: 37, 19: 41, 20: 37, 21: 41, 22: 49, 23: 51, 24:71, 25: 91, 26: 101}
    elif size == 1024:
        if vn:
            min_step = {7:6, 8:8, 9:13, 10:18, 11:26, 12:44, 13:79, 14:128, 15: 160}
            widths = {7:5, 8:11, 9:17, 10:21, 11:27, 12:51, 13:71, 14:61, 15: 101}
        else:
            min_step = {11: 10, 12: 14, 13: 19, 14: 22, 15: 27, 16: 36, 17: 43, 18: 55, 19: 67, 20: 72, 21: 86, 22: 104, 23: 125, 24:161, 25: 197, 26: 217}
            widths = {11: 7, 12: 9, 13: 15, 14: 17, 15: 25, 16: 25, 17: 31, 18: 37, 19: 41, 20: 37, 21: 41, 22: 49, 23: 51, 24:71, 25: 91, 26: 101}
    else:
        print('Size must be one of: 128, 256, 512, 1024')
        exit(1)

    print('Loading file...')
    data_file = cca_dir + 'data/{}_{}.csv'.format(size, neighborhood)
    raw_data = np.loadtxt(data_file, dtype=int, delimiter=',')

    data = {k: [] for k in k_range}
    for row in raw_data:
        if row[0] in k_range:
            data[row[0]].append(row[2:])

    for i in range(1):
        print('Run', i)
        debris_lengths = {k: [] for k in k_range}
        droplet_lengths = {k: [] for k in k_range}
        defect_lengths = {k: [] for k in k_range}

        # noisy_widths = {k: odd_round(widths[k] * (random.random() + 0.5)) for k in widths}
        noisy_widths = {k: widths[k] for k in widths}

        showing = False
        print('Finding phase lengths...')
        for k in k_range:
            print('k =', k)
            # print('Unique trials:', len(set(tuple(trial) for trial in data[k])))
            invalid = 0

            for trial in range(len(data[k])):

                min_diff = np.argmin(data[k][trial])

                # deriv1 = np.gradient(means, 1)[min_step[k]:]

                # print(data[k][trial])

                deriv2 = np.gradient(np.gradient(data[k][trial], 1), 1)[min_step[k]:]
                direct_deriv = scipy.signal.savgol_filter(data[k][trial], noisy_widths[k], 3, mode='nearest', deriv=2)[min_step[k]:]

                min_deriv = np.argmin(direct_deriv) + min_step[k]
                max_deriv = np.argmax(direct_deriv) + min_step[k]

                if max_deriv - min_diff < 0 or min_deriv - max_deriv < 0:
                    invalid += 1
                    continue

                if trial == 4 and k == 13:
                    if size == 128:
                        yticks = [0, 4000, 8000, 12000, 16000]
                    elif size == 256:
                        yticks = [0, 15000, 30000, 45000, 60000]
                    elif size == 512:
                        yticks = [0, 50000, 100000, 150000, 200000, 250000, 300000]
                    elif size == 1024:
                        yticks = [0, 200000, 400000, 600000, 800000, 1000000]

                    fig, axes = plt.subplots(nrows=2, figsize=(5, 3))
                    plt.locator_params(nbins=5)
                    plt.rc('text', usetex=True)
                    plt.rc('font', family='serif')

                    axes[0].plot(range(500), data[k][trial])
                    axes[0].scatter(min_diff, np.min(data[k][trial]))
                    axes[0].scatter(min_deriv, data[k][trial][min_deriv])
                    axes[0].scatter(max_deriv, data[k][trial][max_deriv])
                    axes[0].set_ylabel('$\Delta(t)$')
                    axes[0].set_xlim(0, 500)
                    axes[0].set_ylim(bottom=0)
                    if size == 128:
                        axes[0].set_ylim(top=17000)

                    axes[1].plot(range(min_step[k], 500), deriv2, label='Raw derivative')
                    axes[1].plot(range(min_step[k], 500), direct_deriv, color='black', label='Savitzky-Golay output')
                    axes[1].set_ylabel("$\Delta''(t)$")
                    axes[1].set_xlim(0, 500)

                    # axes[1].legend(loc='upper right', fontsize=8)
                    axes[0].tick_params(axis='x', which='major', labelsize=8)
                    axes[1].tick_params(axis='both', which='major', labelsize=8)
                    plt.sca(axes[0])
                    plt.yticks(yticks, yticks, fontsize=8)
                    plt.sca(axes[1])

                    plt.suptitle('$\Delta(t)$ curve ($k={}$, {} neighborhood)'.format(k, 'von Neumann' if vn else 'Moore'))
                    plt.xlabel('Step ($t$)')
                    plt.yticks(rotation='horizontal')
                    # os.makedirs(cca_dir + 'plots/diff_curves/{}_{}/'.format(size, neighborhood), exist_ok=True)
                    # plt.savefig(cca_dir + 'plots/diff_curves/{0}_{1}/{0}_{1}_k_{2}.pdf'.format(size, neighborhood, k), bbox_inches='tight')
                    plt.savefig('ex-curve-points.pdf', bbox_inches='tight')
                    plt.show()
                    plt.close()


                debris_lengths[k].append(min_diff)
                droplet_lengths[k].append(max_deriv - min_diff)
                defect_lengths[k].append(min_deriv - max_deriv)

            print('Invalid:', invalid)

        # print('Pickling results...')
        # with open(cca_dir + '/pickles/noisy_{}_{}_{}.pkl'.format(i, size, neighborhood), 'wb') as f:
        #     pickle.dump([debris_lengths, droplet_lengths, defect_lengths], f)


if __name__ == '__main__':
    main('vn', 256)
    # for neighborhood, size in itertools.product(['vn', 'moore'], [128, 256, 512, 1024]):
    #     main(neighborhood, size)
