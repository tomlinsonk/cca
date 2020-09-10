import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import itertools
import warnings
import random

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

cca_dir = '/Users/tomlinsonk/Projects/Research/cca/'

neighborhoods = ['vn', 'moore']
sizes = [256, 512, 1024]

exponents = {
    ('vn', 'Debris'): 2.5566666666666666,
    ('vn', 'Droplet'): 4.8126488150890,
    ('vn', 'Defect'): 3.154766355140188,
    ('moore', 'Debris'):2.5633333333333335,
    ('moore', 'Droplet'): 4.340642838776035,
    ('moore', 'Defect'): 2.8148706350642296,

}

constants = {
    ('vn', 'Debris'): 0.017665536065735697,
    ('vn', 'Droplet'): 0.00052960800893556,
    ('vn', 'Defect'): 0.016296612303674927,
    ('moore', 'Debris'):0.005629936857258731,
    ('moore', 'Droplet'): 0.00023985853799396366,
    ('moore', 'Defect'): 0.011050062693277131,

}


vn_range = range(7, 16)
moore_range = range(11, 20)
Z = 1.96


def linear_model(x, m, b):
    return m * x + b


def plot_noisy(phase, normal_data, noisy_data, neighborhood):
    if neighborhood == 'moore':
        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':
        k_range = vn_range
        log_range = np.log(vn_range)

    plt.figure(figsize=(4, 3))

    power = []
    constant = []

    for data in noisy_data:

        log_means = [np.log(np.mean(data[k])) for k in k_range]

        log_data = [point for k in k_range for point in np.log(data[k])]
        log_k = [np.log(k) for k in k_range for point in np.log(data[k])]

        log_error = [np.std(data[k], ddof=1) / np.mean(data[k]) for k in k_range]
        # print(log_error)

        popt, pcov = curve_fit(linear_model, log_k, log_data)

        power.append(popt[0])
        constant.append(popt[1])

        # plt.scatter(log_k + np.random.normal(0, 0.01, len(log_k)), log_data, alpha=0.1)
        jitter = np.random.normal(0, 0.01, len(log_range))
        plt.plot(log_range + jitter, linear_model(log_range + jitter, *popt), 'r')
        plt.errorbar(log_range + jitter, log_means, yerr=log_error, fmt='b.' if data == normal_data else 'g.', capsize=3)
        # plt.title('{0} Phase Length ({1}x{1} grid, {2} neighborhood)'.format(phase, size,
        #                                                                      'Von Neuman' if neighborhood == 'vn' else 'Moore'))

    print('log {} = ({:3f} +- {:3f}) log k + {:3f} +- {:3f}'.format(phase, np.mean(power), np.std(power) / np.sqrt(len(power)) * Z, np.mean(constant), np.std(constant) /  np.sqrt(len(power)) * Z))

    plt.xlabel('log k')
    plt.ylabel('log {} Phase Length'.format(phase))
    # os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    # plt.savefig(cca_dir + 'plots/phase_lengths/{}_{}_{}.pdf'.format(size, neighborhood, phase.lower()),
    #             bbox_inches='tight')

    plt.show()


def plot_bootstrap(phase, data, neighborhood):
    if neighborhood == 'moore':
        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':
        k_range = vn_range
        log_range = np.log(vn_range)

    plt.figure(figsize=(4, 3))

    power = []
    constant = []

    for i in range(8):
        sampled_data = {k: data[k][i*128:(i+1)*128] for k in k_range}

        log_means = [np.log(np.mean(sampled_data[k])) for k in k_range]

        log_data = [point for k in k_range for point in np.log(sampled_data[k])]
        log_k = [np.log(k) for k in k_range for point in np.log(sampled_data[k])]

        log_error = [np.std(sampled_data[k], ddof=1) / np.mean(sampled_data[k]) / np.sqrt(len(data[k])) * Z for k in k_range]
        # print(log_error)

        popt, pcov = curve_fit(linear_model, log_k, log_data)
        power.append(popt[0])
        constant.append(popt[1])

        # plt.title('{0} Phase Length ({1}x{1} grid, {2} neighborhood)'.format(phase, size,
        #                                                                      'Von Neuman' if neighborhood == 'vn' else 'Moore'))

    print('log {} = ({:3f} +- {:3f}) log k + {:3f} +- {:3f}'.format(phase, np.mean(power), np.std(power) / np.sqrt(len(power)) * Z, np.mean(constant), np.std(constant) /  np.sqrt(len(power)) * Z))

    log_means = [np.log(np.mean(data[k])) for k in k_range]
    log_error = [np.std(data[k], ddof=1) / np.mean(data[k]) for k in k_range]

    # plt.scatter(log_k + np.random.normal(0, 0.01, len(log_k)), log_data, alpha=0.1)
    plt.plot(log_range, linear_model(log_range, np.mean(power), np.mean(constant)), 'black')
    # plt.plot(log_range, linear_model(log_range, np.mean(power) + np.std(power), np.mean(constant)), 'r')
    # plt.plot(log_range, linear_model(log_range, np.mean(power) - np.std(power), np.mean(constant)), 'b')

    plt.fill_between(log_range, linear_model(log_range, np.mean(power) - np.std(power), np.mean(constant)), linear_model(log_range, np.mean(power) + np.std(power), np.mean(constant)), color='black', alpha=0.3)
    plt.errorbar(log_range, log_means, yerr=log_error, fmt='b.', capsize=3)

    plt.xlabel('log k')
    plt.ylabel('log {} Phase Length'.format(phase))
    os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    plt.savefig(cca_dir + 'plots/phase_lengths/{}_{}_{}.pdf'.format(size, neighborhood, phase.lower()),
                bbox_inches='tight')

    try:
        plt.show()
    except KeyboardInterrupt:
        exit()


def plot(phase, data, neighborhood, exponent, constant):
    if neighborhood == 'moore':
        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':
        k_range = vn_range
        log_range = np.log(vn_range)


    means = np.array([np.mean(data[k]) for k in k_range])
    error = [np.std(data[k], ddof=1) / np.sqrt(len(data[k])) * Z for k in k_range]

    log_means = [np.log(np.mean(data[k])) for k in k_range]

    log_data = [point for k in k_range for point in np.log(data[k])]
    log_k = [np.log(k) for k in k_range for point in np.log(data[k])]

    log_error = [np.std(data[k], ddof=1) / np.mean(data[k]) / np.sqrt(len(data[k])) * Z for k in k_range]
    # print(log_error)

    popt, pcov = curve_fit(linear_model, log_k, log_data)
    print('log {} = ({:3f} +- {:3f}) log k + {:3f} +- {:3f}'.format(phase, popt[0], pcov[0][0], popt[1], pcov[1][1]))

    plt.figure(figsize=(4, 3))
    # plt.yscale('log')
    # plt.xscale('log')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.scatter(log_k + np.random.normal(0, 0.01, len(log_k)), log_data, alpha=0.1)
    # plt.plot(log_range, linear_model(log_range, *popt), 'r')
    # plt.errorbar(log_range, log_means, yerr=log_error, fmt='b.', capsize=3)
    plt.errorbar(k_range, means, yerr=error, fmt='b.', capsize=3)
    plt.plot(k_range, constant * (np.array(k_range) ** exponent))
    # plt.title('{0} Phase Length ({1}x{1} grid, {2} neighborhood)'.format(phase, size,
    #                                                                      'Von Neuman' if neighborhood == 'vn' else 'Moore'))
    plt.xlabel('log k')
    plt.ylabel('log {} Phase Length'.format(phase))
    os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    plt.savefig(cca_dir + 'plots/phase_lengths/{}_{}_{}.pdf'.format(size, neighborhood, phase.lower()),
                bbox_inches='tight')

    try:
        plt.show()
    except KeyboardInterrupt:
        exit()


def plot_grid_sizes(data_256, data_512, data_1024, neighborhood):
    if neighborhood == 'moore':

        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':

        k_range = vn_range
        log_range = np.log(vn_range)

    colors = ["#7235e2",
"#d9a10c",
"#a70018"]


    sizes = [256, 512, 1024]

    axes = []

    plt.figure(figsize=(9, 2.25))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for index, phase in enumerate(['Debris', 'Droplet', 'Defect']):
        if index > 0:
            axes.append(plt.subplot(1, 3, index + 1,  sharey=axes[0]))

        else:
            axes.append(plt.subplot(1, 3, index + 1))

        for i, data in enumerate([data_256[index], data_512[index], data_1024[index]]):
            log_data = [np.log(np.mean(data[k])) for k in data]
            log_error = [np.std(data[k], ddof=1) / np.mean(data[k]) for k in data]

            popt, pcov = curve_fit(linear_model, log_range, log_data)
            # print(pcov)

            plt.plot(log_range + (i-1) / 75, linear_model(log_range, *popt),  color=colors[i], label='{0}x{0}'.format(sizes[i]))
            plt.errorbar(log_range + (i-1) / 75, log_data, yerr=log_error, fmt='.', capsize=3, color=colors[i])

        plt.title('$\\textbf{' + phase + '}$')
        plt.xlabel('$\log k$')
        if index == 0:
            plt.ylabel('$\log$ Phase Length'.format(phase))
            plt.legend(loc='upper left', fontsize=10, title='\\textbf{Grid size}')

        else:
            plt.setp(axes[index].get_yticklabels(), visible=False)

    plt.subplots_adjust(wspace=0)
    os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    plt.savefig(cca_dir + 'plots/phase_lengths/all_sizes_{}.pdf'.format(neighborhood),
                bbox_inches='tight')
    plt.show()


def single(neighborhood, size):
    # Getting back the objects:
    with open(cca_dir + 'pickles/{}_{}.pkl'.format(size, neighborhood), 'rb') as f:
        debris_lengths, droplet_lengths, defect_lengths = pickle.load(f)

    # plot_bootstrap('Debris', debris_lengths, neighborhood)
    # plot_bootstrap('Droplet', droplet_lengths, neighborhood)
    # plot_bootstrap('Defect', defect_lengths, neighborhood)

    plot('Debris', debris_lengths, neighborhood, exponents[neighborhood, 'Debris'], constants[neighborhood, 'Debris'])
    plot('Droplet', droplet_lengths, neighborhood, exponents[neighborhood, 'Droplet'], constants[neighborhood, 'Droplet'])
    plot('Defect', defect_lengths, neighborhood, exponents[neighborhood, 'Defect'], constants[neighborhood, 'Defect'])


def all_noisy(neighborhood, size):
    # Getting back the objects:
    with open(cca_dir + 'pickles/{}_{}.pkl'.format(size, neighborhood), 'rb') as f:
        debris_lengths, droplet_lengths, defect_lengths = pickle.load(f)

    noisy_debris = []
    noisy_droplets = []
    noisy_defects = []

    for i in range(8):
        with open(cca_dir + 'pickles/noisy_{}_{}_{}.pkl'.format(i, size, neighborhood), 'rb') as f:
            x, y, z = pickle.load(f)
            noisy_debris.append(x)
            noisy_droplets.append(y)
            noisy_defects.append(z)

    plot_noisy('Debris', debris_lengths, noisy_debris, neighborhood)
    plot_noisy('Droplet', droplet_lengths, noisy_droplets, neighborhood)
    plot_noisy('Defect', defect_lengths, noisy_defects, neighborhood)


def compare(neighborhood):

    with open(cca_dir + 'pickles/256_{}.pkl'.format(neighborhood), 'rb') as f:
        data_256 = pickle.load(f)

    with open(cca_dir + 'pickles/512_{}.pkl'.format(neighborhood), 'rb') as f:
        data_512 = pickle.load(f)

    with open(cca_dir + 'pickles/1024_{}.pkl'.format(neighborhood), 'rb') as f:
        data_1024 = pickle.load(f)

    # print(debris_128)

    plot_grid_sizes(data_256, data_512, data_1024, neighborhood)

    plt.show()


if __name__ == '__main__':


    # for neighborhood, size in itertools.product(neighborhoods, sizes):
    #     print('Plotting {} {}...'.format(neighborhood, size))
    #     all_noisy(neighborhood, size)
    #
    # for neighborhood, size in itertools.product(neighborhoods, sizes):
    #     print('Plotting {} {}...'.format(neighborhood, size))
    #     single(neighborhood, size)

    for neighborhood in neighborhoods:
        print('Comparing sizes for {}...'.format(neighborhood))
        compare(neighborhood)
