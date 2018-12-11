import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import itertools
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


cca_dir = '/Users/tomlinsonk/Projects/Research/cca/'

neighborhoods = ['vn', 'moore']
sizes = [128, 256, 512]

vn_range = range(7, 16)
moore_range = range(11, 20)

def linear_model(x, m, b):
    return m * x + b



# ax = plt.axes()
# ax.set_yscale('log')
# ax.set_xscale('log')
# 
def plot(phase, data, neighborhood):
    if neighborhood == 'moore':
        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':
        k_range = vn_range
        log_range = np.log(vn_range)

    log_means = [np.log(np.mean(data[k])) for k in k_range]

    for k in k_range:
        for point in data[k]:
            if point < 0:
                print('AH', point)

    

    log_data = [point for k in k_range for point in np.log(data[k])]
    log_k = [np.log(k) for k in k_range for point in np.log(data[k])]

    log_error = [np.std(data[k], ddof=1) / np.mean(data[k]) for k in k_range]
    # print(log_error)


    popt, pcov = curve_fit(linear_model, log_k, log_data)
    print('log {} = ({:3f} +- {:3f}) log k + {:3f} +- {:3f}'.format(phase, popt[0], pcov[0][0], popt[1], pcov[1][1]))
    
    plt.figure(figsize=(4, 3))
    plt.scatter(log_k + np.random.normal(0, 0.01, len(log_k)), log_data, alpha=0.1)
    plt.plot(log_range, linear_model(log_range, *popt))
    plt.errorbar(log_range, log_means, yerr=log_error)
    # plt.title('{0} Phase Length ({1}x{1} grid, {2} neighborhood)'.format(phase, size, 'Von Neuman' if neighborhood == 'vn' else 'Moore'))
    plt.xlabel('log k')
    plt.ylabel('log Phase Length')
    os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    plt.savefig(cca_dir + 'plots/phase_lengths/{}_{}_{}.pdf'.format(size, neighborhood, phase.lower()), bbox_inches='tight')

    plt.show()

def plot_grid_sizes(phase, data_128, data_256, data_512, neighborhood):
    if neighborhood == 'moore':
        k_range = moore_range
        log_range = np.log(moore_range)
    elif neighborhood == 'vn':
        k_range = vn_range
        log_range = np.log(vn_range)

    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(4, 3))

    sizes = [128, 256, 512]

    for i, data in enumerate([data_128, data_256, data_512]):
        log_data = [np.log(np.mean(data[k])) for k in data]
        log_error = [np.std(data[k], ddof=1) / len(data[k]) / np.mean(data[k]) for k in data]

        popt, pcov = curve_fit(linear_model, log_range, log_data)
        print(pcov)
        
        plt.plot(log_range + (i - 1) / 100, linear_model(log_range, *popt), color=colors[i], label='${0}\\times{0}$'.format(sizes[i]))
        plt.errorbar(log_range + (i - 1) / 100, log_data, yerr=log_error, color=colors[i])

    plt.xlabel('log k')
    plt.ylabel('log {} Phase Length'.format(phase))
    if phase == 'Defect':
        plt.legend(loc='lower right', fontsize=10)
    os.makedirs(cca_dir + 'plots/phase_lengths/', exist_ok=True)
    plt.savefig(cca_dir + 'plots/phase_lengths/all_sizes_{}_{}.pdf'.format(neighborhood, phase.lower()), bbox_inches='tight')

    plt.show()

def single(neighborhood, size):

    # Getting back the objects:
    with open(cca_dir + 'pickles/{}_{}.pkl'.format(size, neighborhood), 'rb') as f: 
        debris_lengths, droplet_lengths, defect_lengths = pickle.load(f)

    plot('Debris', debris_lengths, neighborhood)
    plot('Droplet', droplet_lengths, neighborhood)
    plot('Defect', defect_lengths, neighborhood)

def compare(neighborhood):
    with open(cca_dir + 'pickles/128_{}.pkl'.format(neighborhood), 'rb') as f: 
        debris_128, droplet_128, defect_128 = pickle.load(f)

    with open(cca_dir + 'pickles/256_{}.pkl'.format(neighborhood), 'rb') as f: 
        debris_256, droplet_256, defect_256 = pickle.load(f)

    with open(cca_dir + 'pickles/512_{}.pkl'.format(neighborhood), 'rb') as f: 
        debris_512, droplet_512, defect_512 = pickle.load(f)

    print(debris_128)

    plot_grid_sizes('Debris', debris_128, debris_256, debris_512, neighborhood)
    plot_grid_sizes('Droplet', droplet_128, droplet_256, droplet_512, neighborhood)
    plot_grid_sizes('Defect', defect_128, defect_256, defect_512, neighborhood)

if __name__ == '__main__':
    for neighborhood, size in itertools.product(neighborhoods, sizes):
        print('Plotting {} {}...'.format(neighborhood, size))
        single(neighborhood, size)

    # for neighborhood in neighborhoods:
    #     print('Comparing sizes for {}...'.format(neighborhood))
    #     compare(neighborhood)
