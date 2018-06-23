import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

lines = []
with open('data/vn_neighbor_diff_data.csv') as f:
    lines = [line.split(',') for line in f.readlines()[1:]]
    
data = {int(line[0]): [] for line in lines}
for line in lines:
    data[int(line[0])].append(list(map(int, line[2:1000])))

trial = data[10][0]
averages = {types: np.mean(data[types], axis=0) for types in data}
uncertainties = {types: (np.var(data[types], ddof=1, axis=0) / len(data))**0.5 for types in data}


for types in range(10, 20):

    plot_data = averages[types]

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].errorbar(np.arange(len(plot_data)), plot_data, yerr=uncertainties[types], label='{}'.format(types))

    # plt.legend(loc='lower right')
    plt.ylim(ymin=0, ymax=256**2+1000)
    plt.xlabel('Step')
    plt.ylabel('Cells Changed')
    plt.legend(loc='lower right', title='k')

    x = np.arange(len(plot_data))
    y = plot_data

    f2 = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, 300, num=100000)


    axarr[1].plot(np.arange(len(plot_data)), savgol_filter(np.gradient(np.gradient(plot_data, 1), 1), 51, 2, mode='nearest'))
    axarr[1].plot(np.arange(len(plot_data)), np.gradient(np.gradient(plot_data, 1), 1))

    plt.ylim(ymin=-20, ymax=20)
    plt.show()