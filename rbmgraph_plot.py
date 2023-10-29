import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

data = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/rbmgraphless_sigma_100_seed_1000')
sigma = data['sigma'].values[0]
cms = [plt.get_cmap('Greens'), plt.get_cmap('Reds'), plt.get_cmap('Oranges'), plt.get_cmap('Blues')]
N2_list = [5, 10, 100, 200]
F_list = [-1, -100, - 1000, -10000]
tau = 0.1
hh = 0.01
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
axs = [ax1, ax2, ax3, ax4]
for n in range(len(N2_list)):
    dd = data.loc[data['N2'] == N2_list[n], :]
    xx = np.arange(len(F_list))
    res_arr = np.array(dd['J_tilda'])
    axs[n].plot(xx, res_arr, color=cms[n](200))
    axs[n].set_xticks(xx)
    axs[n].set_xticklabels(F_list)
    axs[n].set_title('N2=' + str(N2_list[n]))
    axs[n].set_xlabel('F')
    axs[n].set_ylabel('J_tilda')
fig.suptitle('J_tilda values with sigma =' + str(sigma))
plt.subplots_adjust(hspace=0.5)
plt.show()
# plt.yscale('log')
# plt.xscale('log')


