import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

data1 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.1plans_init-10-10+0_lambda=1000_M=1000_ite100_seed_1000')

data2 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.1plans_init-10-10+0_lambda=1000_M=10_ite100_seed_1000')

data3 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.1plans_init-10-10-10_lambda=1000_M=10_ite100_seed_1000')

df1 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.2plans_init+0+0+0_b=-0.005_t=1_ite100_seed_1000')

df2 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.2plans_init+0+0+0_b=-0.01_t=1_ite100_seed_1000')

df3 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.2plans_init+0+0+0_b=-0.1_t=1_ite100_seed_1000')

df4 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.2plans_init+0+0+0_b=-0.5_t=1_ite100_seed_1000')

df5 = pd.read_csv('/Users/zihezhou/Desktop/purdue/RBM Code 2020/Python/venv/RBMexperiments2021/7.2plans_init+0+0+0_b=-1_t=1_ite100_seed_1000')

dfs = [df1, df2, df3, df4, df5]

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
dd = data1
arr1 = np.arange(1, dd.shape[0]+1) #ite
arr2 = dd['trueval']
ax1.plot(arr1, arr2, marker=".", color='Pink')
plt.xlabel('k', fontsize=16)
ax1.set_ylabel('values')
# plt.yscale('log')
# plt.xscale('log')
plt.title('Value v.s iteration k with various settings', fontsize=14)
plt.show()

# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# cms = [plt.get_cmap('Greys'), plt.get_cmap('Blues'), plt.get_cmap('Purples'), plt.get_cmap('Oranges'), plt.get_cmap('Greens')]
# b_list = [-0.005, -0.01, -0.1, -0.5, -1]
# for i in range(len(dfs)):
#     dd = dfs[i]
#     arr1 = np.arange(1, dd.shape[0]+1) #ite
#     arr2 = dd['trueval']
#     ax2.plot(arr1, arr2, marker=".", color=cms[i](200), label='b=' + str(b_list[i]))
# ax2.legend(bbox_to_anchor=(1.001, 1), loc='best', fontsize=10)
# plt.xlabel('k', fontsize=16)
# ax2.set_ylabel('values')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.title('Value v.s iteration k with various settings', fontsize=16)
# plt.show()