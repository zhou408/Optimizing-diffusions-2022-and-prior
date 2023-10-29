import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns


# plot function value wrt iteration
'''
index = index=pd.RangeIndex(1, 1001, 1)
# dim = 10
# ite = np.arange(1, dim+1)
# value = np.random.uniform(0, 2, dim)
# data = pd.Series(np.random.randn(10), index=pd.RangeIndex(1, 11, 1))
df = pd.DataFrame({'iteration': ite, 'x': x, 'y': y}, index)
# list_data = [df['x'], df['y']]
pal = sns.color_palette("mako_r", 6)
sns.set_palette(pal)
sns.set_style("darkgrid")
# ax = sns.lineplot(data=list_data)
# ax = sns.lineplot(data=df)
sns.lineplot(x='Year', y='Function value', hue='variable', data=pd.melt(df, ['ite']))
#ax.set(xlabel='iteration', ylabel='function value')

# df1 = pd.DataFrame(np.c_[x, y])
#ax = sns.lineplot(data=df1)

#ax.plot()
# df.plot()
'''

ite = np.arange(1, 1001)
x = np.random.randn(1000)
y = np.sin(x)
df2 = pd.DataFrame(columns=ite, data=[x, y])
df2 = df2.set_index([["Method1", "Method2"]])
df3 = df2.stack().reset_index()
df3.columns = ['Method', 'Iteration', 'Function values']
pal = sns.color_palette("mako_r", 6)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function values', hue='Method', data=df3)
# ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)

opt = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt(value).csv')
opt1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1(value).csv')
opt_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6(value).csv')
opt1_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
opt4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value).csv')
opt5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value).csv')
opt4_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv')
opt5_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')

opt = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt(value).csv')
opt.columns = ['Function values']
opt.to_csv("D:/purdue/RBM/Sim3/Python/data/opt1(value).csv", index=None)

opt1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1(value).csv')
opt1.index = opt1.index + 1
opt1.loc[0] = opt1.columns.to_list()[0]
opt1.sort_index(inplace=True)
opt1.columns = ['Function values']
opt1.to_csv("D:/purdue/RBM/Sim3/Python/data/opt1(value).csv", index=None)

opt_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6(value).csv')
opt_wolfe.index = opt_wolfe.index + 1
opt_wolfe.loc[0] = opt_wolfe.columns.to_list()[0]
opt_wolfe.sort_index(inplace=True)
opt_wolfe.columns = ['Function values']
opt_wolfe.to_csv("D:/purdue/RBM/Sim3/Python/data/opt6(value).csv", index=None)

opt1_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
opt1_wolfe.index = opt1_wolfe.index + 1
opt1_wolfe.loc[0] = opt1_wolfe.columns.to_list()[0]
opt1_wolfe.sort_index(inplace=True)
opt1_wolfe.columns = ['Function values']
opt1_wolfe.to_csv("D:/purdue/RBM/Sim3/Python/data/opt7(value).csv", index=None)


opt4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value).csv')
opt4.index = opt4.index + 1
opt4.loc[0] = opt4.columns.to_list()[0]
opt4.sort_index(inplace=True)
opt4.columns = ['Function values']
opt4.to_csv("D:/purdue/RBM/Sim3/Python/data/opt4(value).csv", index=None)


opt5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value).csv')
opt5.index = opt5.index + 1
opt5.loc[0] = opt5.columns.to_list()[0]
opt5.sort_index(inplace=True)
opt5.columns = ['Function values']
opt5.to_csv("D:/purdue/RBM/Sim3/Python/data/opt5(value).csv", index=None)


opt4_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv')
opt4_wolfe.index = opt4_wolfe.index + 1
opt4_wolfe.loc[0] = opt4_wolfe.columns.to_list()[0]
opt4_wolfe.sort_index(inplace=True)
opt4_wolfe.columns = ['Function values']
opt4_wolfe.to_csv("D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv", index=None)

opt5_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')
opt5_wolfe.index = opt5_wolfe.index + 1
opt5_wolfe.loc[0] = opt5_wolfe.columns.to_list()[0]
opt5_wolfe.sort_index(inplace=True)
opt5_wolfe.columns = ['Function values']
opt5_wolfe.to_csv("D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv", index=None)

