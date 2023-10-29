import matplotlib as plt
import pandas as pd
import numpy as np
import seaborn as sns


opt = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt(value).csv')
opt1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1(value).csv')
opt_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6(value).csv')
opt1_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
opt4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value).csv')
opt5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value).csv')
opt4_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv')
opt5_wolfe = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')

opt.columns = ['SPSA']
opt1.columns = ['SPSA_G(x)']
opt_wolfe.columns = ['SPSA_Wolfe']
opt1_wolfe.columns = ['SPSA_G(x)_Wolfe']
opt4.columns = ['MCGD']
opt5.columns = ['MCGD_G(x)']
opt4_wolfe.columns = ['MCGD_Wolfe']
opt5_wolfe.columns = ['MCGD_G(x)_Wolfe']

df = opt
df['SPSA_Wolfe'] = opt_wolfe['SPSA_Wolfe']
df['MCGD'] = opt4['MCGD']
df['MCGD_Wolfe'] = opt4_wolfe['MCGD_Wolfe']


df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Method', 'Iteration', 'Function values']
pal = sns.color_palette("mako_r", 6)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function values', hue='Method', data=df2)

dfgx = opt1
dfgx['SPSA_G(x)_Wolfe'] = opt1_wolfe['SPSA_G(x)_Wolfe']
dfgx['MCGD_G(x)'] = opt5['MCGD_G(x)']
dfgx['MCGD_G(x)_Wolfe'] = opt5_wolfe['MCGD_G(x)_Wolfe']

dfgx1 = dfgx.transpose()
dfgx2 = dfgx1.stack().reset_index()
dfgx2.columns = ['Method', 'Iteration', 'Function values']
pal = sns.color_palette("mako_r", 6)
sns.set_palette(pal)
ax1 = sns.lineplot(x='Iteration', y='Function values', hue='Method', data=dfgx2)

ax = sns.lineplot(x='Iteration', y='Function values', hue='Method', data=df2)

plot = sns.lineplot(data=opt)
plot.set(xlabel='iteration', ylabel='function value')

plot1 = sns.lineplot(data=opt1)
plot1.set(xlabel='iteration', ylabel='function value')

plot2 = sns.lineplot(data=opt_wolfe)
plot2.set(xlabel='iteration', ylabel='function value')

plot3 = sns.lineplot(data=opt1_wolfe)
plot3.set(xlabel='iteration', ylabel='function value')

plot4 = sns.lineplot(data=opt4)
plot4.set(xlabel='iteration', ylabel='function value')

plot5 = sns.lineplot(data=opt5)
plot5.set(xlabel='iteration', ylabel='function value')

plot6 = sns.lineplot(data=opt4_wolfe)
plot6.set(xlabel='iteration', ylabel='function value')

plot7 = sns.lineplot(data=opt5_wolfe)
plot7.set(xlabel='iteration', ylabel='function value')