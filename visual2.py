import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

###########################################
opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['SPSA']
df['Third'] = opt_3['SPSA']
df['Fourth'] = opt_4['SPSA']
df['Fifth'] = opt_5['SPSA']
df['Average'] = df.mean(numeric_only=True, axis=1)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 1))
plt.title('SPSA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt1(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt1(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt1(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt1(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['SPSA_G(x)']
df['Third'] = opt_3['SPSA_G(x)']
df['Fourth'] = opt_4['SPSA_G(x)']
df['Fifth'] = opt_5['SPSA_G(x)']
df['Average'] = df.mean(numeric_only=True, axis=1)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-0.5, 20))
plt.title('SPSA with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt6(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt6(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt6(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt6(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['SPSA_Wolfe']
df['Third'] = opt_3['SPSA_Wolfe']
df['Fourth'] = opt_4['SPSA_Wolfe']
df['Fifth'] = opt_5['SPSA_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 0.2))
plt.title('SPSA Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt7(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt7(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt7(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt7(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['SPSA_G(x)_Wolfe']
df['Third'] = opt_3['SPSA_G(x)_Wolfe']
df['Fourth'] = opt_4['SPSA_G(x)_Wolfe']
df['Fifth'] = opt_5['SPSA_G(x)_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-0.5, 1))
plt.title('SPSA with G(x) Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################


opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['MCGD']
df['Third'] = opt_3['MCGD']
df['Fourth'] = opt_4['MCGD']
df['Fifth'] = opt_5['MCGD']
df['Average'] = df.mean(numeric_only=True, axis=1)

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 30))
plt.title('MCGD')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################



opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['MCGD_G(x)']
df['Third'] = opt_3['MCGD_G(x)']
df['Fourth'] = opt_4['MCGD_G(x)']
df['Fifth'] = opt_5['MCGD_G(x)']
df['Average'] = df.mean(numeric_only=True, axis=1)

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-23, 15))
plt.title('MCGD with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4(value)_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4(value)_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4(value)_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4(value)_Wolfe.csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['MCGD_Wolfe']
df['Third'] = opt_3['MCGD_Wolfe']
df['Fourth'] = opt_4['MCGD_Wolfe']
df['Fifth'] = opt_5['MCGD_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 30))
plt.title('MCGD Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value)_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value)_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value)_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value)_Wolfe.csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['MCGD_G(x)_Wolfe']
df['Third'] = opt_3['MCGD_G(x)_Wolfe']
df['Fourth'] = opt_4['MCGD_G(x)_Wolfe']
df['Fifth'] = opt_5['MCGD_G(x)_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-23, 15))
plt.title('MCGD with G(x) Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt9_Wolfe(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt9_Wolfe(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt9_Wolfe(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt9_Wolfe(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt9_Wolfe(value).csv')

opt_1.columns = ['First']
df = opt_1[0:100]
df['Second'] = opt_2['MCGD_pde(Wolfe)'][0:100]
df['Third'] = opt_3['MCGD_pde(Wolfe)'][0:100]
df['Fourth'] = opt_4['MCGD_pde(Wolfe)'][0:100]
df['Fifth'] = opt_5['MCGD_pde(Wolfe)'][0:100]
df['Average'] = df.mean(numeric_only=True, axis=1)

# df['PDE'] = pd.Series([0.759442448116539 for x in range(len(df.index))])
df['PDE_MC'] = pd.Series([5.883638663142211 for x in range(len(df.index))])
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 7)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 50))
plt.title('MCGD and PDE Function Value Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt8_value.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt8_value.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt8_value.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt8_value.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt8_value.csv')

opt_1.columns = ['First']
df = opt_1[0:100]
df['Second'] = opt_2['SPSA_pde'][0:100]
df['Third'] = opt_3['SPSA_pde'][0:100]
df['Fourth'] = opt_4['SPSA_pde'][0:100]
df['Fifth'] = opt_5['SPSA_pde'][0:100]
df['Average'] = df.mean(numeric_only=True, axis=1)

# df['PDE'] = pd.Series([0.759442448116539 for x in range(len(df.index))])
df['PDE_MC'] = pd.Series([5.883638663142211 for x in range(len(df.index))])
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 7)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(0, 50))
plt.title('SPSA and PDE Function Value Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################


opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt7(value).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt7(value).csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt7(value).csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt7(value).csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt7(value).csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['SPSA_G(x)_Wolfe']
df['Third'] = opt_3['SPSA_G(x)_Wolfe']
df['Fourth'] = opt_4['SPSA_G(x)_Wolfe']
df['Fifth'] = opt_5['SPSA_G(x)_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)
df['PDE_MC'] = pd.Series([-19.7937051229088 for x in range(len(df.index))])


df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 7)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-25, 25))
plt.title('SPSA and PDE Function Value Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt5(value)_wolfe.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt5(value)_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt5(value)_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt5(value)_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt5(value)_Wolfe.csv')

opt_1.columns = ['First']
df = opt_1
df['Second'] = opt_2['MCGD_G(x)_Wolfe']
df['Third'] = opt_3['MCGD_G(x)_Wolfe']
df['Fourth'] = opt_4['MCGD_G(x)_Wolfe']
df['Fifth'] = opt_5['MCGD_G(x)_Wolfe']
df['Average'] = df.mean(numeric_only=True, axis=1)
df['PDE_MC'] = pd.Series([-19.7937051229088 for x in range(len(df.index))])

df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Iteration', 'Function value']
pal = sns.color_palette("husl", 7)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Run', data=df2)
ax.set(ylim=(-23, 15))
plt.title('MCGD and PDE Function Value Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
###########################################