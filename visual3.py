import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({'First': data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-1000, 10))
plt.title('SPSA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt1.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt1.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt1.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt1.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA_G(x).csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-2000, 10))
plt.title('SPSA with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt6.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt6.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt6.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt6.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA_Wolfe.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-5500, 10))
plt.title('SPSA Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt7.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt7.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt7.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt7.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA_G(x)_Wolfe.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-10000, 10))
plt.title('SPSA with G(x) Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-20, 10))
plt.title('MCGD')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD_G(x).csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-20, 12))
plt.title('MCGD with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(Wolfe).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4_Wolfe.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD_Wolfe.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-40, 0))
plt.title('MCGD Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(Wolfe).csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5_Wolfe.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]


df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD_G(x)_Wolfe.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 5)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-100, 20))
plt.title('MCGD with G(x) Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt9_Wolfe.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt9_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt9_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt9_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt9_Wolfe.csv')
opt_6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/T5d1_drift.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]

data6 = [-7.361362529812827127e-01, -0.991546, -0.964050, -0.957955, -0.937242]

df = pd.DataFrame({ 'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df['PDE'] = data6
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD_PDE.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-5, 1))
plt.title('MCGD and PDE Optimal Drift Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt8.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt8.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt8.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt8.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt8.csv')
opt_6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/T5d1_drift.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]

data6 = [-7.361362529812827127e-01, -0.991546, -0.964050, -0.957955, -0.937242]

df = pd.DataFrame({'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df['PDE'] = data6
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA_PDE.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-10, 1))
plt.title('SPSA and PDE Optimal Drift Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt5_Wolfe.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt5_Wolfe.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt5_Wolfe.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt5_Wolfe.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt5_Wolfe.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]

data6 = [-999.999, -999.999, -999.999, -999.999, 9.999]

df = pd.DataFrame({'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df['PDE'] = data6
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/MCGD_PDE2.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-100, 10))
plt.title('MCGD and PDE Optimal Drift Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
##############################################

opt_1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt7.csv')
opt_2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt7.csv')
opt_3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt7.csv')
opt_4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt7.csv')
opt_5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt7.csv')

data1 = opt_1.iloc[1].values[0]
data1 = data1.strip('][').split(' ')
data1 = list(filter(None, data1))
data1 = [float(i) for i in data1]

data2 = opt_2.iloc[1].values[0]
data2 = data2.strip('][').split(' ')
data2 = list(filter(None, data2))
data2 = [float(i) for i in data2]

data3 = opt_3.iloc[1].values[0]
data3 = data3.strip('][').split(' ')
data3 = list(filter(None, data3))
data3 = [float(i) for i in data3]

data4 = opt_4.iloc[1].values[0]
data4 = data4.strip('][').split(' ')
data4 = list(filter(None, data4))
data4 = [float(i) for i in data4]

data5 = opt_5.iloc[1].values[0]
data5 = data5.strip('][').split(' ')
data5 = list(filter(None, data5))
data5 = [float(i) for i in data5]

data6 = [-999.999, -999.999, -999.999, -999.999, 9.999]

df = pd.DataFrame({'First':data1})
df['Second'] = data2
df['Third'] = data3
df['Fourth'] = data4
df['Fifth'] = data5
df['PDE'] = data6
df1 = df.transpose()
df1.columns = ['1', '2', '3', '4', '5']
df1.to_csv("D:/purdue/RBM/Sim3/Python/drift/SPSA_PDE2.csv", index=None)
df2 = df1.stack().reset_index()
df2.columns = ['Run', 'Time step', 'Optimal drift']
pal = sns.color_palette("husl", 6)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Time step', y='Optimal drift', hue='Run', data=df2)
ax.set(ylim=(-2000, 10))
plt.title('SPSA and PDE Optimal Drift Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)