import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

'''
data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value)_Wolfe.csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value)_Wolfe.csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value)_Wolfe.csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value)_Wolfe.csv')

frames = [data1, data2, data3, data4, data5]
result = pd.concat(frames, axis=1)
'''

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value)_Wolfe.csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value)_Wolfe.csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value)_Wolfe.csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value)_Wolfe.csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt5(value)_Wolfe.csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt5(value)_Wolfe.csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt5(value)_Wolfe.csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt5(value)_Wolfe.csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt5(value)_Wolfe.csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt5(value)_Wolfe.csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt5(value)_Wolfe.csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt5(value)_Wolfe.csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt5(value)_Wolfe.csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt5(value)_Wolfe.csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt5(value)_Wolfe.csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt5(value)_Wolfe.csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt5(value)_Wolfe.csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt5(value)_Wolfe.csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt5(value)_Wolfe.csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt5(value)_Wolfe.csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt5(value)_Wolfe.csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt5(value)_Wolfe.csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt5(value)_Wolfe.csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt5(value)_Wolfe.csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt5(value)_Wolfe.csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt5(value)_Wolfe.csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt5(value)_Wolfe.csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt5(value)_Wolfe.csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt5(value)_Wolfe.csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)

print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(-25, 15))
plt.title('MCGD with G(x) Under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt5(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt5(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt5(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt5(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt5(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt5(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt5(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt5(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt5(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt5(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt5(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt5(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt5(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt5(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt5(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt5(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt5(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt5(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt5(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt5(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt5(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt5(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt5(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt5(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt5(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(-25, 15))
plt.title('MCGD with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt9_Wolfe(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt9_Wolfe(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt9_Wolfe(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt9_Wolfe(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt9_Wolfe(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt9_Wolfe(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt9_Wolfe(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt9_Wolfe(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt9_Wolfe(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt9_Wolfe(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt9_Wolfe(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt9_Wolfe(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt9_Wolfe(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt9_Wolfe(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt9_Wolfe(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt9_Wolfe(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt9_Wolfe(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt9_Wolfe(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt9_Wolfe(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt9_Wolfe(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt9_Wolfe(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt9_Wolfe(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt9_Wolfe(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt9_Wolfe(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt9_Wolfe(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt9_Wolfe(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt9_Wolfe(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt9_Wolfe(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt9_Wolfe(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt9_Wolfe(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:102]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[100].mean())
print(result.iloc[100].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*10].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*10].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df['PDE_MC'] = pd.Series([np.full(10, 5.883638663142211)])
df = df.fillna(5.883638663142211)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 8)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(5, 30))
plt.title('MCGD and PDE Percentiles Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt8_value.csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt8_value.csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt8_value.csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt8_value.csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt8_value.csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt8_value.csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt8_value.csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt8_value.csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt8_value.csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt8_value.csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt8_value.csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt8_value.csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt8_value.csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt8_value.csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt8_value.csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt8_value.csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt8_value.csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt8_value.csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt8_value.csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt8_value.csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt8_value.csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt8_value.csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt8_value.csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt8_value.csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt8_value.csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt8_value.csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt8_value.csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt8_value.csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt8_value.csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt8_value.csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[100].mean())
print(result.iloc[100].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*10].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*10].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df['PDE_MC'] = pd.Series([np.full(10, 5.883638663142211)])
df = df.fillna(5.883638663142211)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 8)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(5, 30))
plt.title('SPSA and PDE Percentiles Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(0, 0.5))
plt.title('SPSA')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt1(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt1(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt1(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt1(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt1(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt1(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt1(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt1(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt1(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt1(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt1(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt1(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt1(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt1(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt1(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt1(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt1(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt1(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt1(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt1(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt1(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt1(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt1(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt1(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt1(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt1(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt1(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt1(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt1(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt1(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(0, 0.5))
plt.title('SPSA with G(x)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt6(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt6(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt6(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt6(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt6(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt6(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt6(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt6(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt6(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt6(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt6(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt6(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt6(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt6(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt6(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt6(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt6(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt6(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt6(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt6(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt6(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt6(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt6(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt6(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt6(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt6(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt6(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt6(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt6(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt6(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(0, 0.08))
plt.title('SPSA under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt7(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt7(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt7(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt7(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt7(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt7(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt7(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt7(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt7(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt7(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt7(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt7(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt7(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt7(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt7(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt7(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt7(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt7(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt7(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt7(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt7(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt7(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt7(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt7(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt7(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt7(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt7(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt7(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt7(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(0, 0.15))
plt.title('SPSA with G(x) under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value)(wolfe).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4(value)_Wolfe.csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4(value)_Wolfe.csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4(value)_Wolfe.csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4(value)_Wolfe.csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt4(value)_Wolfe.csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt4(value)_Wolfe.csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt4(value)_Wolfe.csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt4(value)_Wolfe.csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt4(value)_Wolfe.csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt4(value)_Wolfe.csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt4(value)_Wolfe.csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt4(value)_Wolfe.csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt4(value)_Wolfe.csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt4(value)_Wolfe.csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt4(value)_Wolfe.csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt4(value)_Wolfe.csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt4(value)_Wolfe.csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt4(value)_Wolfe.csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt4(value)_Wolfe.csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt4(value)_Wolfe.csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt4(value)_Wolfe.csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt4(value)_Wolfe.csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt4(value)_Wolfe.csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt4(value)_Wolfe.csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt4(value)_Wolfe.csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt4(value)_Wolfe.csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt4(value)_Wolfe.csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt4(value)_Wolfe.csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt4(value)_Wolfe.csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(0, 10))
plt.title('MCGD under Wolfe Conditions')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################

data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt4(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt4(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt4(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt4(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt4(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt4(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt4(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt4(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt4(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt4(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt4(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt4(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt4(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt4(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt4(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt4(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt4(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt4(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt4(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt4(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt4(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt4(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt4(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt4(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt4(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt4(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt4(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt4(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt4(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt4(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1001]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1000].mean())
print(result.iloc[1000].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 9)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
# plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
ax.set(ylim=(0, 10))
plt.title('MCGD')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt5(value)(wolfe).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt5(value)_Wolfe.csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt5(value)_Wolfe.csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt5(value)_Wolfe.csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt5(value)_Wolfe.csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt5(value)_Wolfe.csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt5(value)_Wolfe.csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt5(value)_Wolfe.csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt5(value)_Wolfe.csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt5(value)_Wolfe.csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt5(value)_Wolfe.csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt5(value)_Wolfe.csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt5(value)_Wolfe.csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt5(value)_Wolfe.csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt5(value)_Wolfe.csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt5(value)_Wolfe.csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt5(value)_Wolfe.csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt5(value)_Wolfe.csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt5(value)_Wolfe.csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt5(value)_Wolfe.csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt5(value)_Wolfe.csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt5(value)_Wolfe.csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt5(value)_Wolfe.csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt5(value)_Wolfe.csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt5(value)_Wolfe.csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt5(value)_Wolfe.csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt5(value)_Wolfe.csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt5(value)_Wolfe.csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt5(value)_Wolfe.csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt5(value)_Wolfe.csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1002]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1001].mean())
print(result.iloc[1001].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df['PDE_MC'] = pd.Series([np.full(8, -19.7937051229088)])
df = df.fillna(-19.7937051229088)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 8)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(-25, 15))
plt.title('MCGD and PDE Percentiles Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################


data1 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data/opt7(value).csv')
data2 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data2/opt7(value).csv')
data3 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data3/opt7(value).csv')
data4 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data4/opt7(value).csv')
data5 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data5/opt7(value).csv')
data6 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data6/opt7(value).csv')
data7 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data7/opt7(value).csv')
data8 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data8/opt7(value).csv')
data9 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data9/opt7(value).csv')
data10 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data10/opt7(value).csv')
data11 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data11/opt7(value).csv')
data12 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data12/opt7(value).csv')
data13 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data13/opt7(value).csv')
data14 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data14/opt7(value).csv')
data15 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data15/opt7(value).csv')
data16 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data16/opt7(value).csv')
data17 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data17/opt7(value).csv')
data18 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data18/opt7(value).csv')
data19 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data19/opt7(value).csv')
data20 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data20/opt7(value).csv')
data21 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data21/opt7(value).csv')
data22 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data22/opt7(value).csv')
data23 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data23/opt7(value).csv')
data24 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data24/opt7(value).csv')
data25 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data25/opt7(value).csv')
data26 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data26/opt7(value).csv')
data27 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data27/opt7(value).csv')
data28 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data28/opt7(value).csv')
data29 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data29/opt7(value).csv')
data30 = pd.read_csv('D:/purdue/RBM/Sim3/Python/data30/opt7(value).csv')

frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15,data16, data17, data18, data19, data20, data21, data22, data23, data24, data25, data26, data27, data28, data29, data30]
result = pd.concat(frames, axis=1)[0:1002]
result = result.reset_index(drop=True)
result = result.fillna(0)
print(result.iloc[1001].mean())
print(result.iloc[1001].median())

for i in range(1, 11):
    if i == 1:
        df = pd.DataFrame(result.iloc[i*100].sort_values()).transpose()
    else:
        df = df.append(result.iloc[i*100].sort_values())

df.columns = list(range(1, 31))
# df = df.iloc[:, [6, 12, 18, 24]]
# df.columns = ['20%', '40%', '60%', '80%']
# df = df.iloc[:, [3, 6, 9, 12, 15, 18, 21, 24, 27]]
# df.columns = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
df = df.iloc[:, [3, 6, 12, 15, 18, 24, 27]]
df.columns = ['10%', '20%', '40%', '50%', '70%', '80%', '90%']
df['PDE_MC'] = pd.Series([np.full(7, -19.7937051229088)])
df = df.fillna(-19.7937051229088)
df1 = df.transpose()
df2 = df1.stack().reset_index()
df2.columns = ['Percentile', 'Iteration', 'Function value']
pal = sns.color_palette("rocket", 8)
sns.set(font_scale=1.5)
sns.set_palette(pal)
ax = sns.lineplot(x='Iteration', y='Function value', hue='Percentile', data=df2)
ax.set(ylim=(-23, 10))
plt.title('SPSA and PDE Percentiles Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#############################################################################
