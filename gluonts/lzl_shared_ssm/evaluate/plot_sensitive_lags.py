import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 10), sharey=True)
plt.ylim(88, 98)
my_y_ticks = np.arange(88, 99, 2)
plt.yticks(my_y_ticks)
plt.tick_params(labelsize=15)

crypto_90_5 = [[91.69, 90.52, 90.42, 90.52, 90.96, 91.86],
        [90.16, 91.39, 90.79, 90.83, 90.61, 91.9],
        [91.19, 91.21, 91.15, 91.54, 90.74, 91.54],
        [92.04, 91.75, 91.74, 91.04, 91.26, 91.75],
        [89.28, 90.74, 90.83, 92.21, 90.87, 90.87]]
crypto_90_5 =  np.reshape(np.array(crypto_90_5), (5*6))
crypto_90_5 = pd.DataFrame(crypto_90_5, columns=["ACC"])
crypto_90_5["lag"] = [5]*6 + [6]*6 + [7]*6 + [8]*6 + [9]*6

crypto_60_3 = [[92.57,91.53, 91.64, 91.70, 93.24, 92.57],
        [92.52, 92.52 ,92.47 ,92.37 ,92.17 ,92.07],
        [91.62, 92.17, 92.89, 92.65, 93.24, 91.23 ],
        [93.73,92.83,93.24,92.81,91.57,91.56 ],
        [91.94, 92.38, 93.14, 91.71, 93.27,92.49]]
crypto_60_3 =  np.reshape(np.array(crypto_60_3), (5*6))
crypto_60_3 = pd.DataFrame(crypto_60_3, columns=["ACC"])
crypto_60_3["lag"] = [3]*6 + [4]*6 + [5]*6 + [6]*6 + [7]*6

crypto_30_1 = [[92.18,93.02,93.07,91.91,93.19,93.69],
        [91.96, 91.95, 94.02, 93.96, 93.77, 93.97],
        [93.54, 93.02, 93.24, 94.02, 93.00,93.37],
        [93.36, 93.89, 93.29, 93.84, 92.73, 92.98],
        [91.11, 93.38, 93.63, 93.13, 92.01, 93.10]]
crypto_30_1 =  np.reshape(np.array(crypto_30_1), (5*6))
crypto_30_1 = pd.DataFrame(crypto_30_1, columns=["ACC"])
crypto_30_1["lag"] = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6

sns.set_style("whitegrid")
sns.boxplot(x=crypto_90_5["lag"], y=crypto_90_5["ACC"], ax = ax[2], color="skyblue")
ax[2].set_title(r"90 $\rightarrow$ 5", fontsize=25)
ax[2].set_xlabel(r"$lags$",fontsize=23)
ax[2].set_ylabel("",fontsize=1)
ax[2].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=crypto_60_3["lag"], y=crypto_60_3["ACC"], ax = ax[1], color="skyblue")
ax[1].set_title(r"60 $\rightarrow$ 3", fontsize=25)
ax[1].set_xlabel(r"$lags$",fontsize=23)
ax[1].set_ylabel("",fontsize=1)
ax[1].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=crypto_30_1["lag"], y=crypto_30_1["ACC"], ax = ax[0], color="skyblue")
ax[0].set_title(r"30 $\rightarrow$ 1", fontsize=25)
ax[0].set_xlabel(r"$lags$",fontsize=23)
ax[0].set_ylabel("Accuracy",fontsize=23)
ax[0].tick_params(labelsize = 20, grid_alpha=0.5)
fig.tight_layout()
plt.savefig('crypto_sen_lag.pdf',format = 'pdf',bbox_inches='tight')
plt.close()
# exit()


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 10), sharey=True)
plt.ylim(88, 98)
my_y_ticks = np.arange(88, 99, 2)
plt.yticks(my_y_ticks)
plt.tick_params(labelsize=15)
indice_90_5 = [[94.57, 94.28, 94.31, 94.4, 94.52, 94.27],
        [94.5, 94.34, 94.55, 94.0, 94.13, 94.25],
        [93.63, 94.33, 94.09, 93.97, 94.44, 94.28],
        [93.81, 93.92, 93.85, 93.83, 91.59, 94.48],
        [94.5, 94.09, 92.49, 94.24, 94.64, 94.14]]
indice_90_5 = np.reshape(np.array(indice_90_5), (5*6))
indice_90_5 = pd.DataFrame(indice_90_5, columns=["ACC"])
indice_90_5["lag"] = [5]*6 + [6]*6 + [7]*6 + [8]*6 + [9]*6

indice_60_3 = [[95.2, 94.51, 95.05, 94.52, 95.22, 93.98],
        [95.18, 94.88, 95.2, 94.76, 94.33, 95.31],
        [94.72, 94.77, 94.54, 94.99, 95.07, 94.84],
        [94.53, 94.91, 95.23, 94.85, 94.83, 94.99],
        [95.12, 94.28, 94.5, 94.58, 95.29, 94.71]]
indice_60_3 = np.reshape(np.array(indice_60_3), (5*6))
indice_60_3 = pd.DataFrame(indice_60_3, columns=["ACC"])
indice_60_3["lag"] = [3]*6 + [4]*6 + [5]*6 + [6]*6 + [7]*6

indice_30_1 = [[96.04, 95.87, 96.37, 96.06, 96.47, 96.15],
        [96.53, 96.19, 96.28, 96.0, 96.3, 96.41],
        [96.32, 96.45, 96.12, 96.35, 96.29, 96.33],
        [96.25, 96.24, 95.97, 96.1, 96.16, 96.28],
        [96.26, 96.11, 96.28, 95.71, 96.42, 96.19]]
indice_30_1 = np.reshape(np.array(indice_30_1), (5*6))
indice_30_1 = pd.DataFrame(indice_30_1, columns=["ACC"])
indice_30_1["lag"] = [1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6

sns.set_style("whitegrid")
sns.boxplot(x=indice_90_5["lag"], y=indice_90_5["ACC"], ax = ax[2], color="skyblue")
ax[2].set_title(r"90 $\rightarrow$ 5", fontsize=25)
ax[2].set_xlabel(r"$lags$",fontsize=23)
ax[2].set_ylabel("",fontsize=1)
ax[2].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=indice_60_3["lag"], y=indice_60_3["ACC"], ax = ax[1], color="skyblue")
ax[1].set_title(r"60 $\rightarrow$ 3", fontsize=25)
ax[1].set_xlabel(r"$lags$",fontsize=23)
ax[1].set_ylabel("",fontsize=1)
ax[1].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=indice_30_1["lag"], y=indice_30_1["ACC"], ax = ax[0], color="skyblue")
ax[0].set_title(r"30 $\rightarrow$ 1", fontsize=25)
ax[0].set_xlabel(r"$lags$",fontsize=23)
ax[0].set_ylabel("Accuracy",fontsize=23)
ax[0].tick_params(labelsize = 20, grid_alpha=0.5)
fig.tight_layout()
plt.savefig('indice_sen_lag.pdf',format = 'pdf',bbox_inches='tight')
plt.close()