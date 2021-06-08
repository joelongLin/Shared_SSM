import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 10), sharey=True)
plt.ylim(90, 97)
my_y_ticks = np.arange(90, 97, 2)
plt.yticks(my_y_ticks)
plt.tick_params(labelsize=15)


crypto_90_5 = [[91.69, 90.52, 90.42, 90.52, 90.96, 91.86],
        [91.99, 91.43, 91.78, 90.76, 91.55, 91.26],
        [90.12, 91.53, 91.88, 91.57, 90.85, 92.08],
        [90.47, 91.38, 91.69, 91.28, 91.4, 90.7]]
crypto_90_5 =  np.reshape(np.array(crypto_90_5), (4*6))
crypto_90_5 = pd.DataFrame(crypto_90_5, columns=["ACC"])
crypto_90_5["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6

crypto_60_3 = [[93.18, 93.55, 93.3, 93.66, 91.74, 90.37],
        [92.28, 92.09, 91.52, 92.05, 92.72, 91.99],
        [92.68, 91.98, 92.97, 92.45, 91.98, 91.84],
        [91.53, 92.05, 91.92, 92.2, 92.73, 93.58]]
crypto_60_3 =  np.reshape(np.array(crypto_60_3), (4*6))
crypto_60_3 = pd.DataFrame(crypto_60_3, columns=["ACC"])
crypto_60_3["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6

crypto_30_1 = [[92.18, 93.02, 93.07, 91.91, 93.19, 93.69],
        [94.67, 94.65, 94.3, 94.41, 94.55, 94.5],
        [94.76, 94.71, 94.37, 94.35, 94.56, 94.7],
        [94.3, 94.85, 94.67, 94.68, 94.32, 92.65]]
crypto_30_1 =  np.reshape(np.array(crypto_30_1), (4*6))
crypto_30_1 = pd.DataFrame(crypto_30_1, columns=["ACC"])
crypto_30_1["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6

sns.set_style("whitegrid")
sns.boxplot(x=crypto_30_1["K"], y=crypto_30_1["ACC"], ax=ax[0], color="skyblue")
ax[0].set_title(r"30 $\rightarrow$ 1", fontsize=25)
ax[0].set_xlabel(r"$K$",fontsize=23)
ax[0].set_ylabel("Accuracy",fontsize=23)
ax[0].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=crypto_60_3["K"], y=crypto_60_3["ACC"], ax=ax[1], color="skyblue")
ax[1].set_title(r"60 $\rightarrow$ 3", fontsize=25)
ax[1].set_xlabel(r"$K$",fontsize=23)
ax[1].set_ylabel("",fontsize=1)
ax[1].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=crypto_90_5["K"], y=crypto_90_5["ACC"], ax=ax[2], color="skyblue")
ax[2].set_title(r"90 $\rightarrow$ 5", fontsize=25)
ax[2].set_xlabel(r"$K$",fontsize=23)
ax[2].set_ylabel("",fontsize=1)
ax[2].tick_params(labelsize = 20, grid_alpha=0.5)

fig.tight_layout()
plt.savefig('crypto_sen_K.pdf',format = 'pdf',bbox_inches='tight')
plt.close();
# exit();


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 10), sharey=True)
plt.ylim(90, 97)
my_y_ticks = np.arange(90, 97, 2)
plt.yticks(my_y_ticks)
plt.tick_params(labelsize=15)

indice_90_5 = [[94.4, 94.52, 94.28, 94.57, 94.31, 94.27],
        [94.33, 94.47, 94.35, 92.87, 94.58, 93.74],
        [93.96, 94.02, 94.53, 93.94, 93.7, 94.64],
        [94.01, 94.3, 94.03, 94.08, 94.43, 94.31]]
indice_90_5 = np.reshape(np.array(indice_90_5), (4*6))
indice_90_5 = pd.DataFrame(indice_90_5, columns=["ACC"])
indice_90_5["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6

indice_60_3 = [[95.05, 93.98, 94.51, 95.2, 95.22, 94.52],
        [93.58, 94.65, 95.11, 95.45, 94.42, 95.33],
        [95.42, 94.98, 94.87, 95.22, 95.39, 94.49],
        [95.03, 94.86, 94.69, 93.54, 95.39, 95.19]]
indice_60_3 = np.reshape(np.array(indice_60_3), (4*6))
indice_60_3 = pd.DataFrame(indice_60_3, columns=["ACC"])
indice_60_3["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6

indice_30_1 = [[96.06, 96.04, 96.47, 96.37, 96.15, 95.87],
        [96.16, 96.31, 96.52, 96.23, 96.39, 96.52],
        [96.36, 96.33, 96.11, 96.42, 96.49, 96.36],
        [95.6, 95.96, 96.28, 96.49, 95.55, 96.29]]
indice_30_1 = np.reshape(np.array(indice_30_1), (4*6))
indice_30_1 = pd.DataFrame(indice_30_1, columns=["ACC"])
indice_30_1["K"] = [2]*6 + [3]*6 + [4]*6 + [5]*6




sns.set_style("whitegrid")
sns.boxplot(x=indice_30_1["K"], y=indice_30_1["ACC"], ax = ax[0], color="skyblue")
ax[0].set_title(r"30 $\rightarrow$ 1", fontsize=25)
ax[0].set_xlabel(r"$K$",fontsize=23)
ax[0].set_ylabel("Accuracy",fontsize=23)
ax[0].tick_params(labelsize = 20, grid_alpha=0.5)


sns.boxplot(x=indice_60_3["K"], y=indice_60_3["ACC"], ax = ax[1], color="skyblue")
ax[1].set_title(r"60 $\rightarrow$ 3", fontsize=25)
ax[1].set_xlabel(r"$K$",fontsize=23)
ax[1].set_ylabel("",fontsize=1)
ax[1].tick_params(labelsize = 20, grid_alpha=0.5)

sns.boxplot(x=indice_90_5["K"], y=indice_90_5["ACC"], ax = ax[2], color="skyblue")
ax[2].set_title(r"90 $\rightarrow$ 5", fontsize=25)
ax[2].set_xlabel(r"$K$",fontsize=23)
ax[2].set_ylabel("",fontsize=1)
ax[2].tick_params(labelsize = 20, grid_alpha=0.5)
fig.tight_layout()
plt.savefig('indice_sen_K.pdf',format = 'pdf',bbox_inches='tight')
plt.close()