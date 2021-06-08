# 用于绘制对比实验的图
import numpy as np
import matplotlib.pyplot as plt
label_list = [r'$\tau$=1',r'$\tau$=3',r'$\tau$=5']
x = np.array(range (len(label_list)))
figsize = (4.8,4)
fontsize = 10
ylabelsize = 15;
legendsize=8
width = 0.6/4


## cryptocurrency
fig = plt.figure(figsize=figsize)
# plt.subplot(221)
without_env_best =[86.3, 89.1, 86.35]
without_env_mean =[85.5, 85.76, 82.34]
without_lag_best = [94.6, 92.48, 91.81]
without_lag_mean = [92.28, 91.22, 91.21]
without_dpn_best = [84.79,78.19,69.43]
without_dpn_mean = [84.65,77.5,69.0]
sssm_best = [94.02, 93.73, 92.21]
sssm_mean = [93.77,93.20,91.92]
plt.grid( linestyle="dotted")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim((80, 105))
my_y_ticks = np.arange(60, 105, 9)
plt.yticks(my_y_ticks)
plt.ylabel("Accuracy", fontsize=ylabelsize)

# 以下版本为输仅输出 AVG 但是包含3个消融实验的结
# plt.bar(x,without_dpn_mean,width=width, label='SSSM-DPN',tick_label = label_list,fc = '#60acfc')
# plt.bar(x+width,without_env_mean,width=width, label='SSSM-ENV',tick_label = label_list,fc = '#5bc49f')
# plt.bar(x+width*2,without_lag_mean,width=width, label='SSSM-LAG',tick_label = label_list,fc = '#feb64d')
# plt.bar(x+width*3,sssm_best,width=width, label='SSSM',fc = '#ff7c7c')
# '/' '-' '.' '\\'
plt.bar(x,without_dpn_mean,width=width, label='SSSM-DPN',tick_label = label_list, hatch='/', edgecolor="black", lw=1, fc = '#60acfc')
plt.bar(x+width,without_env_mean,width=width, label='SSSM-ENV',tick_label = label_list, hatch='\\', edgecolor="black", lw=1, fc = '#5bc49f')
plt.bar(x+width*2,without_lag_mean,width=width, label='SSSM-LAG',tick_label = label_list, hatch='x', edgecolor="black", lw=1, fc = '#feb64d')
plt.bar(x+width*3,sssm_best,width=width, label='SSSM', hatch='-', edgecolor="black", lw=1, fc = '#ff7c7c')
plt.legend(fontsize = legendsize)
plt.savefig('crypto_ablation.pdf',format = 'pdf',bbox_inches='tight')
plt.show()


## economic indices
fig = plt.figure(figsize=figsize)
without_dpn_best = [92.45, 91.02, 90]
without_dpn_mean = [92.43, 90.73, 89.9]
without_env_best =[96.08, 94.71, 93]
without_env_mean =[95.85, 93.39, 90.08]
without_lag_best = [96.52, 95.15, 94.29]
without_lag_mean = [96.14, 94.58, 93.88]
sssm_best = [96.53, 95.31, 94.64]
sssm_mean = [96.44,95.24,94.56]

plt.grid( linestyle="dotted")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim((90, 100))
my_y_ticks = np.arange(85,97, 3)
plt.yticks(my_y_ticks)
plt.ylabel("Accuracy", fontsize=ylabelsize)

# plt.bar(x,without_dpn_mean,width=width, label='SSSM-DPN',tick_label = label_list,fc = '#60acfc')
# plt.bar(x+width,without_env_mean,width=width, label='SSSM-ENV',tick_label = label_list,fc = '#5bc49f')
# plt.bar(x+width*2,without_lag_mean,width=width, label='SSSM-LAG',tick_label = label_list,fc = '#feb64d')
# plt.bar(x+width*3,sssm_mean,width=width, label='SSSM',fc = '#ff7c7c')

plt.bar(x,without_dpn_mean,width=width, label='SSSM-DPN',tick_label = label_list, hatch='/', edgecolor="black", lw=1, fc = '#60acfc')
plt.bar(x+width,without_env_mean,width=width, label='SSSM-ENV',tick_label = label_list, hatch='\\', edgecolor="black", lw=1, fc = '#5bc49f')
plt.bar(x+width*2,without_lag_mean,width=width, label='SSSM-LAG',tick_label = label_list, hatch='x', edgecolor="black", lw=1, fc = '#feb64d')
plt.bar(x+width*3,sssm_best,width=width, label='SSSM', hatch='- ', edgecolor="black", lw=1, fc = '#ff7c7c')

plt.legend(fontsize = legendsize)
plt.savefig('indice_ablation.pdf',format = 'pdf',bbox_inches='tight')
plt.show()

exit();

