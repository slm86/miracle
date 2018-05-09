import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load DATA
N = np.array([2,10,30,50,70,100,300,500,700,1000])

name = 'experiments_new_metrics'
data_file = name +'.tsv'
data = pd.read_table(data_file, index_col=0)
smart_mae = np.array(data['smartMAE_mean'])
dumb_mae = np.array(data['dumbMAE_mean'])
dumb_mae_err = np.array(data['dumbMAE_std'])

N = np.insert(N,0,[1])
smart_mae = np.insert(smart_mae,0,[2.16E-08])
dumb_mae = np.insert(dumb_mae,0,[0.4022])
dumb_mae_err = np.insert(dumb_mae_err,0,[0])

# Plotting starts here
fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])

ax1.plot(N,smart_mae,'--',lw=1.5,c='r',marker='x',markersize=8,label='VAE')
ax1.errorbar(N, dumb_mae, yerr=dumb_mae_err, fmt='--x',
             lw=1.5, markersize=8, color='k',
             elinewidth=2, capsize=2, label='simple AE')
ax1.set_ylim([-0.01,1])
ax1.set_xscale('log')
ax1.set_xlim([1,1100])
ax1.set_xticks([1,10,100,1000])
ax1.set_xticklabels([1,10,100,1000])
ax1.set_xlabel('Latent space dimension, N',fontsize=12)
ax1.set_ylabel('Average MAE of reconstruction',fontsize=12)

ax2.plot(N,smart_mae,'--',lw=1.5,c='r',marker='x',markersize=8,label='VAE')
ax2.errorbar(N, dumb_mae, yerr=dumb_mae_err, fmt='--x',
             lw=1.5, markersize=8, color='k',
             elinewidth=2, capsize=2, label='simple AE')
ax2.set_yscale('log')
ax2.set_ylim([0.000000001,1])
ax2.set_xscale('log')
ax2.set_xlim([1,1100])
ax2.set_xticks([1,10,100,1000])
ax2.set_xticklabels([1,10,100,1000])
ax2.set_yticks([1,0.1,0.001,0.00001,0.0000001,0.000000001])
ax2.set_xlabel('N',fontsize=8)
ax2.set_ylabel('Average MAE',fontsize=8)

ax1.legend(bbox_to_anchor=(0.35,0.85),
           bbox_transform=plt.gcf().transFigure)
plt.show()


# Previous code
# fig = plt.figure(figsize=[6,4])
# ax = fig.add_subplot(1, 1, 1)
#
# s1 = ax.plot(N,smart_mae,'--',lw=1.5,c='r',marker='x',markersize=8,label='VAE')
#
# # s2 = ax.plot(N,dumb_mae,'--',lw=0.5,c='b',marker='+',markersize=10)
# s2 = ax.errorbar(N, dumb_mae, yerr=dumb_mae_err, fmt='--x',
#                  lw=1.5, markersize=8, color='k',
#                  elinewidth=2, capsize=2, label='simple AE')
#
# ax.set_yscale('log')
# ax.set_ylim([0.0000000001,1])
# # ax.set_ylim([-0.01,1])
# ax.set_xscale('log')
# ax.set_xlim([1,1100])
#
# # b = [1]#,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001]
# # ax.set_yticks(b)
# # ax.set_yticklabels([1])
#
# ax.set_xticks([1,10,100,1000])
# ax.set_xticklabels([1,10,100,1000])
# # ax.hlines(1,0,1100)
#
# ax.set_xlabel('N')# ax.set_xlabel('Latent space dimension, N')
# ax.set_ylabel('Average MAE')# ax.set_ylabel('Average MAE of reconstruction')
# # ax.legend()
# plt.show()