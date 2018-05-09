import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pylab import *

def get_hex():
    y_mapping = dict()
    acronym_ordered = list()
    with open('tcga_colors.tsv') as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            y_mapping[row['Study Abbreviation']] = row['Hex Colors']
            # acronym_ordered.append(row['Hex Colors'])
    return y_mapping


acronym_hex_map = get_hex()

# USER !!!
folder_name = "experiments_new"
save_folder = 'figures\\'
is_dumb = True
N_series = [2,10,100,1000]
run_id = 1
other_comment = ''
high_res = False

if is_dumb:
    model_name = 'dumb'
else: model_name = 'vae'

if high_res:
    resolution_dpi = 60
else: resolution_dpi = 60

fig_name = save_folder+'Plot3_'+folder_name+'_'+model_name+'_run'+str(run_id)+other_comment+'.png'

fig = plt.figure(figsize=[24,24])
font_label = 40
font_text = 25
font_tick_label = 25
text_position = [0.62,0.93]
tick_length = 15

ax1 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
ax3 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
ax4 = fig.add_axes([0.5, 0.1, 0.4, 0.4])

#First plot
N = N_series[0]
name = folder_name+'\\'+model_name+'_model_N'+str(N)+'_run'+str(run_id)
plot_title = 'Simple AE, N='+str(N)

encoded_file = name + '_latent.tsv'
encoded_df_main = pd.read_table(encoded_file, index_col=0)
acronyms = encoded_df_main['acronym']
colours_tybalt = [acronym_hex_map[i] for i in acronyms]

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax1.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax1.set_xlim([-50,50])
ax1.set_xticks(np.arange(-40,40,10))
ax1.set_yticks(np.arange(-40,50,10))
ax1.set_xticklabels([])
ax1.set_ylim([-50,50])
ax1.tick_params(labelsize=font_tick_label,length=tick_length,direction='inout')
ax1.text(text_position[0], text_position[1],
         plot_title,horizontalalignment='left',transform=ax1.transAxes,size=font_text)

# Second plot
N = N_series[1]
name = folder_name+'\\'+model_name+'_model_N'+str(N)+'_run'+str(run_id)
plot_title = 'Simple AE, N='+str(N)

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax2.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax2.set_xlim([-50,50])
ax2.set_ylim([-50,50])
ax2.set_xticks(np.arange(-40,50,10))
ax2.set_xticklabels([])
ax2.set_yticks(np.arange(-40,50,10))
ax2.set_yticklabels([])
ax2.tick_params(axis='y',direction='inout',length=tick_length)
ax2.tick_params(labelsize=font_tick_label)
ax2.text(text_position[0], text_position[1],
         plot_title,horizontalalignment='left',transform=ax2.transAxes,size=font_text)

# Third plot
N = N_series[2]
name = folder_name+'\\'+model_name+'_model_N'+str(N)+'_run'+str(run_id)
plot_title = 'Simple AE, N='+str(N)

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax3.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax3.set_xlim([-50,50])
ax3.set_ylim([-50,50])
ax3.set_xticks(np.arange(-40,50,10))
ax3.set_yticks(np.arange(-40,50,10))
ax3.tick_params(axis='x',top=True,direction='inout',length=5)
ax3.tick_params(labelsize=font_tick_label,length=tick_length,direction='inout')
ax3.text(text_position[0], text_position[1],
         plot_title,horizontalalignment='left',transform=ax3.transAxes,size=font_text)

# Fourth plot
N = N_series[3]
name = folder_name+'\\'+model_name+'_model_N'+str(N)+'_run'+str(run_id)
plot_title = 'Simple AE, N='+str(N)

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax4.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax4.set_xlim([-50,50])
ax4.set_ylim([-50,50])
ax4.set_xticks(np.arange(-40,50,10))
ax4.set_yticks(np.arange(-40,50,10))
ax4.tick_params(axis='y',direction='inout',length=tick_length)
ax4.tick_params(axis='x',top=True,direction='inout',length=tick_length)
ax4.set_yticklabels([])
ax4.tick_params(labelsize=font_tick_label)
ax4.text(text_position[0], text_position[1],
         plot_title,horizontalalignment='left',transform=ax4.transAxes,size=font_text)

fig = gcf()
fig.text(0.5, 0.04, 'latent variable 1', ha='center', size=font_label)
fig.text(0.02, 0.5, 'latent variable 2', va='center', rotation='vertical', size=font_label)
# plt.show()

plt.savefig(fig_name,dpi=resolution_dpi)
print('Saved ',fig_name,' to disk')
plt.close()