import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

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

N = 10
run_id = 1
name = 'experiments_new\\vae_model_N' + str(N) + '_run' + str(run_id)
plot_title = 'VAE, N='+str(N)
encoded_file = name + '_latent.tsv'

encoded_df_main = pd.read_table(encoded_file, index_col=0)
acronyms = encoded_df_main['acronym']
colours_tybalt = [acronym_hex_map[i] for i in acronyms]

fig = plt.figure(figsize=[12,6])
ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8])
ax2 = fig.add_axes([0.5, 0.1, 0.4, 0.8])

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax1.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax1.set_xlabel('z variable 1')
ax1.set_ylabel('z variable 2')
ax1.set_xlim([-40,40])
ax1.set_xticks(np.arange(-40,40,10))
ax1.set_ylim([-40,40])
ax1.set_title(plot_title)


# Second plot
N = 1000
run_id = '1'
name = 'experiments_new\\vae_model_N' + str(N) + '_run' + str(run_id)
plot_title = 'VAE, N='+str(N)

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']

ax2.scatter(x,y,facecolors=colours_tybalt,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
ax2.set_xlabel('z variable 1')
ax2.set_xlim([-40,40])
ax2.set_ylim([-40,40])
ax2.set_xticks(np.arange(-30,50,10))
ax2.set_yticklabels([])
ax2.set_title(plot_title)

plt.show()