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
run_id = 'TEST'
name = 'experiments_new\\vae_model_N' + str(N) + '_run' + str(run_id)
plot_title = 'VAE, N='+str(N)
encoded_file = name + '_latent.tsv'

encoded_df_main = pd.read_table(encoded_file, index_col=0)
acronyms = encoded_df_main['acronym']
colours = [acronym_hex_map[i] for i in acronyms]
# cancer_types = list(acronym_hex_map.keys())
# labels = [cancer_types[i] for i in acronyms]
# print(labels)

tsne_out_file = name + '_tsne.tsv'
tsne_df_main = pd.read_table(tsne_out_file, index_col=0)
x = tsne_df_main['1']
y = tsne_df_main['2']
plt.figure()
plt.scatter(x,y,facecolors=colours,edgecolors='k',marker='o',s=25,linewidths=0.25)#label=colours)
plt.xlabel('z variable 1')
plt.ylabel('z variable 2')
plt.xlim([-40,40])
plt.ylim([-40,40])
# plt.legend()
# plt.savefig(fig_name)
# print('Saved ',fig_name,' to disk')
plt.title(plot_title)
plt.show()
"""

is_dumb = False

for N in [10,20,30,50,70,100,300,500,700,1000]:
    if is_dumb:
        name = 'experiments\\dumb_model_N'+str(N)
        fig_name = 'experiments\\figures\\dumb_model_N' + str(N) + '_blobs.png'
        plot_title = 'simple AE, N=' + str(N)
    else:
        name = 'experiments\\vae_model_N' + str(N)
        fig_name = 'experiments\\figures\\vae_model_N' + str(N) + '_blobs.png'
        plot_title = 'VAE, N=' + str(N)

    encoded_file = name + '_latent.tsv'
    encoded_df_main = pd.read_table(encoded_file, index_col=0)
    acronyms = encoded_df_main['acronym']

    unique_acronyms = set(acronyms)
    acronym_mapping = dict()
    for idx, elem in enumerate(unique_acronyms):
        acronym_mapping[elem] = idx

    colours = [acronym_mapping[i] for i in acronyms]
    colours_tybalt = [acronym_hex_map[i] for i in acronyms]

    tsne_out_file = name + '_tsne.tsv'
    tsne_df_main = pd.read_table(tsne_out_file, index_col=0)

    x = tsne_df_main['1']
    y = tsne_df_main['2']

    plt.figure()
    plt.scatter(x, y, c=colours_tybalt, edgecolors='k', marker='o', s=25, linewidths=0.25)  # label=colours)
    plt.xlabel('z variable 1')
    plt.ylabel('z variable 2')
    plt.xlim([-40, 45])
    plt.ylim([-40, 45])

    plt.title(plot_title)
    plt.savefig(fig_name)
    print('Saved ',fig_name,' to disk')
    plt.close()
    """
