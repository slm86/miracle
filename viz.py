import os
import pandas as pd
from sklearn import manifold

def get_tsne(name):
    # Load VAE feature
    encoded_file = name + '_latent.tsv'
    encoded_df_main = pd.read_table(encoded_file, index_col=0)

    encoded_df = encoded_df_main.drop(columns=['acronym'],axis=1)

    # Perform t-SNE on VAE encoded_features
    print('Starting tsne for ',name)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                         learning_rate=300, n_iter=400)
    tsne_out = tsne.fit_transform(encoded_df)
    tsne_out = pd.DataFrame(tsne_out, columns=['1', '2'])
    tsne_out.index = encoded_df.index
    tsne_out.index.name = 'tcga_id'
    tsne_out_file = name + '_tsne' + '.tsv'
    tsne_out.to_csv(tsne_out_file, sep='\t')
    print('Saved tsne file')

