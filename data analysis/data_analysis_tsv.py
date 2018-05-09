import tensorflow as tf
import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback
from keras.models import model_from_json
from evaluation import get_acronyms
from viz import get_tsne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vae import get_vae, get_dumb_autoencoder, WarmUpCallback, mean_absolute_error_array, std_absolute_error_array


def run_metrics_analysis(data,
                         rnaseq_df_saved,
                         model_parameters,
                         name,
                         is_dumb=True):
    if is_dumb:
        get_autoencoder = get_dumb_autoencoder
    else:
        get_autoencoder = get_vae

    rnaseq_test_df, rnaseq_train_df = data
    loaded_model, encoder = get_autoencoder(original_dim=np.int(rnaseq_test_df.shape[1]), **model_parameters)
    # loaded_model.summary()
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk:", name)

    predictions = loaded_model.predict(np.array(rnaseq_test_df))
    mae = mean_absolute_error_array(np.array(rnaseq_test_df), np.array(predictions))
    stdae = std_absolute_error_array(np.array(rnaseq_test_df), np.array(predictions))

    return (mae, stdae)


def main():
    # Read data file
    rnaseq_file = 'pancan_scaled_zeroone_rnaseq.tsv.gz'
    rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
    rnaseq_df_saved = rnaseq_df.copy(deep=True)
    print(rnaseq_df.shape)

    # Split 10% test set randomly
    test_set_percent = 0.1
    data_test = rnaseq_df.sample(frac=test_set_percent, random_state=1)
    data_train = rnaseq_df.drop(data_test.index)
    data = (data_test, data_train)

    folder_name = 'experiments_new2'

    n_start = 1
    n_end = 5
    n_runs = n_end - n_start + 1

    file_name = folder_name + '_metrics.txt'
    fout = open(file_name, 'w')
    fout.write('Smart AE\n')

    # Run experiments for smart VAE
    for N in [2, 10, 30, 50, 70, 100, 300, 500, 700, 1000]:
        fout.write('\nN=')
        fout.write(str(N))
        mae = np.zeros(n_runs)
        stdae = np.zeros(n_runs)
        model_parameters = dict(latent_dim=N,
                                intermediate_dim=256)
        for i in np.arange(n_start, n_end + 1):
            name = folder_name + '/vae_model_N' + str(N) + '_run' + str(i)
            mae_dummy, stdae_dummy = run_metrics_analysis(data,
                                                          rnaseq_df_saved,
                                                          model_parameters,
                                                          name=name,
                                                          is_dumb=False)

            mae[i - n_start] = mae_dummy
            stdae[i - n_start] = stdae_dummy
        print(mae)
        fout.write('\nMean Abs Error\n')
        fout.write(str(np.mean(mae)))
        fout.write('\t')
        fout.write(str(np.std(mae)))
        fout.write('\n')

        fout.write('Standard Abs Error\n')
        fout.write(str(np.mean(stdae)))
        fout.write('\t')
        fout.write(str(np.std(stdae)))
        fout.write('\n')

    fout.write('\n')
    fout.write('\n')
    fout.write('Dumb AE\n')
    # Now do same experiments for dumb AE
    for N in [2, 10, 30, 50, 70, 100, 300, 500, 700, 1000]:
        fout.write('\nN=')
        fout.write(str(N))
        mae = np.zeros(n_runs)
        stdae = np.zeros(n_runs)
        model_parameters = dict(latent_dim=N,
                                intermediate_dim=256)
        for i in np.arange(n_start, n_end + 1):
            name = folder_name + '/dumb_model_N' + str(N) + '_run' + str(i)
            mae_dummy, stdae_dummy = run_metrics_analysis(data,
                                                          rnaseq_df_saved,
                                                          model_parameters,
                                                          name=name,
                                                          is_dumb=True)
            mae[i - n_start] = mae_dummy
            stdae[i - n_start] = stdae_dummy
        print(mae)
        fout.write('\nMean Abs Error\n')
        fout.write(str(np.mean(mae)))
        fout.write('\t')
        fout.write(str(np.std(mae)))
        fout.write('\n')

        fout.write('Standard Abs Error\n')
        fout.write(str(np.mean(stdae)))
        fout.write('\t')
        fout.write(str(np.std(stdae)))
        fout.write('\n')

    print('Finished, saved file')


if __name__ == '__main__':
    main()