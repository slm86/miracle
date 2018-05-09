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


def run_loaded_experiment(data,
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
    loaded_model.summary()
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk")

    predictions = loaded_model.predict(np.array(rnaseq_test_df))
    mae = mean_absolute_error_array(np.array(rnaseq_test_df),np.array(predictions))
    stdae = std_absolute_error_array(np.array(rnaseq_test_df), np.array(predictions))
    print('Mean Abs Error', mae)
    print('Std Abs Error', stdae)
    # fout = open(name + 'metrics.txt', 'w')
    # fout.write('Mean Abs Error\n')
    # fout.write(str(mae))
    # fout.write('\nStd Abs Error\n')
    # fout.write(str(stdae))
    # print('Wrote metrics file for ',name)

    # Do not mess with this part of the code
    # Get information you need to create the latent.tsv file
    encoded_df = encoder.predict(np.array(rnaseq_df_saved))
    sample_id_list = list(rnaseq_df_saved.index)

    sample_to_encoding = dict()
    for idx, sample_id in enumerate(sample_id_list):
        sample_to_encoding[sample_id] = encoded_df[idx]

    acronym_map, acronym_ordered = get_acronyms()

    fout = open(name + '_latent.tsv', 'w')
    # write the header first
    fout.write('sample_id')
    for i in range(model_parameters['latent_dim']):
        fout.write('\t%d' % (i+1))
    fout.write('\tacronym\n')
    print('Wrote latent data file for ',name)

    # now write the data
    for sample_id in acronym_ordered:
        fout.write(sample_id)
        for elem in sample_to_encoding[sample_id]:
            fout.write('\t%f' % elem)
        fout.write('\t%s\n' % acronym_map[sample_id])



def train_and_save(data, rnaseq_df, model_parameters, name, is_dumb=True):

    # Set hyper parameters
    json_name = name + ".json"
    h5_name = name + ".h5"
    if is_dumb:
        get_autoencoder = get_dumb_autoencoder
    else:
        get_autoencoder = get_vae


    rnaseq_test_df, rnaseq_train_df = data
    autoencoder, encoder = get_autoencoder(original_dim=np.int(rnaseq_df.shape[1]), **model_parameters)
    autoencoder.summary()

    beta = K.variable(0)
    kappa = 1
    training_parameters = dict(shuffle=True,
                               epochs=50,
                               batch_size=50,
                               validation_data=(np.array(rnaseq_test_df), None),
                               callbacks=[WarmUpCallback(beta, kappa)])

    # fit Model
    if (get_autoencoder == get_vae):
        autoencoder.fit(np.array(rnaseq_train_df), **training_parameters)
    elif (get_autoencoder == get_dumb_autoencoder):
        autoencoder.fit(rnaseq_train_df, rnaseq_train_df, nb_epoch=50, batch_size=50, shuffle=True,
                        validation_data=(rnaseq_test_df, rnaseq_test_df))

    # serialize model to JSON
    model_json = autoencoder.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights(h5_name)
    print("Saved model to disk for ",name)


def main():
    # Read data file
    rnaseq_file = 'pancan_scaled_zeroone_rnaseq.tsv.gz'
    rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
    rnaseq_df_saved = rnaseq_df.copy(deep=True)
    print(rnaseq_df.shape)

    # print(np.max(np.array(rnaseq_df_saved)), np.argmax(np.array(rnaseq_df_saved)))
    # print(np.min(np.abs(np.array(rnaseq_df_saved))))

    # Split 10% test set randomly
    test_set_percent = 0.1
    data_test = rnaseq_df.sample(frac=test_set_percent,random_state=1)
    data_train = rnaseq_df.drop(data_test.index)
    data = (data_test, data_train)


    # # Run experiments for smart VAE
    for i in np.arange(1,6):
        for N in [2,10,30,50,70,100,300,500,700,1000]:
            model_parameters = dict(latent_dim=N,
                                    intermediate_dim=256)

            name = 'experiments_new\\vae_model_N'+str(N)+'_run'+str(i)

            train_and_save(data, rnaseq_df_saved, model_parameters, name=name, is_dumb=False)

            run_loaded_experiment(data,
                                  rnaseq_df_saved,
                                  model_parameters,
                                  name=name,
                                  is_dumb=False)
            get_tsne(name=name)

            # Now do same experiments for dumb AE
            model_parameters = dict(latent_dim=N,
                                    intermediate_dim=256)
            name = 'experiments_new\\dumb_model_N'+str(N)+'_run'+str(i)
            train_and_save(data, rnaseq_df_saved, model_parameters, name=name, is_dumb=True)
            run_loaded_experiment(data,
                                  rnaseq_df_saved,
                                  model_parameters,
                                  name=name,
                                  is_dumb=True)
            get_tsne(name=name)


    # N = 100
    # run_id = 'TEST'
    # name = 'experiments_new\\vae_model_N' + str(N) + '_run' + str(run_id)
    # model_parameters = dict(latent_dim=N,
    #                         intermediate_dim=256)
    # # train_and_save(data, rnaseq_df_saved, model_parameters, name=name, is_dumb=False)
    # run_loaded_experiment(data,
    #                       rnaseq_df_saved,
    #                       model_parameters,
    #                       name=name,
    #                       is_dumb=False)

if __name__ == '__main__':
    main()