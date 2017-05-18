'''
Trains a Deep Convnet for NLP!
'''

import os
import sys

import pandas as pd
import numpy as np
import gzip
import glob
import os.path
import time
import h5py
import string

import keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.initializers import RandomNormal


class crepeK2(object):

    def __init__(self, num_classes, model_loc='saved_models/crepe.hdf5', train_data_loc='data/train.csv', val_data_loc='data/test.csv', test_data_loc='data/test.csv'):
        """
        Some global variables used by our model
        """

        self.model_loc = model_loc
        self.train_data_loc = train_data_loc
        self.val_data_loc = val_data_loc
        self.test_data_loc = test_data_loc
        self.num_classes = num_classes

        tr_df = pd.read_csv(self.train_data_loc)
        va_df = pd.read_csv(self.val_data_loc)
        te_df = pd.read_csv(self.test_data_loc)

        self.num_train = len(tr_df)
        self.num_val = len(va_df)
        self.num_test = len(te_df)

        # Define our Alphabet
        self.alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation))
        self.alpha_map = {}
        for idx, c in enumerate(self.alphabet):
            self.alpha_map[c] = idx

        # Initialise the model
        self.model = self._get_model()


    def fit(self, batch_size=100, epochs=5):
        """
        The fit function.
        """

        # set up our data generators
        train_data = self._data_gen(self.train_data_loc, batch_size)
        val_data = self._data_gen(self.val_data_loc, batch_size)

        print(self.model.summary())

        self.model.fit_generator(generator=train_data, steps_per_epoch=self.num_train / batch_size,
                                 validation_data=val_data, validation_steps=self.num_val / batch_size,
                                 epochs=epochs, callbacks=self._get_callbacks(), verbose=1)


    def evaluate(self, batch_size=100):
        """
        Prints the evaluation score, using the current weights and the specified test set.
        """

        test_data = self._data_gen(self.test_data_loc, batch_size)

        score = model.evaluate_generator(generator=test_data, steps=self.num_val / batch_size)

        print 'Test Loss: ' + score[0]
        print 'Test Accuracy: ' + score[1]
        
    
    def predict(self, batch_size=100, pred_data_loc=None, model_loc=None):
        """
        Predicts using the current weights (or can specify a set of weights to be loaded).
        """

        # If we don't have an input, then we just use the test data previously specified
        # Otherwise, predict using the new input
        if pred_data_loc is None:
            pred_data = self.test_data_loc
        
        else:
            pred_data = pred_data_loc
        
        pr_df = pd.read_csv(pred_data)
        num_pred = len(pr_df)
        
        pred_data = self._data_gen(pred_data, batch_size, mode='pred')
        
        # Check if we have a weight file being passed, if so, load that, otherwise don't.
        if model_loc:
            self.model.load_weights(model_loc)
        
        return self.model.predict_generator(generator=pred_data, steps=num_pred / batch_size)

        
    def _get_callbacks(self):
        """
        A helper function which defines our callbacks
        """

        # Define our model callbacks and save path, checkpoint
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=0.001, mode='min')
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(self.model_loc, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        return [checkpoint, reduce_lr, earlyStop]


    def _data_gen(self, file_name, batch_size, mode='train'):
        """
        A generator that yields a batch of samples.

        Mode can be "train" or "pred".
        If pred, only an X value is yielded.
        """

        while True:

            reader = pd.read_csv(file_name, chunksize=batch_size)

            for data in reader:
                data['X'] = data['X'].apply(lambda x: self._get_char_seq(str(x).lower()))

                if mode == 'train':
                    yield (np.asarray(data['X'].tolist()), np_utils.to_categorical(np.asarray(data['y'].tolist()), num_classes=self.num_classes))

                elif mode == 'pred':
                    yield np.asarray(data['X'].tolist())

                else:
                    raise Exception('Invalid mode specified. Must be "train" or "pred".')


    def _get_char_seq(self, desc):
        """
        Converts a sequence of characters into a sequence of one hot encoded "frames"
        """

        INPUT_LENGTH = 1014
        seq = []

        for char in list(reversed(desc))[0:INPUT_LENGTH]:
            # we reverse the description then get the first 1014 chars (why 1014? Because that's what they did in the paper...)
            # Get the index of character in the alphabet list
            try:
                fr = np.zeros(len(self.alphabet))
                fr[self.alpha_map[char]] = 1
                seq.append(fr)

            except (ValueError, KeyError):
                # character is not in index
                seq.append(np.zeros(len(self.alphabet)))

        # Now check the generated input and pad out to 1014 if too short
        if INPUT_LENGTH - len(seq) > 0:
            seq.extend([np.zeros(len(self.alphabet)) for i in range(0, INPUT_LENGTH - len(seq))])

        return np.array(seq)


    def _get_model(self):
        """
        Returns the model.
        """

        k_init = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        model = Sequential()
        # Layer 1
        model.add(Convolution1D(input_shape=(1014, len(self.alphabet)), filters=256, kernel_size=7, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        # Layer 2
        model.add(Convolution1D(filters=256, kernel_size=7, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        # Layer 3, 4, 5
        model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        # Layer 6
        model.add(Convolution1D(filters=256, kernel_size=3, padding='valid', activation='relu', kernel_initializer=k_init))
        model.add(MaxPooling1D(pool_size=3, strides=3))
        # Layer 7
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer=k_init))
        model.add(Dropout(0.5))
        # Layer 8
        model.add(Dense(1024, activation='relu', kernel_initializer=k_init))
        model.add(Dropout(0.5))
        # Layer 9
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model