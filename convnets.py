from __future__ import division,print_function
import math, os, json, sys, re
import _pickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain
from imp import reload

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imread, imresize, imsave

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.metrics import label_ranking_average_precision_score, accuracy_score, f1_score, fbeta_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.utils import resample

from IPython.lib.display import FileLink
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import *
from keras.layers.merge import *
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import applications
import tensorflow as tf

import seaborn as sns


def get_VGG16():
    '''It calls the convolutional part of the vgg model. 
    The model will mainly serve as feature extractor from the images'''
    
    #importing convolutional layers of vgg16 from keras
    model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
    #setting the convolutional layers to non-trainable 
    for layer in model.layers:
        layer.trainable = False
    return(model)

def top_model_vgg_multi(n_classes, 
                        do=0.5,
                        lr=0.01, 
                        loss_function = 'binary_crossentropy'):
    """ Top model multi:MLP 
    The top model corresponds to the VGG16's classification layers.
    The model is adapted for MULTILABEL classification tasks.
    It corresponds to a One vs Rest classification.
    
    Parameters:
        -------------
    
    n_classes : int 
        How many classes are you trying to classify ? 
    
    do : float, optional (default=0.5)
        Dropout rate
    
    lr: float, optional (default=0.01)
        Learning rate
        
    loss_function: str, optional (default='binary_crossentropy')
        loss function, used when the model is compiled
        
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    """
    
    ### top_model takes output from VGG conv and then adds 2 hidden layers
    cnn_model = get_VGG16()
    
    top_model = Flatten(name='top_flatten')(cnn_model.output)
    top_model = Dense(200, activation='relu', name='top_relu_1')(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dropout(do)(top_model)
    top_model = Dense(200, activation='relu', name='top_relu_2')(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dropout(do)(top_model)
    
    ### the last multilabel layers with the number of classes
    predictions = Dense(n_classes, activation='sigmoid')(top_model)
    
    model = Model(cnn_model.input, predictions)
    
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss=loss_function, optimizer=adam)
    
    return(model)

                
def get_batches(dirname, 
                gen=image.ImageDataGenerator(), 
                shuffle=True, 
                batch_size=1, 
                class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

class BaggingClassifierMLP(object):
    """ Bagging CLassifier
    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it. It will automatically apply bootstrap.
    
    
    
    Parameters:
    -----------    
    n_estimators: int, optional (default=5)
        The number of base estimators in the ensemble. 
    
    max_samples: float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
        
    verbose: int, optional (default=0)
        0 or 1. It shows the progress of keras model fitting and predicting.
    """
    def __init__(self,
                 n_estimators=5,
                 max_samples=1.0,
                 verbose=0):
        
        self.n_estimators=n_estimators
        self.max_samples=max_samples
        self.verbose=verbose

    def bootstrap(self, strat_index, X, Y, embeddings):

        """    
        Parameters:
        -----------
        strat_index: list, array or matrix
            The stratified index of data in the fold.
        
        X: list, array, or matrix
            The index of convolutional features.
        
        Y: list, array 
            The LabelBinarized target set.
        
        embeddings: list, array, dict
            The convolutional features created with the feedforward pass.

        Output:
        -----------
        boot_X: list, array
            the bootstrapped X samples
        
        boot_Y: list, array
            the bootstrapped Y samples
        """

        boot_X = []
        boot_Y = []
        for i in range(self.n_estimators):
            index = resample(strat_index,n_samples=int(np.round(self.max_samples * len(strat_index))),random_state=12+i)
            X_train = np.array([embeddings.get(k)['cnn_embedding'] for k in X[index]])
            boot_X.append(X_train)
            del X_train
            y_train = Y[index]
            boot_Y.append(y_train)
            del y_train
        return(boot_X, boot_Y)

    def fit(self,
            base_estimator,
            X_train_samples,Y_train_samples,
            X_val, y_val,
            epochs=15,
            batch_size=60):
        """
        Parameters:
        -----------
        base_estimator: function
            The model to use. It must be a keras model.
        
        X_train_samples: list or array. 
            The bootstrapped samples train data.
        
        Y_trian_samples: list or array. 
            The bootstrapped target samples
        
        X_val: list or array
            The validation embeddings
            
        Y_val: list or array
            The validation targets
            
        epochs: int, optional (default=15)
            The number of epoches. specific to keras.
            
        batch_size: int, optional (default=60)
            The size of the batch. specific to keras.
        
        Output:
        -----------
        weights: list
            Keras weights for the different models used.
        """
        weights = []
        n_samples = len(X_train_samples)
        i = 0
        for X_train, y_train in zip(X_train_samples,Y_train_samples):
            model = base_estimator
            i += 1
            print('step A: fitting model - sample', i)
            model.fit(X_train,
                      y_train,
                      validation_data= (X_val, y_val),
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=self.verbose)
            weights.append(model.get_weights()) 


        return(weights)


    def predict(self,
                base_estimator,
                X_test,
                weights,
                ):
        """
        Parameters:
        -----------
        base_estimator: function
            The model to use. It must be a keras model.
            
        X_test: list or array
            Image embeddings.
        
        weights: list or array
            Trained model weights. The weights are specific to given model architectures.
            
        Output:
        -----------
        predictions: matrix
            Averaged predictions (Bagged predictions)
        """
        
        model = base_estimator
        predictions = []
        
        n_samples = len(weights)
        # ensembling
        print('Step 1: Predict')
        for weight in weights:
            model.set_weights(weight)
            predictions.append(model.predict(X_test,verbose=self.verbose))

        # averaging
        print('Step 2: Averaging predictions')
        bagging_pred = [sum(e)/n_samples for e in zip(*predictions)]

        return(bagging_pred)
    