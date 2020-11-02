import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from sklearn import preprocessing
from combo.models.score_comb import aom, moa

#models
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.loda import LODA
from pyod.models.iforest import IForest
#from pyod.models.mcd import MCD
from pyod.models.cblof import CBLOF

from pyod.models.lof import LOF
#from pyod.models.knn import KNN
#from pyod.models.sod import SOD
#from pyod.models.sos import SOS
#from pyod.models.feature_bagging import FeatureBagging
#from pyod.models.abod import ABOD
#from pyod.models.ocsvm import OCSVM
#from pyod.models.lscp import LSCP
#from pyod.models.cof import COF

from pyod.models.vae import VAE

import pandas as pd
import numpy as np
from scipy.stats import rankdata

import traceback

##########
import time
from datetime import datetime
import pytz
class log:
    def_tz = pytz.timezone('Pacific/Auckland')
    def info(text):        
        print(f'{datetime.now(log.def_tz).replace(microsecond=0)} : {text}');
#############

def rank_fun(arr):
    return rankdata(arr, method = 'dense')

class Anomaly:
    def_dict = '. !?:,\'%-()\/$|&;[]{}"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def_unknown = 'UN'
    
    def __init__(self, max_len_col = None, max_num_col = None):
                
        self.max_len_col = max_len_col
        self.max_num_col = max_num_col
        self.unknown_char = ''
        self.char_ind = None
        self.ind_char = None
        
    def create_dict(self, char_dict = None, unknown_char = None):
        char_dict = char_dict or self.def_dict
        unknown_char = unknown_char or self.def_unknown
        char_list = list(set(char_dict)) #set() additionally randomizes char order
        #char_list = sorted(char_list) #sorted alphabetically
        char_list.insert(0, unknown_char)
        #char_list.insert(len(char_list)//2, unknown_char)        
        
        self.unknown_char = unknown_char
        self.char_ind = dict((c, i) for i, c in enumerate(char_list))
        self.ind_char = dict((i, c) for i, c in enumerate(char_list))
    
    def calc_max_len(self, df, col_len = 'max'):
        measurer = np.vectorize(len)
        #cols_length = measurer(df.select_dtypes(include=[object]).values.astype(str)).max(axis=0)
        cols_length = np.quantile(measurer(df.select_dtypes(include=[object]).values.astype(str)), 0.95, axis=0)
        if col_len == 'mean':
            max_len = int(cols_length.mean())
        elif col_len == 'max':
            max_len = int(max(cols_length))
        elif col_len == 'min':
            max_len = int(min(cols_length))
        else:
            raise ValueError("Only mean/max/min for max_col_len is available") 
       
        self.max_len_num = max_len
        
        return max_len
        
    def text_process(self, df, max_len_col = None, max_num_col = None, 
                  to_lower = False, mirror_out = False, col_len = 'max',
                  char_dict = None, unknown_char = None):
        
        self.max_len_col = max_len_col or self.max_len_col or self.calc_max_len(df, col_len)
        self.max_num_col = max_num_col or self.max_num_col or df.shape[1]
        
        log.info(f'num_col={self.max_num_col}, len_col={self.max_len_col}')
        
        self.create_dict(char_dict, unknown_char)
        
        unk_index = self.char_ind[self.unknown_char]
        
        x_raw = df.to_numpy()
        
        x = np.ones((len(x_raw), self.max_num_col, self.max_len_col), dtype=np.int64) * unk_index
        y = np.zeros((len(x_raw), self.max_num_col, self.max_len_col), dtype=np.object)
 
        for i, doc in enumerate(x_raw):
            for j, sentence in enumerate(doc):
                if sentence is not np.nan and j < self.max_num_col:
                    try:
                        trunc_sentence = sentence[0:self.max_len_col]
                        if to_lower:
                            trunc_sentence = trunc_sentence.lower()
                            
                        for t, char in enumerate(trunc_sentence):
                            #log.info(f'i = {i}, j = {j}, t = {t}, char = {char}')
                            if mirror_out:
                                l = self.max_len_col - t - 1
                            else:
                                l = t - self.max_len_col
                            
                            if char not in self.char_ind:
                                x[i, j, l] = unk_index
                                y[i, j, l] = "UNC"
                            else:
                                x[i, j, l] = self.char_ind[char]
                                y[i, j, l] = char
                    except:
                        log.info(f'i={i}, j={j}, t={t}, sentence = {sentence}')
                        log.info(traceback.print_exc())
        return x, y
    def full_autoencoder(self, X, neurons_list = [64, 32, 32, 64], 
                     hidden_activation = 'relu', output_activation = 'sigmoid', 
                     activity_regularizer = keras.regularizers.l2(), dropout_rate = 0.20,
                     optimizer = keras.optimizers.Adam(), loss = keras.losses.mean_squared_error,
                     batch_size = 32, epochs = 100, patience = 5, validation_split = 0.1, verbose = 1):
        model = Sequential()
        # Input layer
        model.add(Dense(
            X.shape[1], activation = hidden_activation,
            input_shape=(X.shape[1],),
            activity_regularizer=activity_regularizer))
        model.add(Dropout(dropout_rate))

        # Hidden layers
        for i, hidden_neurons in enumerate(neurons_list, 1):
            model.add(Dense(
                hidden_neurons,
                activation=hidden_activation,
                activity_regularizer=activity_regularizer))
            model.add(Dropout(dropout_rate))

        # Output layers
        model.add(Dense(X.shape[1], activation=output_activation,
                        activity_regularizer=activity_regularizer))


        model.compile(loss = loss, optimizer = optimizer)
        if verbose >= 1:
            log.info(model.summary())    
        
        #Early Stopping
        my_callbacks = [EarlyStopping(patience=patience)]
    
        #Additional shuffling
        X_shuffle = np.copy(X)
        np.random.shuffle(X_shuffle)
        
        #Fit on shuffled
        model.fit(X_shuffle, X_shuffle, epochs = epochs, batch_size = batch_size, 
                  shuffle=True, validation_split = validation_split,
                  callbacks=my_callbacks, verbose = verbose)
        
        #Predict on original
        pred = model.predict(X)
        
        return np.sqrt(np.sum(np.square(pred - X), axis=1)).ravel()    
    
    def fit(self, X, shrink_cols = True, data_scaler = preprocessing.MaxAbsScaler(), 
            quick_methods = True, slow_methods = False, nn_methods = False, 
            contamination = 0.05, use_score_rank = False, random_state = None, verbose = 0):

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        elif len(X.shape) > 3:
            raise ValueError("Expected number of dimensions: 2 or 3") 
        
        if shrink_cols:
            X = X[:,~np.all(X == 0, axis=0)]
            log.info('zero columns shrinked')
        if data_scaler:
            X = data_scaler.fit_transform(X)
            log.info(f'used {data_scaler} data scaler')
            #log.info(X[0:1,:])
        
        n_rows = X.shape[0]
        n_features = X.shape[1]
        log.info (f'n_rows = {n_rows}, n_features = {n_features}')
        
        quick_scores = np.zeros([n_rows, 0])
        slow_scores = np.zeros([n_rows, 0])
        nn_scores = np.zeros([n_rows, 0])
        
        if quick_methods:
            # Define anomaly detection tools to be compared
            quick_classifiers = {
                'PCA_randomized':
                    PCA(contamination=contamination, random_state=random_state, 
                        standardization = False, svd_solver = 'randomized'),
                'PCA_full':
                    PCA(contamination=contamination, random_state=random_state, 
                        standardization = False, svd_solver = 'full'),                               
                'COPOD':
                   COPOD(contamination=contamination),  
                f'HBOS': 
                    HBOS(contamination=contamination),
                f'HBOS_{200}': 
                    HBOS(contamination=contamination, n_bins = 200),                
                f'HBOS_{300}':  
                    HBOS(contamination=contamination, n_bins = 300), 
                'LODA':
                    LODA(contamination=contamination),
                'LODA_200':
                    LODA(contamination=contamination, n_random_cuts  = 200),
                'LODA_300':
                    LODA(contamination=contamination, n_random_cuts  = 300),                
                'IForest_100':
                    IForest(contamination=contamination, random_state=random_state, 
                            n_estimators = 100, bootstrap = False, n_jobs = -1),
                'IForest_200':
                    IForest(contamination=contamination, random_state=random_state, 
                            n_estimators = 200, bootstrap = False, n_jobs = -1),                
                'IForest_bootstrap':
                    IForest(contamination = contamination, random_state=random_state, 
                            n_estimators = 150, bootstrap = True, n_jobs = -1), 
                #'MCD': 
                #    MCD(contamination=contamination, random_state=random_state, assume_centered = False),
                #'MCD_centered': 
                #    MCD(contamination=contamination, random_state=random_state, assume_centered = True),    
                f'CBLOF_16':
                    CBLOF(contamination=contamination, random_state=random_state, n_clusters = 16),
                f'CBLOF_24':
                    CBLOF(contamination=contamination, random_state=random_state, n_clusters = 24),
                f'CBLOF_32':
                    CBLOF(contamination=contamination, random_state=random_state, n_clusters = 32)
            }
            
            quick_scores = np.zeros([n_rows, len(quick_classifiers)])

            for i, (clf_name, clf) in enumerate(quick_classifiers.items()):
                log.info(f'{i+1} - fitting {clf_name}')
                try:
                    clf.fit(X)
                    quick_scores[:, i] = clf.decision_scores_
                except:
                    log.info(traceback.print_exc())
                else:    
                    log.info(f'Base detector {i+1}/{len(quick_classifiers)} is fitted for prediction') 

            quick_scores = np.nan_to_num(quick_scores)
            
        if slow_methods:
            # initialize a set of detectors for LSCP
            detector_list = [LOF(n_neighbors=10), LOF(n_neighbors=15), LOF(n_neighbors=20)]
            slow_classifiers = {               
                #'Angle-based Outlier Detector (ABOD)': #too slow and nan results
                #   ABOD(contamination=contamination),
                #'One-class SVM (OCSVM)':
                #   OCSVM(contamination=contamination, cache_size = 2000, shrinking = False, tol = 1e-2),   
                #'LSCP': #slow and no parallel
                #   LSCP(detector_list, contamination=contamination, random_state=random_state, local_region_size = 30),
                #'Feature Bagging': #ensemble #no real par
                #   FeatureBagging(LOF(n_neighbors=20), contamination=contamination, 
                #                  random_state=random_state, n_jobs = -1),                
                #'SOS' : # too memory inefficient  
                #    SOS(contamination=contamination),
                #'COF': # memory inefficient
                #   COF(contamination=contamination),                  
                #'SOD':
                #    SOD(contamination = contamination),
                #'KNN': 
                #   KNN(contamination=contamination, n_jobs = -1),
                #'KNN_50': 
                #   KNN(contamination=contamination, leaf_size = 50, n_jobs = -1),
                #'KNN_70': 
                #   KNN(contamination=contamination, leaf_size = 70, n_jobs = -1),

                'LOF_4':
                   LOF(n_neighbors=4, contamination=contamination, n_jobs = -1),
                'LOF_5':
                   LOF(n_neighbors=5, contamination=contamination, n_jobs = -1),                
                'LOF_6':
                   LOF(n_neighbors=6, contamination=contamination, n_jobs = -1),
                'LOF_7':
                   LOF(n_neighbors=7, contamination=contamination, n_jobs = -1),                
                'LOF_8':
                   LOF(n_neighbors=8, contamination=contamination, n_jobs = -1),
                'LOF_9':
                   LOF(n_neighbors=9, contamination=contamination, n_jobs = -1),                
                'LOF_10':
                   LOF(n_neighbors=10, contamination=contamination, n_jobs = -1),
                'LOF_12':
                   LOF(n_neighbors=12, contamination=contamination, n_jobs = -1),  
                'LOF_14':
                   LOF(n_neighbors=14, contamination=contamination, n_jobs = -1),
                'LOF_16':
                   LOF(n_neighbors=16, contamination=contamination, n_jobs = -1),
                'LOF_18':
                   LOF(n_neighbors=18, contamination=contamination, n_jobs = -1),
                'LOF_20':
                   LOF(n_neighbors=20, contamination=contamination, n_jobs = -1), 
                'LOF_22':
                   LOF(n_neighbors=22, contamination=contamination, n_jobs = -1)            
            }
            
            slow_scores = np.zeros([n_rows, len(slow_classifiers)])

            for i, (clf_name, clf) in enumerate(slow_classifiers.items()):
                log.info(f'{i+1} - fitting {clf_name}')
                try:
                    clf.fit(X)
                    slow_scores[:, i] = clf.decision_scores_
                except:
                    log.info(traceback.print_exc())
                else:    
                    log.info(f'Base detector {i+1}/{len(slow_classifiers)} is fitted for prediction') 
            
            slow_scores = np.nan_to_num(slow_scores)
        
        if nn_methods:
            
            nn_classifiers = {}
            n_list = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
            n_idx = next(x[0] for x in enumerate(n_list) if x[1] < n_features)
            for i in range(3,6):
                n_enc = n_list[n_idx:n_idx+i-1] 
                n_dec = n_enc[::-1]
                n_enc_dec = n_enc + n_dec
                nn_classifiers[f'FULL_AE_{len(n_enc + n_dec)}'] = {'clf': self.full_autoencoder, 
                                                                   'hidden_layers' : n_enc_dec
                                                                  }
                nn_classifiers[f'VAE_{len(n_enc_dec)}'] = {'clf': VAE(contamination = contamination, random_state = random_state,
                                                                      encoder_neurons  = n_enc, decoder_neurons = n_dec,
                                                                      preprocessing = False, epochs = 32, verbosity = verbose), 
                                                            'hidden_layers' : n_enc + n_dec
                                                            }                
                
            
            nn_scores = np.zeros([n_rows, len(nn_classifiers)])
            
            for i, (clf_name, clf) in enumerate(nn_classifiers.items()):
                log.info(f'''{i+1} - fitting {clf_name} with layers {clf['hidden_layers']}''')
                try:
                    if clf['clf'] == self.full_autoencoder:
                        nn_scores[:, i] = clf['clf'](X, neurons_list = clf['hidden_layers'], verbose = verbose)
                    else:
                        clf['clf'].fit(X)
                        nn_scores[:, i] = clf['clf'].decision_scores_                        
                except:
                    log.info(traceback.print_exc())
                else:    
                    log.info(f'Base detector {i+1}/{len(nn_classifiers)} is fitted for prediction')             

            nn_scores = np.nan_to_num(nn_scores)

            
        all_scores = np.concatenate((quick_scores, slow_scores, nn_scores), axis=1)
        all_scores = all_scores[:,~np.all(all_scores == 0, axis=0)]
        log.info(f'total scores = {all_scores.shape[1]}')
        
        all_scores_norm = np.copy(all_scores)
        if use_score_rank:
            all_scores_norm = np.apply_along_axis(rank_fun, 0, all_scores_norm)
            log.info(f'score rank applied')
        all_scores_norm = preprocessing.MinMaxScaler().fit_transform(all_scores_norm)
        
        if all_scores_norm.shape[1] >= 12:
            score_by_aom = aom(all_scores_norm, method = 'dynamic', n_buckets = round(all_scores_norm.shape[1]/4))
            score_by_moa = moa(all_scores_norm, method = 'dynamic', n_buckets = round(all_scores_norm.shape[1]/4))
            score_by_avg = np.mean(all_scores_norm, axis = 1) 
            score_by_max = np.max(all_scores_norm, axis = 1)
        else:
            score_by_avg = np.mean(all_scores_norm, axis = 1)
            score_by_max = np.max(all_scores_norm, axis = 1)
            score_by_aom = score_by_avg
            score_by_moa = score_by_max
        return score_by_aom, score_by_moa, score_by_max, score_by_avg, all_scores, all_scores_norm