import pickle
import csv
import os
import pandas as pd
import yaml
import math
import time
import numpy as np
import nltk
import json
import random
import joblib
from datetime import datetime
from gensim.models import Word2Vec, KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from src.embedding import WordToVec as Embedding_Model
from src.DataLoader import getCFilesFromText, LoadToken, GenerateLabels, LoadPickleData, SavedPickle
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.BPLSTM import *
from src.DLSTM_python import *
from src.BLSTM_python import *
from src.PLSTM_python import *
from src.BLSTM import *
from src.LSTM import *
from src.D_constraint import d_constraint
from src.Newton import Newton
import myutils
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import manifold
import seaborn as sns

test_size = 0.2
validation_size = 0.2
config = yaml.safe_load(open('/home/swj/VD/config/config.yaml', 'r'))
random_seed = config['embedding_settings']['seed']
python_tokenizer_saved_path = '/home/swj/VD/embedding/python_'
def verbose(msg):
    ''' Verbose function for print information to stdout'''
    print('[INFO]', msg)
    
def loadData(data_path):
    ''' Load data for training/validation'''
    verbose('Loading data from '+ os.getcwd() + os.sep + data_path + '....')
    total_list, total_list_id = getCFilesFromText(data_path)
    #total_list, total_list_id = LoadToken(data_path)
    verbose("The length of the loaded data list is : " + str(len(total_list)))
    return total_list, total_list_id

def loadAST(data_path):
    ''' Load data for training/validation'''
    verbose('Loading data from '+ os.getcwd() + os.sep + data_path + '....')
    #total_list, total_list_id = getCFilesFromText(data_path)
    total_list, total_list_id = LoadToken(data_path)
    verbose("The length of the loaded data list is : " + str(len(total_list)))
    return total_list, total_list_id

def Tokenization(data_list):
    tokenizer = Tokenizer(num_words=None, filters=',', lower=False, char_level=False, oov_token=None) 
    tokenizer.fit_on_texts(data_list)
    # Save the tokenizer.
    with open(python_tokenizer_saved_path + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)
    
def LoadToknizer(path_of_tokenizer):
    tokenizer = LoadPickleData(path_of_tokenizer)
    return tokenizer

def LoadTokenizer(data_list):
    tokenizer = LoadPickleData(python_tokenizer_saved_path + 'tokenizer.pickle')
    #print('data list, total sequnces: ')
    total_sequences = tokenizer.texts_to_sequences(data_list)
    #print(data_list[3], '\n', total_sequences[3])
    word_index = tokenizer.word_index

    return total_sequences, word_index 
    
def padding(sequences_to_pad):
    padded_seq = pad_sequences(sequences_to_pad, maxlen = config['model_settings']['model_para']['max_sequence_length'], padding ='post')
    return padded_seq

def patitionData(data_list_pad, data_list_id):
    
    test_size = config['training_settings']['dataset_config']['Test_set_ratio']
    validation_size = config['training_settings']['dataset_config']['Validation_set_ratio'] 
    data_list_label = GenerateLabels(data_list_id)
        
    if not config['training_settings']['using_separate_test_set']:
        train_vali_set_x, test_set_x, train_vali_set_y, test_set_y, train_vali_set_id, test_set_id = train_test_split(data_list_pad, data_list_label, data_list_id, test_size=test_size, random_state=random_seed)
        tuple_with_test = train_vali_set_x, train_vali_set_y, test_set_x, test_set_y
        return tuple_with_test
    else:
        train_set_x, validation_set_x, train_set_y, validation_set_y, train_set_id, validation_set_id = train_test_split(train_vali_set_x, train_vali_set_y, train_vali_set_id, test_size=validation_size, random_state=random_seed)
        tuple_without_test = train_set_x, train_set_y, train_set_id, validation_set_x, validation_set_y, validation_set_id
        return tuple_without_test