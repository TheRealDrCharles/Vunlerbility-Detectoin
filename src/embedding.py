import os
import pickle
import numpy as np
from src.DataLoader import LoadPickleData

"""
Tokenization:
    1. Perform tokenization and save tokenizer.
    2. Load tokenizer.
"""
class Embedding_Model():
    
    def __init__ (self, config):
        self.config = config
        self.tokenizer_path = self.config['embedding_settings']['embedding_model_saved_path'] 
        self.tokenizer_saved_path = self.config['embedding_settings']['embedding_model_saved_path']
        assert os.path.exists(self.tokenizer_path)
        assert os.path.exists(self.tokenizer_saved_path)
        self.n_workers = self.config['embedding_settings']['n_workers']
        self.seed = self.config['embedding_settings']['seed']
        
    def LoadTokenizer(self, data_list):
        tokenizer = LoadPickleData(self.tokenizer_path + 'tokenizer.pickle')
        #print('data list, total sequnces: ')
        total_sequences = tokenizer.texts_to_sequences(data_list)
        #print(data_list[3], '\n', total_sequences[3])
        word_index = tokenizer.word_index
        
        return total_sequences, word_index            

class WordToVec(Embedding_Model):
    ''' Handler for Word2vec training progress...'''
    def __init__(self,config):
        super(WordToVec, self).__init__(config)
        
        self.wordtovec_size = self.config['embedding_settings']['word2vec']['size']
        self.wordtovec_window = self.config['embedding_settings']['word2vec']['window']
        self.wordtovec_min_count = self.config['embedding_settings']['word2vec']['min_count']
        self.wordtovec_algorithm = self.config['embedding_settings']['word2vec']['algorithm']
        
    def TrainWordToVec(self, data_list):
        from gensim.models import Word2Vec
        
        print ("----------------------------------------")
        print ("Start training the Word2Vec model. Please wait.. ")
        # 2. Train a Vocabulary with Word2Vec -- using the function provided by gensim
        w2vModel = Word2Vec(data_list, workers = self.n_workers, vector_size = self.wordtovec_size, window = self.wordtovec_window, min_count = self.wordtovec_min_count, sg = self.wordtovec_algorithm, seed = self.seed)
        print ("Model training completed!")
        print ("----------------------------------------")
        print ("The trained word2vec model: ")
        print (w2vModel)
        
        w2vModel.wv.save_word2vec_format(self.tokenizer_saved_path + "w2v_model.txt", binary=False)
        
    def ApplyWordToVec(self, word_index):
        
        print ("-------------------------------------------------------")
        print ("Loading trained Word2vec model. ")
        w2v_model = open(self.tokenizer_saved_path + "w2v_model.txt")        
        print ("The trained word2vec model: ")
        print (w2v_model)
        
        embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

        # Use the loaded model
        for line in w2v_model:
            if not line.isspace():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        w2v_model.close()
        
        print ('Found %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.zeros((len(word_index) + 1, self.wordtovec_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
               
        return embedding_matrix, self.wordtovec_size