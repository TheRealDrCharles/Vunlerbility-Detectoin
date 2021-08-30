from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Bidirectional, GRU, LSTM, GlobalMaxPooling1D, Bidirectional
from keras.layers.core import Dropout
from src.attention.attention import Attention
from src.HAN import AttentionLayer
class customized_model(object):
    
    def __init__ (self, config):
        
        self.MAX_LEN = config['model_settings']['model_para']['max_sequence_length']
        self.use_dropout = config['model_settings']['model_para']['use_dropout']
        self.dropout_rate = config['model_settings']['model_para']['dropout_rate']
        self.LOSS_FUNCTION = config['model_settings']['loss_function']
        self.OPTIMIZER = config['model_settings']['optimizer']['type']
        self.dnn_size = config['model_settings']['model_para']['dnn_size']
        self.rnn_size = config['model_settings']['model_para']['rnn_size']
        self.embedding_trainable = config['model_settings']['model_para']['embedding_trainable']
        
    def build_LSTM_with_HAN(self, word_index, embedding_matrix, embedding_dim, GPU_flag):
        inputs = Input(shape=(self.MAX_LEN,))

        sharable_embedding = Embedding(len(word_index) + 1,
                                   embedding_dim,
                                   weights=[embedding_matrix],
                                   input_length=self.MAX_LEN,
                                   trainable=self.embedding_trainable)(inputs)

        #lstm_1 = LSTM(self.rnn_size, activation='tanh', return_sequences=True)(sharable_embedding)
        
        bilstm_1 = Bidirectional(LSTM(self.rnn_size, activation='tanh', return_sequences=True), merge_mode='concat')(sharable_embedding)
        
        droput_layer_1 = Dropout(self.dropout_rate)(bilstm_1)
        
        att_layer = Attention(128)(droput_layer_1)
                
        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(att_layer)
            dense_1 = Dense(int(self.dnn_size/2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size/2), activation='relu')(att_layer)
            
        dense_2 = Dense(int(self.rnn_size/4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)
        
        model = Model(inputs=inputs, outputs = dense_3, name='LSTM_with_ATT')
        
        model.compile(loss=self.LOSS_FUNCTION,
                 optimizer=self.OPTIMIZER,
                 metrics=['accuracy'])

        return model