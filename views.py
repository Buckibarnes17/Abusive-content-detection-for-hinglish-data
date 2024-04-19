from django.shortcuts import render
from rest_framework.permissions import IsAuthenticated
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.views import APIView
from keras.models import load_model
import json
import os
from .models import User
from .serializers import UserSerializer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW
from torch.utils.data import SequentialSampler
from transformers import BertModel, BertTokenizer,TFAutoModel
from transformers import RobertaTokenizer
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append('/Users/raghavkatyal/Documents/AbusiveContentDetection/Django_Backend/djangoApi/myapi')
from src import data
from src.models.layers import AttentionWithContext
from src.models.layers import Attention
# Model
# Load the tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# bert= BertModel.from_pretrained('bert-base-multilingual-cased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#For Aggression
# class BERT_Arch(nn.Module):

#     def __init__(self, bert):
#         super(BERT_Arch, self).__init__()
        
#         self.bert = bert 
        
#         # dropout layer
#         self.dropout = nn.Dropout(0.1)
      
#         # relu activation function
#         self.relu =  nn.ReLU()

#         # dense layer 1
#         self.fc1 = nn.Linear(768,512)
      
#         # dense layer 2 (Output layer)
#         self.fc2 = nn.Linear(512,3)

#         #softmax activation function
#         self.softmax = nn.LogSoftmax(dim=1)

#     #define the forward pass
#     def forward(self, sent_id, mask):
        
#         #pass the inputs to the model  
#         _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
#         x = self.fc1(cls_hs)

#         x = self.relu(x)

#         x = self.dropout(x)

#         # output layer
#         x = self.fc2(x)
      
#         # apply softmax activation
#         x = self.softmax(x)

#         return x


#For Hate

# class BERT_Arch(nn.Module):

#     def __init__(self, bert):
#         super(BERT_Arch, self).__init__()
        
#         self.bert = bert 
        
#         self.dropout = nn.Dropout(0.1)
      
#         self.relu =  nn.ReLU()

#         self.fc1 = nn.Linear(768,512)
      
#         self.fc2 = nn.Linear(512,2)

#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, sent_id, mask):
#         _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
#         x = self.fc1(cls_hs)

#         x = self.relu(x)

#         x = self.dropout(x)

#         x = self.fc2(x)
      
#         x = self.softmax(x)

#         return x

#     # define the forward pass
#     def forward(self, input_ids, attention_mask=None):
#         # pass the inputs to the model  
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         cls_hs = outputs.last_hidden_state[:, 0, :]  # Extract the CLS token representation
#         x = self.fc1(cls_hs)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x

#TensorFlow
class OuterProductMHSA(layers.Layer):
    def __init__(self, embed_dim):
        super(OuterProductMHSA, self).__init__()
        self.embed_dim = embed_dim
        
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)

    def outer_product_attention(self, query, key, value):
        score = tf.einsum('bnd,bmd->bnmd', query, key)
        weights = tf.nn.softmax(score, axis=2)
        output = tf.einsum('bnmd,bmd->bnd', score, value)
        return output, weights

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        attention, weights = self.outer_product_attention(query, key, value)
        
        return attention
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return input_shape[0], input_shape[1], self.embed_dim
        elif len(input_shape) == 3:
            return input_shape[0], input_shape[1], input_shape[2], self.embed_dim
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, outer_attention=False, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.outer_attention = outer_attention
        if outer_attention == False:
            self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        else:
            self.att1 = OuterProductMHSA(embed_dim)
            self.att2 = MultiHeadSelfAttention(embed_dim, num_heads)
            self.attn_weights = layers.Dense(1)

        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        if self.outer_attention == False:
            attn_output = self.att(inputs)
        else:
            mha_attn = self.att2(inputs)
            outer_attn = self.att1(inputs)
            weights = tf.nn.sigmoid(self.attn_weights(mha_attn))
            attn_output = weights*mha_attn + (1-weights)*outer_attn

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim
    


class CharModel(tf.keras.models.Model):
    def __init__(self, max_word_char_len, char_vocab_size, n_units):
        super(CharModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=char_vocab_size, output_dim=n_units)
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_units, dropout=0.2, return_sequences=True))
        self.n_units = n_units  # Fix the typo here

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2 * self.n_units)  # Assuming bidirectional LSTM


def BERT(word_vocab_size, char_vocab_size, wpe_vocab_size, n_out, transformer_model_pretrained_path='roberta-base', seq_output=False, vectorizer_shape=None,\
                             n_heads=8, max_word_char_len=20, max_text_len=20, max_char_len=100, n_layers=2, n_units=128, emb_dim=128):
    
    word_inputs = tf.keras.layers.Input((max_text_len,), dtype=tf.int32)
    char_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)
    subword_inputs = tf.keras.layers.Input((max_text_len,max_word_char_len,), dtype=tf.int32)
    wpe_inputs = tf.keras.layers.Input((max_char_len,), dtype=tf.int32)

    automodel = TFAutoModel.from_pretrained(transformer_model_pretrained_path)
    x = automodel(wpe_inputs)[1]
    x = tf.keras.layers.Dropout(0.2)(x)
    
    if vectorizer_shape:
        tfidf = tf.keras.layers.Input((vectorizer_shape,))
        x = tf.keras.layers.Dense(n_units)(tf.keras.layers.Concatenate()([x,tfidf]))
    else:
        x = tf.keras.layers.Dense(n_units)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    out = tf.keras.layers.Dense(n_out, activation='softmax')(x)
    
    if vectorizer_shape:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs,tfidf], out)
    else:
        model = tf.keras.models.Model([word_inputs,char_inputs,subword_inputs,wpe_inputs], out)

    return model

def predict_abusiveness(input_text):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1)
    char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1)

    word_tokenizer.fit_on_texts(input_text)
    char_tokenizer.fit_on_texts(input_text)
    n_words = len(word_tokenizer.word_index)+1
    n_chars = len(char_tokenizer.word_index)+1
    n_subwords = tokenizer.vocab_size
    
    word_test_inputs = word_tokenizer.texts_to_sequences([input_text])
    word_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs, maxlen=50)

    model = HAN(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=2,max_word_char_len=20,\
                                             max_text_len=50, max_char_len=200,\
                                             n_layers=2, n_units=128, emb_dim=128)
    
    subword_test_inputs = np.asarray(data.data_utils.subword_tokenization(input_text, char_tokenizer,50, 20))

    char_test_inputs = char_tokenizer.texts_to_sequences([input_text])
    char_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_test_inputs, maxlen=200)

    # tf.model.load_model('HAN_ce_with_features.h5')
    test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs])
    
    #Prev Bert
    # input_ids = torch.tensor(inputs['input_ids'])
    # attention_mask = torch.tensor(inputs['attention_mask'])
    print(test_pred)


   
#Prev (Bert_hate)

#     cpu = torch.device('cpu')
#     with torch.no_grad():
#         preds = model(input_ids.to(cpu), attention_mask.to(cpu))
#         preds = preds.detach().cpu().numpy()


# # model's performance
#     preds = np.argmax(preds, axis = 1)
    
    # print(preds[0])
    # pred = ""
    # if preds[0] == 0:
    #     pred = 'Not Hate'
    # elif preds[0] == 1:
    #     pred = 'Hate'
    # else:
    #     pred = 'non-aggressive '
    # if preds[0] == 0:
    #     pred = 'Covertly Aggresive'
    # elif preds[0] == 1:
    #     pred = 'Openly Aggresive'
    # else:
    #     pred = 'non-aggressive '
    # # predsLabel = np.array(['CAG' if preds[i]==0 else 'OAG' if preds[i]==1 else 'NAG' for i in range()])
    # return pred



# API Views:
class MyModel(APIView):
    PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.join(PATH, 'bert_model_save.pth')
    def post(self, request):
        # Extract the text from the request data
        data = request.data
        input_text = data.get("data", None)
        print(input_text)
        if input_text is None:
            return Response({"status":"ss", "message": "Provide data!"})
        
        
        # Predict the abusiveness of the input text
        # res =predict_abusiveness(input_text)
        print("path: ",self.PATH)
        model = torch.load(self.MODEL)
        print("type: ",type(model))
        # res = model.predict(input_text)
        for key, value in model.items():
           print(key)
        # model.eval()
      
         
        # print("res: ",res)
        # print(type(res))
        # print(res)
        # Return the predict`ed score``
        return Response({"status":"ok", "score": "res"})
    
class CreateUser(APIView):
    serializer_class = UserSerializer
    def post(self, request):
        serializer = self.serializer_class(data = request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response({ "user_id": serializer.data['id'], "status": "ok", "message": "Registered successfully, please login" })
        else:
            return Response({ "status": "Registration failed", "message": str(serializer.errors) })

class LoginView(APIView):
    def post(self, request, *args, **kwargs):
        user = getLoginUser(request)
        if user == None:
            return Response({
                "status": "Login failed", 
                "message": f"No user with the corresponding username and password exists"
                })
        return Response({ "status": "ok"})

def getLoginUser(request):
    data = request.data
    email = data.get('email', None)
    password = data.get('password', None)
    try:
        user = User.objects.get(email = email, password = password)
        print(user)
        return user
    except:
        return None