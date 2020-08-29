import sys
if len(sys.argv) != 2:
    print("Usage: predictor /path/to/file.txt")
    sys.exit()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import jieba
from tl_util import get_token, load_obj, num_to_name
from predict_data_reprocessing import file_to_input_string


using_model_path = "./tl-textmodel-ep149.h5"
using_tokenizer_path = "./tl_tokenizer"
tokens_dict = None
tokens_index = 90
max_length = 43438  # max len in current data
trunc_type='post'
input_filepath = str(sys.argv[1])
print("Converting file for engine to process...")
input_string = file_to_input_string( input_filepath )

#text = ["我","是","跨性别"]                                 # text(list) should be the output of jieba
#text = [""]                                 
#text_list = []
#for item in text:
#    token = get_token(item,tokens_dict,tokens_index)
#    text_list.append( token )
#
#token_list_str = [str(integer) for integer in text_list]    # from [1,2,3] to ['1','2','3']
#token_string = ' '.join(token_list_str)                     # so that we could call join()

token_string = input_string

print("Loading tokenizer from training session...")
tokenizer = load_obj(using_tokenizer_path)

print("Loading trained model: ", using_model_path)
model = tf.keras.models.load_model(using_model_path)
model.summary()

padded_text = pad_sequences(tokenizer.texts_to_sequences([token_string]),
        maxlen=max_length,
        truncating=trunc_type)
predictotron = model.predict( padded_text )
indices = None
for item in predictotron:                                   # item is a single list, corresponding to a single input
    np_array = np.array( item )
    #indices = np.argpartition(np_array, -10)[-10:]
    indices = np_array.argsort()[-12:][::-1]

    indices = indices.tolist()
    print("\n\n>>--------------------------------")
    for item in indices:
        print( num_to_name(item) )
    print("--------------------------------<<\n")

