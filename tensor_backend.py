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
from tl_util import save_obj
import jieba
import sys

model = None

model_count = 10

class callback_save_model(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global model
        global model_count
        if model == None:
            pass
        if model_count > 0:
            model_count -= 1
        if model_count <= 0:
            model.save('tl-textmodel-ep' + str(epoch) + '.h5')
            print("\nSaved model at epoch: ",epoch," ...")
            model_count = 10


# using parts of the codes from:
# https://medium.com/blocksurvey/building-multi-class-text-classifier-using-tensorflow-keras-2148586e69ad

print("Done importing.")

train_y = []
classes_count = 254

# path to data 
data_dir = "./data.json"
with open(data_dir) as json_data:
    data = json.load(json_data)

maxLen=0
df=pd.DataFrame(columns=['Text','Class Name'])
#df=pd.DataFrame(columns=['Class Name','Text'])
index=0
for item in data:
    for element in data[item]:
        if maxLen<len(element):
            maxLen=len(element)
        df=df.append(pd.Series([element,item],index=df.columns),ignore_index=True)
print("Max length supported: ",maxLen)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['Class Name'])

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

train_y = onehot_encoded
train_y = np.array(train_y)

# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[1, :])])

train_dataset, test_dataset = df,df[1000:]
#train_dataset, test_dataset = df,df

# the 4 lines of codes below randomize the input
#row_count = df.shape[0]
#df = df.sample(frac=1).reset_index(drop=True)
#train_dataset = df
#test_dataset  = df[:int(row_count*0.2)]

train_row_count = train_dataset.shape[0]
test_row_count = test_dataset.shape[0]
print("Train: ",train_row_count, ", Test: ", test_row_count)


vocab_size = 13000
embedding_dim = 16
max_length = maxLen
trunc_type='post'
oov_tok = '<OOV>'
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_dataset['Text'])
save_obj(tokenizer, "tl_tokenizer")
print("Training tokenizer saved for prediction.")


word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_dataset['Text'])

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_dataset['Class Name'])
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

print("Creating Model...")
model = tf.keras.Sequential([
    #tf.keras.layers.Embedding(vocab_size, 13),
    tf.keras.layers.Embedding(vocab_size, classes_count),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(13, activation='softmax')
    tf.keras.layers.Dense(classes_count, activation='softmax')
    ])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
NUM_EPOCHS = 200
model.fit(padded, train_y, epochs=NUM_EPOCHS,batch_size=10,
        callbacks=[callback_save_model()])
# history = model.fit(padded,train_y, epochs=NUM_EPOCHS
# , validation_data=(testing_padded,test_y))
model.evaluate(padded, train_y)

model.save('tl-textmodel-ends.h5')


