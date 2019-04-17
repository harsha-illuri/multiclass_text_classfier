import pandas as pd
import pickle

import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Embedding, Dropout, MaxPooling1D, LSTM
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
pd.set_option('display.expand_frame_repr', False)
from keras.models import load_model

def cleanText(line):
    # Converting to lower
    line = line.lower()

    # Removing alphanumerics
    tokens = [word for word in line.split() if word.isalpha()]

    # Removing Punctuations
    translator = str.maketrans("", "", string.punctuation)
    tokens = [word.translate(translator) for word in tokens]

    # Removing stop_words
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if not word in stop_words]

    # Removing short_words
    tokens = [word for word in tokens if len(word) > 1]
    return tokens



#load data
pd.set_option('display.expand_frame_repr', False)
# assumes input has first col string and second col set obj with classes
df = pd.read_hdf('dataset.h5', key='data')

# encode classes
one_hot = MultiLabelBinarizer()
Y = one_hot.fit_transform(df['tags'])

# save encoder obj, can be used for prediction
with open('onehot.pickle', 'wb') as handle:
    pickle.dump(one_hot, handle, protocol=pickle.HIGHEST_PROTOCOL)


# hyper params
sentence_length = 20
lstm_size = 20
number_of_classes = len(one_hot.classes_)

# build tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Title'])
vocabulary_size = len(tokenizer.word_index) + 1
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


#embedding matrix
embeddings_index = dict()
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector


# pad sentences
encoded_sent = tokenizer.texts_to_sequences(df['Title'])
padded_sent = pad_sequences(encoded_sent, sentence_length, padding='post')


#model
model = Sequential()
model.add(Embedding(vocabulary_size, 100, input_length=sentence_length, weights=[embedding_matrix], trainable=True))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(lstm_size))
model.add(Dense(number_of_classes, activation='sigmoid'))

# atleast one class match
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# perfect match of all classes
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


history = model.fit(padded_sent, Y, validation_split=0.1, batch_size=512, epochs=10, verbose =1)

model.save('model.h5')


# works only for accuracy not with categorical_accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


