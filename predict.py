import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# hyper params, should use the same as used in train
sentence_length = 20
lstm_size = 20
confidence = 0.4  # suggested 0.5

def getTags(text):
    # input a sentence or string
    # return a set of tags in the input


    # load model, tokenizer, and encoder
    model = load_model('30model.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    with open('onehot.pickle', 'rb') as o:
        one_hot = pickle.load(o)
    text = [text]

    # encode and pad
    encoded_sent = tokenizer.texts_to_sequences(text)
    padded_sent = pad_sequences(encoded_sent, sentence_length, padding='post')

    # predict and decode to original tags(classes)
    preds = model.predict(padded_sent)
    y_classes = (preds > confidence).astype(int)
    tags = one_hot.inverse_transform(y_classes)
    return tags
