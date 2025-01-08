import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from data4_2 import *
from train4_3 import *

# Initialize stemmer and stopwords
snow = nltk.stem.SnowballStemmer('english')
# Preprocessing function
def preprocesamiento_words(sentence):       
    sentence = sentence.lower() 
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)
    cleanr = re.compile(r'(http|https)://[^\s]*')
    sentence = re.sub(cleanr, 'httpaddr', sentence)
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)
    cleanr = re.compile(r'[^\s]+@[^\s]+.com')
    sentence = re.sub(cleanr, 'emailaddr', sentence)
    cleanr = re.compile('[0-9]+')
    sentence = re.sub(cleanr, 'number', sentence)
    cleanr = re.compile('[$]+')
    sentence = re.sub(cleanr, 'dollar', sentence)
    cleanr = re.compile('[^a-zA-Z0-9]')
    sentence = re.sub(cleanr, ' ', sentence)
    cleanr = re.compile('subject')
    sentence = re.sub(cleanr, ' ', sentence)
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]
    return words

def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            idx, word = line.strip().split(': ')
            vocab[word] = int(idx)
            
    return vocab

def vectorize(text, vocab):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([text])
    return X

# load data
datafile = '/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/mail.txt'
with open(datafile, 'r') as f:
    text = f.readlines()
    
# preprocess data
text_list = []
for sentence in text:
    text_list.append(preprocesamiento_words(sentence))
    
for text in text_list:
    mail = ' '.join(text)

# load vocabulary
vocab_file = '/home/sprutz/dev/mlp/HW3/HW3_data/P4_files/vocab.txt'
vocab = load_vocab(vocab_file)

# vectorize data
X = vectorize(mail, vocab)
X = X.toarray()
X = X-mean_vec
X = np.dot(X, V.T)

weight_file = '/home/sprutz/dev/mlp/HW3/models/q4weights.npy'
weights = np.load(weight_file)

# predict
z = np.dot(X, weights)
pred = sigmoid(z)
print(pred)
pred = np.round(pred)
print(pred)
