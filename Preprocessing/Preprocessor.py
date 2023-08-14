import re

import nltk
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import PorterStemmer
ps = PorterStemmer()

from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
nltk.download('wordnet')

# ## NLP Workflow

# - Lower casing
# - Tokenization
# - Punctuation removal
# - Stopwords removal
# - Stemming
# - Lemmatization

def preprocess(text):

    #remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)

    #lowercase
    text = text.lower()

    #tokenization
    words = nltk.word_tokenize(text)

    #punctuation mark removal
    words = [word for word in words if word.isalnum()]

    #stopwords removal
    words_stop = []
    for word in words:
        if word not in stopwords.words('english'):
            words_stop.append(word)

    #stemming
    words_stem = []
    for word in words_stop:
        words_stem.append(ps.stem(word))

    #lemmatization
    words_lemmatized = []
    for word in words_stem:
        words_lemmatized.append(lm.lemmatize(word))

    #join words
    text = ' '.join(words_lemmatized)

    return text