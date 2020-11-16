import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
# nltk.download('wordnet') - needed to run to download 'wordnet' resource to use lemmatize function

import pandas as pd

def basic_clean(string):
    # lowercase all characters
    string = string.lower()
    # normalize unicode characters
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove anything that is not a through z, a number, a single quote, or whitespace
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str=True)
    return string

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string_stemmed = ' '.join(stems)
    return string_stemmed

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string_lemmatized = ' '.join(lemmas)
    return string_lemmatized

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    if len(extra_words) > 0:
        for word in extra_words:
            stopword_list.append(word)
        print(f'The words {extra_words} have been added to the stopwords list and will not be returned.')
    if len(exclude_words) > 0:
        for word in exclude_words:
            if word in stopword_list:
                stopword_list.remove(word)
        print(f'The words {exclude_words} have been removed from the stopwords list and will be returned.')
    words = string.split()
    filtered_words = [w for w in words if w not in stopword_list]

    print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    print('---')

    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords