import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
# nltk.download('wordnet') - needed to run to download 'wordnet' resource to use lemmatize function

import pandas as pd

def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    # lowercase all characters
    string = string.lower()
    # normalize unicode characters
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove anything that is not a through z, a number, a single quote, or whitespace
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''    
    # Create tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    # Use tokenizer
    string = tokenizer.tokenize(string, return_str=True)
    return string

def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    # Join our lists of words into a string again and assign to a variable.
    string_stemmed = ' '.join(stems)
    return string_stemmed

def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # Create the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    # Use the lemmatizer on each word in the list of words we created by using split.    
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    # Join our list of words into a string again and assign to a variable.
    string_lemmatized = ' '.join(lemmas)
    return string_lemmatized

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list
    stopword_list = stopwords.words('english')

    # If the optional list extra_words contains any strings:
    if len(extra_words) > 0:
        for word in extra_words:
            # Append the word to the stopword_list
            stopword_list.append(word)
        print(f'The words {extra_words} have been added to the stopwords list and will not be returned.')
    
    # If the optional list exclude_words contains any strings:
    if len(exclude_words) > 0:
        for word in exclude_words:
            if word in stopword_list:
                # Remove the word from the stopword_list
                stopword_list.remove(word)
        print(f'The words {exclude_words} have been removed from the stopwords list and will be returned.')
    
    # Split words in string
    words = string.split()

    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [w for w in words if w not in stopword_list]

    print('Removed {} stopwords'.format(len(words) - len(filtered_words)))
    print('---')

    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)

    return string_without_stopwords

def prep_article_data(df, column, extra_words=[], exclude_words=[]):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)\
                            .apply(lemmatize)
    
    df['stemmed'] = df[column].apply(basic_clean).apply(stem)
    
    df['lemmatized'] = df[column].apply(basic_clean).apply(lemmatize)
    
    return df[['title', column, 'stemmed', 'lemmatized', 'clean']]