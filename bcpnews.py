# ===============================
# Check the versions of libraries
# ===============================

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

# ===============================
# Load libraries
# ===============================
import inspect # To examine classes
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import Counter
#from itertools import chain
#import sys
import ast
import string
import nltk # Natural Language Toolkit
from nltk.corpus import stopwords # Language processing library - stopwords
from nltk.tokenize import word_tokenize # Word tokenizer for titles and corpora
from nltk.stem.snowball import SnowballStemmer # Snowball stemming algorithm based on Porter stemming algorithm
from nltk import ngrams
import re # regular expressions
# We will need to download additional resources from NLTK:
nltk.download('stopwords')
nltk.download('punkt')

# ===============================
# Load the datasets
# ===============================
filePath = "C:/Users/Bluecap/ProgrammingProjects/bcpnews/"
# Train
trainUrl = filePath + "train.csv"
trainSet = pd.read_csv(trainUrl)
# Test
testUrl = filePath + "test.csv"
testSet = pd.read_csv(testUrl)

# ===============================
# Exploration
# ===============================

# Shape and raw data
## First we will look at the shape of the datasets:
print(trainSet.shape)
print(testSet.shape)

## Let's see how the class balance looks like --> 1 = 25%; 0 = 75%
print(trainSet.groupby('flag').size())

## Now let's print some of the first and last rows to see how the data looks like:
### Train set
trainSet.head(1)

### Test set
testSet.head(1)

# 1. Basic feature extraction using text data
# 1.1. Number of words
# 1.1.1. Title
trainSet['title_word_count'] = trainSet.title.apply(lambda x: len(str(x).split()))
testSet['title_word_count'] = testSet.title.apply(lambda x: len(str(x).split()))

### We will now obtain the frequencies for each number of words in title
Counter(trainSet.title_word_count)
### Some word frequencies in titles seem too small...
trainSet.loc[trainSet.title_word_count == 1]

# 1.1.2. Corpus
trainSet['corpus_word_count'] = trainSet.text.apply(lambda x: len(str(x).split()))
testSet['corpus_word_count'] = testSet.text.apply(lambda x: len(str(x).split()))
Counter(trainSet.corpus_word_count)

# 1.1.3. Summary
trainSet['summary_word_count'] = trainSet.summary.apply(lambda x: len(str(x).split()))
testSet['summary_word_count'] = testSet.summary.apply(lambda x: len(str(x).split()))

# 1.1.4. Keywords
trainSet['keywords_word_count'] = trainSet.keywords.apply(lambda x: len(x))
testSet['keywords_word_count'] = testSet.keywords.apply(lambda x: len(x))

# 1.2. Number of characters
# 1.2.1. Title
trainSet['title_char_count'] = trainSet.title.str.len()
testSet['title_char_count'] = testSet.title.str.len()

# 1.2.2. Corpus
trainSet['corpus_char_count'] = trainSet.text.str.len()
testSet['corpus_char_count'] = testSet.text.str.len()

# 1.2.3. Summary
trainSet['summary_char_count'] = trainSet.summary.str.len()
testSet['summary_char_count'] = testSet.summary.str.len()

# 1.3. Average word length
## First we'll define a function to calculate the average word length for a list of strings:
def avg_word_len(sentence):
    words = sentence.split()
    return sum(len(word) for word in words)/len(words)

# 1.3.1. Title
trainSet['avg_title_word_len'] = trainSet.title.apply(lambda x: avg_word_len(x))
testSet['avg_title_word_len'] = testSet.title.apply(lambda x: avg_word_len(x))

# 1.3.2. Corpus
trainSet['avg_corpus_word_len'] = trainSet.text.apply(lambda x: avg_word_len(x))
testSet['avg_corpus_word_len'] = testSet.text.apply(lambda x: avg_word_len(x))

# 1.3.3. Summary
trainSet['avg_summary_word_len'] = trainSet.summary.apply(lambda x: avg_word_len(x))
testSet['avg_summary_word_len'] = testSet.summary.apply(lambda x: avg_word_len(x))

# 1.3.4. Keywords
trainSet['avg_keyword_word_len'] = trainSet.keywords.apply(lambda x: avg_word_len(re.sub('[^\w\s]', '', x)))
testSet['avg_keyword_word_len'] = testSet.keywords.apply(lambda x: avg_word_len(re.sub('[^\w\s]', '', x)))

# 1.4. Number of stopWords
## Before removing the stopwords from the text features, it is a good idea to store the number of stopwords separately,
## in order to have an additional feature that might be helpful to maximize the predictive power of the model.

## Set stop words (Spanish for the time being, we'll see if English is necessary):
stopWords = list(stopwords.words('spanish')) #+ list(stopwords.words('english'))
#stopWordsEN = list(stopwords.words('english'))

# 1.4.1. Title
trainSet['title_stop_count'] = trainSet.title.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))
testSet['test_stop_count'] = testSet.title.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))

# 1.4.2. Corpus
trainSet['corpus_stop_count'] = trainSet.text.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))
testSet['corpus_stop_count'] = testSet.text.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))

# 1.4.3. Summary
trainSet['summary_stop_count'] = trainSet.summary.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))
testSet['summary_stop_count'] = testSet.summary.apply(lambda x: len([word for word in x.split() if (word.lower() in stopWords)]))

# 1.5. Number of numerics
# Since the topics will most likely be related to economic events, there is a good chance that numerics are relevant
# We will use regular expressions to capture percentages with decimal points

numeric_pattern = r"\d+[\.,]?\d*%?" # It works both with dot and comma-separated decimals

# 1.5.1. Title
trainSet['title_numerics_count'] = trainSet.title.apply(lambda x: len(re.findall(numeric_pattern, x)))
testSet['title_numerics_count'] = testSet.title.apply(lambda x: len(re.findall(numeric_pattern, x)))

# 1.5.2. Corpus
trainSet['corpus_numerics_count'] = trainSet.text.apply(lambda x: len(re.findall(numeric_pattern, x)))
testSet['corpus_numerics_count'] = testSet.text.apply(lambda x: len(re.findall(numeric_pattern, x)))

# 1.5.3. Summary
trainSet['summary_numerics_count'] = trainSet.summary.apply(lambda x: len(re.findall(numeric_pattern, x)))
testSet['summary_numerics_count'] = testSet.summary.apply(lambda x: len(re.findall(numeric_pattern, x)))

# 1.6. Number of uppercase words
# Can't hurt to try

# 1.6.1. Title
trainSet['title_caps_count'] = trainSet.title.apply(lambda x: len([word for word in x.split() if word.isupper()]))
testSet['title_caps_count'] = testSet.title.apply(lambda x: len([word for word in x.split() if word.isupper()]))

# 1.6.2. Corpus
trainSet['corpus_caps_count'] = trainSet.text.apply(lambda x: len([word for word in x.split() if word.isupper()]))
testSet['corpus_caps_count'] = testSet.text.apply(lambda x: len([word for word in x.split() if word.isupper()]))

# 1.6.3. Summary
trainSet['title_caps_count'] = trainSet.summary.apply(lambda x: len([word for word in x.split() if word.isupper()]))
testSet['title_caps_count'] = testSet.summary.apply(lambda x: len([word for word in x.split() if word.isupper()]))

# 1.7. Number of punctuation marks
set_punct = set(string.punctuation)

# 1.7.1. Title
trainSet['title_punct_count'] = trainSet.title.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))
testSet['title_punct_count'] = testSet.title.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))

# 1.7.2. Corpus
trainSet['corpus_punct_count'] = trainSet.text.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))
testSet['corpus_punct_count'] = testSet.text.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))

# 1.7.3. Summary
trainSet['summary_punct_count'] = trainSet.summary.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))
testSet['summary_punct_count'] = testSet.summary.apply(lambda x: len([c for s in x.split() for c in s if c in set_punct]))

# 2. Text preprocessing
# 2.1. Lowercase
## Keywords are already lowercase; we will focus on the rest of the text features

# 2.1.1. Title
trainSet['title'] = trainSet.title.apply(lambda x: " ".join(word.lower() for word in x.split()))
testSet['title'] = testSet.title.apply(lambda x: " ".join(word.lower() for word in x.split()))

# 2.1.2. Corpus
trainSet['text'] = trainSet.text.apply(lambda x: " ".join(word.lower() for word in x.split()))
testSet['text'] = testSet.text.apply(lambda x: " ".join(word.lower() for word in x.split()))

# 2.1.1. Summary
trainSet['summary'] = trainSet.summary.apply(lambda x: " ".join(word.lower() for word in x.split()))
testSet['summary'] = testSet.summary.apply(lambda x: " ".join(word.lower() for word in x.split()))

# 2.2. Remove punctuation
# 2.2.1. Title
trainSet['title'] = trainSet.title.str.replace('[^\w\s]', '')
testSet['title'] = testSet.title.str.replace('[^\w\s]', '')

# 2.2.2. Corpus
trainSet['text'] = trainSet.text.str.replace('[^\w\s]', '')
testSet['text'] = testSet.text.str.replace('[^\w\s]', '')

# 2.2.3. Summary
trainSet['summary'] = trainSet.summary.str.replace('[^\w\s]', '')
testSet['summary'] = testSet.summary.str.replace('[^\w\s]', '')

# 2.2.4. Keywords
trainSet['keywords'] = trainSet.keywords.str.replace('[^\w\s]', '')
testSet['keywords'] = testSet.keywords.str.replace('[^\w\s]', '')

# 2.3. Remove stop words
# 2.3.1. Title
trainSet['title'] = trainSet.title.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))
testSet['title'] = testSet.title.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))

# 2.3.2. Corpus
trainSet['text'] = trainSet.text.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))
testSet['text'] = testSet.text.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))

# 2.3.3. Summary
trainSet['summary'] = trainSet.summary.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))
testSet['summary'] = testSet.summary.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))

# 2.3.4. Keywords
trainSet['keywords'] = trainSet.keywords.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))
testSet['keywords'] = testSet.keywords.apply(lambda x: " ".join([w for w in x.split() if w not in stopWords]))

# 2.4. Most frequent words
# We will check for the top 20 words and try to determine whether they might be useful in this classification problem:

# 2.4.1. Title --> Most frequent words in title seem relevant for the classification problem.
pd.Series(" ".join(trainSet.loc[trainSet.flag == 1].title).split()).value_counts()[:20]
pd.Series(" ".join(trainSet.loc[trainSet.flag == 0].title).split()).value_counts()[:20]

# 2.4.2. Corpus --> It's not such a good idea to remove the most frequent words here either... especially taking into account the class balance.
#                   We could do some cherry-picking, but it doesn't seem like it'll be very helpful.
#                   The algorithm itself will take care of ignoring the few most frequent and irrelevant words
pd.Series(" ".join(trainSet.loc[trainSet.flag == 1].text).split()).value_counts()[:20]
pd.Series(" ".join(trainSet.loc[trainSet.flag == 0].text).split()).value_counts()[:20]

# 2.4.3. Summary --> Same reasoning applies; no reason to go through the trouble of cherry-picking at the moment
pd.Series(" ".join(trainSet.loc[trainSet.flag == 1].summary).split()).value_counts()[:20]
pd.Series(" ".join(trainSet.loc[trainSet.flag == 0].summary).split()).value_counts()[:20]

# 2.5. Least frequent words
# 2.5.1. Title --> None of these seem really relevant... we could delete those words appearing just once.
## First, get the list of words appearing just once
titleLFreq = pd.Series(" ".join(trainSet.title).split()).value_counts()
titleLFreq = list(titleLFreq[titleLFreq < 2].index)

## Now, remove them from train and test
trainSet['title'] = trainSet.title.apply(lambda x: " ".join(w for w in x.split() if w not in titleLFreq))
testSet['title'] = testSet.title.apply(lambda x: " ".join(w for w in x.split() if w not in titleLFreq))

# 2.5.2. Corpus
## First, get the list of words appearing just once
corpusLFreq = pd.Series(" ".join(trainSet.text).split()).value_counts()
corpusLFreq = list(corpusLFreq[corpusLFreq < 2].index)

## Now, remove them from train and test
trainSet['text'] = trainSet.text.apply(lambda x: " ".join(w for w in x.split() if w not in corpusLFreq))
testSet['text'] = testSet.text.apply(lambda x: " ".join(w for w in x.split() if w not in corpusLFreq))

# 2.5.3. Summary
## First, get the list of words appearing just once
summaryLFreq = pd.Series(" ".join(trainSet.summary).split()).value_counts()
summaryLFreq = list(summaryLFreq[summaryLFreq < 2].index)

## Now, remove them from train and test
trainSet['summary'] = trainSet.summary.apply(lambda x: " ".join(w for w in x.split() if w not in summaryLFreq))
testSet['summary'] = testSet.summary.apply(lambda x: " ".join(w for w in x.split() if w not in summaryLFreq))

# Caveats: spelling correction and lemmatization
## There will be no automatic spelling correction or lemmatization, since there are no packages for Spanish.
## Plus, since these are newspaper articles, we may assume they are fairly well written.
## We will just use stemming instead of lemmatization.

# 2.6. Stemming
stemmer = SnowballStemmer("spanish")
# 2.6.1. Title
trainSet['title'] = trainSet.title.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))
testSet['title'] = testSet.title.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))

# 2.6.2. Corpus
trainSet['text'] = trainSet.text.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))
testSet['text'] = testSet.text.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))

# 2.6.3. Summary
trainSet['summary'] = trainSet.summary.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))
testSet['summary'] = testSet.summary.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))

# 2.6.4. Keywords
trainSet['keywords'] = trainSet.keywords.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))
testSet['keywords'] = testSet.keywords.apply(lambda x: " ".join(stemmer.stem(w) for w in x.split()))

# ****************************
# Preprocessing done!
# ****************************

# After completing basic feature extraction and text preprocessing, we may now extract more complex features using NLP

# 3. Advanced text processing for feature extraction
# 3.1. N-grams --> We will be working with bigrams and trigrams; no unigrams.
# 3.1.1. Title
ngrams(trainSet.title[0].split(), 2)

# TODO:
# - Try word tokenization instead of regular split() function.
