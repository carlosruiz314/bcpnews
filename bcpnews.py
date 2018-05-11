# ===============================
# Check the versions of libraries
# ===============================

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy as sp
print('scipy: {}'.format(sp.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib as mpl
print('matplotlib: {}'.format(mpl.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn as skl
print('sklearn: {}'.format(skl.__version__))

# ===============================
# Load libraries
# ===============================

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
import nltk
from nltk.corpus import stopwords # Language processing library - stopwords
from nltk.tokenize import word_tokenize # Word tokenizer for titles and corpora
from nltk.stem.snowball import SnowballStemmer # Snowball stemming algorithm based on Porter stemming algorithm
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

# 1.2. Number of characters
# 1.2.1. Title
trainSet['title_char_count'] = trainSet.title.str.len()
testSet['title_char_count'] = testSet.title.str.len()

# 1.2.2. Corpus
trainSet['corpus_char_count'] = trainSet.text.str.len()
testSet['corpus_char_count'] = testSet.text.str.len()

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

# 1.4. Number of stopWords
## Before removing the stopwords from the text features, it is a good idea to store the number of stopwords separately, in order to have an additional feature that might be helpful to maximize the predictive power of the model.

## Set stop words:
stopWords = list(stopwords.words('spanish'))
stopWords
# 1.4.1. Title
trainSet['title_stop_count'] = trainSet.title.apply(lambda x: len([word for word in x.split() if word.lower() in stopWords]))
testSet['test_stop_count'] = testSet.title.apply(lambda x: len([word for word in x.split() if word.lower() in stopWords]))

# 1.4.2. Corpus
trainSet['corpus_stop_count'] = trainSet.text.apply(lambda x: len([word for word in x.split() if word.lower() in stopWords]))
testSet['corpus_stop_count'] = testSet.text.apply(lambda x: len([word for word in x.split() if word.lower() in stopWords]))

# Statistical summaries through dataset.describe() are not useful, since the only numeric variables are id and flag.
# Therefore, let's try to obtain some keyword frequencies as a first way to analyze the available information:



## List comprehension for stopword filtering:
trainKeywordList = []
testKeywordList = []

## Train:
for instance in trainSet.keywords:
    trainKeywordList.append([x for x in instance.replace("[","").replace("]","").replace(",","").split(' ') if x not in stopWords])

## Append new column to the trainSet
#trainSet['KeywordList'] = pd.Series(trainKeywordList, index=trainSet.index)

## Test:
for instance in testSet.keywords:
    testKeywordList.append([x for x in instance.replace("[","").replace("]","").replace(",","").split(' ') if x not in stopWords])

## Append new column to the testSet
#testSet['KeywordList'] = pd.Series(testKeywordList, index=testSet.index)

# We should also clean the titles and corpus of every article before analyzing them:
## 1. Word tokenization for title and corpus in train and test with Snowball stemming, since default Porter module does not support Spanish:
stemmer = SnowballStemmer("spanish")
### Let's iterate over every row in the pandas set:
trainTitleTokens = []
testTitleTokens = []
for title in trainSet.title:
    trainTitleTokens.append([stemmer.stem(x.lower()) for x in word_tokenize(title) if x.lower() not in stopWords and x.lower().isalpha()])

for title in testSet.title:
    testTitleTokens.append([stemmer.stem(x.lower()) for x in word_tokenize(title) if x.lower() not in stopWords and x.lower().isalpha()])
