import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import csv
import re
import sys
from collections import Counter


# fetching data
train_data = pd.read_csv(sys.argv[1], sep='\t', header=None, quoting=csv.QUOTE_NONE)
test_data = pd.read_csv(sys.argv[2], sep='\t', header=None, quoting=csv.QUOTE_NONE)
X_train, Y_train = train_data[1], train_data[2]
X_test, Y_test = test_data[1], test_data[2]

# define my own tokenizer
def my_tokenizer(text):

    # remove illegal words
    text = re.sub(r'[^A-Za-z0-9_#@$% ]','',text)
    # remove urls
    text = re.sub(r'(https?://|www)[^\s]+',' ',text)
    # length >= 2
    return [token for token in text.split() if len(token) >= 2]

# create count vectorizer and fit it with training data
count = CountVectorizer(tokenizer=my_tokenizer, max_features = 1000, lowercase=False)
X_train_bag_of_words = count.fit_transform(X_train)

# transform the test data into bag of words created with fit_transform
X_test_bag_of_words = count.transform(X_test)

# build the model
clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01, criterion='entropy', random_state=0)
clf.fit(X_train_bag_of_words, Y_train)
Y_predicted = clf.predict(X_test_bag_of_words)

# print output
for i, val in enumerate(test_data[0]):
    print(val, Y_predicted[i])
