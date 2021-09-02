# Text-Classification

## Introduction

This project is inspired by a typical real-life scenario. Imagine we have been hired as a Data Scientist by a major news organization. Our job is to analyse the news feed to determine the topic of incoming news articles so they can be organized and distributed to our readers. 

For this project, we are given a collection of news articles and also summaries of the same articles. The articles have been manually labelled as one of five topics: __business, entertainment, politics, sport and tech__. 

## Data and Methods

A training dataset is a .tsv (tab separated values) file containing a number of articles, with one article per line, and linebreaks within articles removed. Each line of the .tsv file has three fields: instance number, text and topic (business, entertainment, politics, sport, tech). A test dataset is a .tsv file in the same format as the training dataset except that we ignore the topic field. Training and test datasets can be drawn from supplied files articles.tsv or summaries.tsv.

For all models, we consider an article to be a collection of words, where a word is a string of at least two letters, numbers or the symbols #, @, _, $ or %, delimited by a space, after removing all other characters (two characters is the default minimum word length for CountVectorizer in scikit-learn). URLs are treated as a space, so delimit words.

We will use and test four supervised learning methods: Decision Trees (DT), Bernoulli Naive Bayes (BNB), Multinomial Naive Bayes (MNB) and Multiple Layer Perceptron (MLP).

## Implementation

We implement four Python programs: (i) [DT_classifier.py](/DT_classifier.py), (ii) [BNB_classifier.py](/BNB_classifier.py), (iii) [MNB_classifier.py](/MNB_classifier.py) and (iv) [MLP_classifier.py](/MLP_classifier.py). These programs, when called from the command line with two file names as arguments, the first a training dataset and the second a test dataset, print the instance number and topic produced by the classifier of each article in the test set when trained on the training set (one per line with a space between the instance number and topic) – each topic being the string “business”, “entertainment”, “politics”, “sport” or “tech”.

For example,

```
python3 DT_classifier.py training.tsv test.tsv > output.txt
```

will write to the file output.txt the instance number and topic of each article in test.tsv, as determined by the Decision Tree classifier trained on training.tsv.

More details can be found in [classifier.ipynb](/classifier.ipynb).
