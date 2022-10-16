import numpy as np
import pandas as pd

#EDA and visualization imports:
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

#pre-processing imports:
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from spellchecker import SpellChecker
from wordcloud import WordCloud

#modelling imports:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#evaluation imports:
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

#global parameters
additional  = ['retweet', 've', 'RT']
all_stop = (stopwords.words('english'), additional)

#import and read csv's into Jupyter:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#create a nested function to process tweets in train and test datasets
def tweet_processing(tweet):
      
#     cleaned_tweet = clean(tweet)
    #Function to take every tweet and generate a list of words (hastags and other punctuations removed)
    def clean_tweet(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = clean_tweet(tweet)
    
    #Function to take every tweet and remove stopwords and symbols
    def clean_stopwords_symbols(tweet):
        tweet_list = [ele for ele in tweet.split()]
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$|^RT[\s]+|https?:\/\/.*[\r\n]*', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word not in all_stop]
        return clean_mess
    no_punc_tweet = clean_stopwords_symbols(new_tweet)
    
    #Function to lemmatize the words in each tweet
    def lemma(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    return lemma(no_punc_tweet)

#split X and y data for modelling
X = train['message']
y = train['sentiment']
testX = test['message']

#train_test_split on data for modelling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

#create the final Pipeline with pre-processing, weighing and modelling combined into a few lines of code.
best_pipe = Pipeline([
    ('vect',CountVectorizer(analyzer=tweet_processing)),  #tokenize the tweets
    ('tfidf', TfidfTransformer()), #weight the classes
    ('classifier', LinearSVC()),
])
best_pipe.fit(X, y)

#make predictions from from fitted model
y_pred = best_pipe.predict(testX)

#create test sentiment column from predictions
test['sentiment'] = y_pred.tolist()

#subset columns for output format
df_final_sub = test[['tweetid', 'sentiment']]

#Export prediction data to .csv for Kaggle submission
df_final_sub.to_csv('final_prediction.csv', index=False)