
#Import libraries

import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import string
nltk.download('stopwords') # downloading the stopwords
#reading the dataset
dataset = pd.read_csv("emails.csv")

#droping the dublicate emails
dataset.drop_duplicates(inplace=True)

#show the missing values
print(dataset.isnull().sum)

#cleaning of text stuff !
def processText(text):
    '''
        What will be do:
        1. Remove punctuation
        2. Remove stopwords
        3. Return list of clean text words
        '''
    nonpun  = [] # filtered string without punctuations
    cleaned_string  = [] # filtered string without punctuations & stopwords
    for char in text:
        if char not in string.punctuation:
            nonpun.append(char)
    nonpun = ''.join(nonpun)
    for word in nonpun.split():
        if word.lower() not in stopwords.words('english'):
            cleaned_string.append(word)
    return cleaned_string
# aaply the above function on the dataset
dataset['text'].head().apply(processText)

'''
EXAMPLE OF THE PROCESS TO PREPARE THE DATA FOR TRAINING ON THE CLASSIFIER, 
THIS CELL/BLOCK ISNT NECESSARY TO RUN THE PROGRAM
'''

message4 = 'hello world hello hello world play' #df['text'][3]
message5 = 'test test test test one hello'

#Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(analyzer = processText).fit_transform([[message4],[message5]])
print(bow)

#convert the string into the number
vectorizer = CountVectorizer(analyzer = processText)
# Learn a vocabulary dictionary of all tokens in the raw documents.
bow_transformer = vectorizer.fit(dataset['text'])
messages_bow = bow_transformer.transform(df['text'])

#Convert string to integer counts, learn the vocabulary dictionary and return term-document matrix
#IN SINGLE LINE
#message_bow = CountVectorizer(analyzer=processText).fit_transform(dataset['text'])

#Split the data into 80% training (X_train & y_train) and 20% testing (X_test & y_test) data sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow,dataset['spam'],test_size=.2,random_state=0)

#Training
#Create and train the Naive Bayes classifier
#The multinomial Naive Bayes classifier is suitable for
# classification with discrete features (e.g., word counts for text classification)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB() #gives highest accuracy
classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)

#Evaluation of accuracy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Classification Report : \n{}".format(classification_report(y_test,y_pred)))
print("Accuracy Score : {}".format(accuracy_score(y_test,y_pred)))
print("Confusion Matrix : \n{}".format(confusion_matrix(y_test,y_pred))) # 9 predictions arre wrong



