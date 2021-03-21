#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility

data_path = '/home/azureuser/nlp_project/code_tut/NLP-Project/dataset/raw/'
output_path = '/home/azureuser/nlp_project/code_tut/NLP-Project/dataset/output/'


# ****** Define functions to create average word vectors
#
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

def getPaddedFeatureVec(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the feature vector for each one return a list of numpy arrays
    # padd with 0's
    
    # Initialize a counter
    reviewLengths = []
    index2word_set = set(model.wv.index2word)
    counter = 0.
    max_sentence = 0
    for review in reviews:
        reviewLengths.append(len(review))
        if max_sentence < len(review):
            max_sentence = len(review)

    num_reviews = len(reviews)
    num_words = max_sentence

    featureArray = np.zeros((num_reviews, num_words, num_features),dtype="float32")
    

    # Loop through the reviews
    for r_index, review in enumerate(reviews):

        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # Determine feature vectors
        for w_index, word in enumerate(review):
            if word in index2word_set:
                featureArray[r_index, w_index, :] = np.add(featureArray[r_index, w_index, :], model[word])
       
        # Increment the counter
        counter = counter + 1.
    
    return featureArray, reviewLengths

def getFeatureList(reviews, model, num_features, num_splits=4):
    # Initialize a counter
    index2word_set = set(model.wv.index2word)
    counter = 0.
    featureVec = np.zeros((num_features,),dtype="float32")
   
    # Try each review as a list of tensors
    feature_list = []
    
    # Loop through the reviews
    for review in reviews:

        # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # Try different lengths here
        review_features = []

        # # All values  
        # # Initialize the list of features
        # for word in review:
        #     review_features.append(featureVec.copy())

        # # Determine feature vectors
        # for index, word in enumerate(review):
        #     if word in index2word_set:
        #         review_features[index] = np.add(review_features[index],model[word])

        # Split into 4
        
        split = chunkIt(review, num_splits)
        for section in split:
            nwords = 0.
            averageVec = featureVec.copy()
            for index, word in enumerate(section):
                if word in index2word_set:
                    nwords = nwords + 1.
                    averageVec = np.add(averageVec,model[word])
                
            featureVec = np.divide(featureVec,nwords)
            review_features.append(averageVec)

        # Increment the counter
        counter = counter + 1.

        feature_list.append(review_features)

    return feature_list

def getSentimentArray(sentiment, useSmall=None):

    if useSmall:
        sentiment = sentiment[0:useSmall]

    sen_length = len(sentiment)
    sentimentArray = np.zeros((sen_length, 1),dtype="long")
    for i, value in enumerate(sentiment):
        sentimentArray[i, 0] = value

    return sentimentArray


def getDocFeatureVec(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    
    # Initialize a counter
    counter = 0.
    
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    # Loop through the reviews
    for review in reviews:
       
       # Print a status message every 1000th review
        if counter%1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
       
        reviewFeatureVecs[int(counter)] = model.infer_vector(review)
       
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def getCleanReviews(reviews, useSmall=False, remove_stopwords=True):
    clean_reviews = []

    if useSmall:
        for review in reviews["review"][0:200]:
            clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=remove_stopwords ))
        return clean_reviews
    else:
        for review in reviews["review"]:
            clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=remove_stopwords ))
        return clean_reviews
