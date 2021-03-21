import time
import numpy
import torch 
import nn_model
import nn_recurrent_model
import nn_tools
import KaggleWord2VecUtility
from gensim.models import Word2Vec
from gen_Doc2Vec_adjustable import generate_doc2vec
from gen_Word2Vev_adjustable import generate_word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from KaggleVectorize import getAvgFeatureVecs, getCleanReviews, getDocFeatureVec, getFeatureList, getSentimentArray
from joblib import Parallel, delayed

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_B = pd.read_csv( "dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
dataset_C = pd.read_csv( "dataset/processed/C.tsv", header=0, delimiter="\t", quoting=3 )
dataset_D = pd.read_csv( "dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
dataset_E = pd.read_csv( "dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
datasets = [dataset_A, dataset_D, dataset_E]

yTrain = getSentimentArray(dataset_A["sentiment"], useSmall=None)
yTrainList = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = getSentimentArray(dataset_B["sentiment"], useSmall=None)
yDevList = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])
yTest = getSentimentArray(dataset_C["sentiment"], useSmall=None)
yTestList = torch.Tensor([y_val for y_val in dataset_C["sentiment"]])

xTrainRaw = getCleanReviews(dataset_A)
xDevRaw = getCleanReviews(dataset_B)
xTestRaw = getCleanReviews(dataset_C)

runSpecification = {}

runSpecification['model'] = 'Standard Neural Network'
runSpecification['num_features'] = 400
runSpecification['context'] = 5
runSpecification['num_splits'] = 5
runSpecification['hidden_size'] = 25
runSpecification['optimizing'] = 'final'

# Look for the model we need
word2vec_name = "word2vec_tuning" + "_" + str(runSpecification['num_features']) + "_" + str(runSpecification['context'])

print(word2vec_name)

try:
    print("LOADING Word2Vec Model")
    word2vec_model = Word2Vec.load(word2vec_name)
except Exception as ex:
    print(ex)
    print("Could't load Word2Vec, BUILDING...")
    word2vec_model = generate_word2vec(word2vec_name, datasets, runSpecification['num_features'], runSpecification['context'])

# Get doc2vec
xTrain = getFeatureList(xTrainRaw, word2vec_model, runSpecification['num_features'], num_splits=runSpecification['num_splits'])
xDev = getFeatureList(xDevRaw, word2vec_model, runSpecification['num_features'], num_splits=runSpecification['num_splits'])
xTest = getFeatureList(xTestRaw, word2vec_model, runSpecification['num_features'], num_splits=runSpecification['num_splits'])

# Step 4 Training NN model
# MODEL_PATH = "rnn" + "_" + str(runSpecification['num_features']) + "_" + str(runSpecification['num_splits']) + "_" + str(runSpecification['hidden_size'])

# Getting an error from loading
# try:
#     r_model = torch.load(MODEL_PATH)
# except:

r_model = nn_recurrent_model.RNN(input_size=runSpecification['num_features'], hidden_size=runSpecification['hidden_size'],)
# r_model.train(xTrain, yTrain, learning_rate = 0.01, epochs=30)
r_model.train_dev(xTrain, yTrain, xDev, yDevList, learning_rate = 0.01, epochs=30)


# Step 5 Evaluate performance
yTrainingPredicted = r_model.predict(xTrain)
training_accuracy = nn_tools.Accuracy(yTrainList, yTrainingPredicted)

yValidatePredicted = r_model.predict(xDev)
print("Prediction Output")
print(yValidatePredicted)
dev_accuracy = nn_tools.Accuracy(yDevList, yValidatePredicted)

yTestPredicted = r_model.predict(xTest)
test_accuracy = nn_tools.Accuracy(yTestList, yTestPredicted)

print(training_accuracy)
print(dev_accuracy)
print(test_accuracy)

# print("ROC")
# (modelFPRs, modelFNRs, thresholds) = nn_tools.TabulateModelPerformanceForROC(r_model, xDev, yDevList)
# print(modelFPRs)
# print(modelFNRs)
# print(thresholds)




