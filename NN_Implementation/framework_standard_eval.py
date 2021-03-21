import time
import numpy
import torch 
import nn_model
import nn_tools
import KaggleWord2VecUtility
from gensim.models import Word2Vec
from gen_Doc2Vec_adjustable import generate_doc2vec
from gen_Word2Vev_adjustable import generate_word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from KaggleVectorize import getAvgFeatureVecs, getCleanReviews, getDocFeatureVec
from joblib import Parallel, delayed

runSpecification = {}
runSpecification['model'] = 'Standard Neural Network'
runSpecification['num_features'] = 350
runSpecification['context'] = 10
runSpecification['h1'] = 10
runSpecification['h2'] = 10
runSpecification['learning_rate'] = 0.1
runSpecification['optimizing'] = 'learning_rate'

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_B = pd.read_csv( "dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
dataset_D = pd.read_csv( "dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
dataset_C = pd.read_csv( "dataset/processed/C.tsv", header=0, delimiter="\t", quoting=3 )
dataset_E = pd.read_csv( "dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
datasets = [dataset_A, dataset_D, dataset_E]

yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])
yTest = torch.Tensor([y_val for y_val in dataset_C["sentiment"]])
xTrainRaw = getCleanReviews(dataset_A)
xDevRaw = getCleanReviews(dataset_B)
xTestRaw = getCleanReviews(dataset_C)


# Look for the model we need
doc2vec_name = "doc2vec_tuning" + "_" + str(runSpecification['num_features']) + "_" + str(runSpecification['context'])
print(doc2vec_name)

try:
    print("LOADING Doc2Vec Model")
    doc2vec_model = Doc2Vec.load(doc2vec_name)
except Exception as ex:
    print(ex)
    print("Could't load Doc2Vec, BUILDING...")
    doc2vec_model = generate_doc2vec(doc2vec_name, datasets, runSpecification['num_features'], runSpecification['context'])

# Get doc2vec
xTrainDoc = getDocFeatureVec(xTrainRaw, doc2vec_model, runSpecification['num_features'])
xTrainDoc = torch.tensor(xTrainDoc)
xDevDoc = getDocFeatureVec(xDevRaw, doc2vec_model, runSpecification['num_features'])
xDevDoc = torch.tensor(xDevDoc)
xTestDoc = getDocFeatureVec(xTestRaw, doc2vec_model, runSpecification['num_features'])
xTestDoc = torch.tensor(xTestDoc)

# Step 4 Training NN model
# MODEL_PATH = "ff2" + "_" + str(runSpecification['h1']) + "_" + str(runSpecification['h2']) + "_" + str(runSpecification['learning_rate'])
# doTraining = False
# try:
#     l_model = torch.load(MODEL_PATH)
# except:
#     l_model = nn_model.NeuralNetwork(input_nodes=runSpecification['num_features'], layer1=runSpecification["h1"], layer2=runSpecification['h2'])
#     l_model.train_model_persample(xTrainDoc, yTrain, learning_rate=runSpecification["learning_rate"])
#     torch.save(l_model, MODEL_PATH)

l_model = nn_model.NeuralNetwork(input_nodes=runSpecification['num_features'], layer1=runSpecification["h1"], layer2=runSpecification['h2'])
# l_model.train_model_persample(xTrainDoc, yTrain, learning_rate=runSpecification["learning_rate"])
l_model.train_model_persample_dev(xTrainDoc, yTrain, xDevDoc, yDev, learning_rate=runSpecification["learning_rate"])

# Step 5 Evaluate performance
yTrainingPredicted = l_model.predict(xTrainDoc)
training_accuracy = nn_tools.Accuracy(yTrain, yTrainingPredicted)

yValidatePredicted = l_model.predict(xDevDoc)
print("Prediction Output")
print(yValidatePredicted)
dev_accuracy = nn_tools.Accuracy(yDev, yValidatePredicted)

yTestPredicted = l_model.predict(xTestDoc)
test_accuracy = nn_tools.Accuracy(yTest, yTestPredicted)

runSpecification['train_accuracy'] = training_accuracy
runSpecification['dev_accuracy'] = dev_accuracy
runSpecification['test_accuracy'] = test_accuracy

print(training_accuracy)
print(dev_accuracy)
print(test_accuracy)

# print("ROC")
# (modelFPRs, modelFNRs, thresholds) = nn_tools.TabulateModelPerformanceForROC(l_model, xDevDoc, yDev)
# print(modelFPRs)
# print(modelFNRs)
# print(thresholds)





