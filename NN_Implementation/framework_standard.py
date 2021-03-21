import time
import numpy
import torch 
import nn_model, nn_model_instructor
import nn_tools
import KaggleWord2VecUtility
from gensim.models import Word2Vec
from gen_Doc2Vec import generate_doc2vec
from gen_Word2Vev import generate_word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from KaggleVectorize import getAvgFeatureVecs, getCleanReviews, getDocFeatureVec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

### CONFIGURATION ###################
load_embedding_doc = True
doc2vec_name = "model_output/doc2vec_feedforward_standard"
ffnn_name = "model_output/feedforward_model.pt"
load_nn = False
#####################################

print()
print("Sentiment Analysis - Neural Network using Word2Vec")
print()

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_B = pd.read_csv( "dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_A = dataset_A[0:200]
# dataset_B = dataset_B[0:200]

# Step 2 Generate or Load Word2Vec/Doc2Vec
try:
    print("LOADING Doc2Vec Model")
    assert(load_embedding_doc)
    doc2vec_model = Doc2Vec.load(doc2vec_name)
except Exception as ex:
    print(ex)
    print("Could't load Doc2Vec, BUILDING...")
    dataset_D = pd.read_csv( "dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
    dataset_E = pd.read_csv( "dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
    datasets = [dataset_A, dataset_D, dataset_E]
    doc2vec_model = generate_doc2vec(doc2vec_name, datasets)

num_features = doc2vec_model.vector_size
        
# Step 3 Gather Features
print("FEATURIZING")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Labels 
yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yTrain = yTrain.to(device)
yDev = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])
yDev = yDev.to(device)

# Get doc2vec
xTrainDoc = getDocFeatureVec(getCleanReviews(dataset_A), doc2vec_model, num_features)
xTrainDoc = torch.tensor(xTrainDoc)
xTrainDoc = xTrainDoc.to(device)
xDevDoc = getDocFeatureVec(getCleanReviews(dataset_B), doc2vec_model, num_features)
xDevDoc = torch.tensor(xDevDoc)
xDevDoc = xDevDoc.to(device)

# Step 4 Training NN model
try:
    assert(load_nn)
    l_model = torch.load(ffnn_name)
except:
    print("TRAINING nn model")
    l_model = nn_model.NeuralNetwork(input_nodes=num_features)
    l_model.to(device)
    nn_model_instructor.train(l_model, xTrainDoc, yTrain)
    torch.save(l_model, ffnn_name)    

# Step 5 Evaluate performance
print()
print("EVALUATION")
yTrainingPredicted = nn_model_instructor.predict(l_model,xTrainDoc)
training_accuracy = nn_tools.Accuracy(yTrain, yTrainingPredicted)
print("Training Accuracy: " + str(training_accuracy))

# yValidatePredicted = l_model.predict(xDevDoc)
# dev_accuracy = nn_tools.Accuracy(yDev, yValidatePredicted)
# print("Development Accuracy: " + str(dev_accuracy))

# # Step 6 Generate ROC curve
# (modelFPRs, modelFNRs, thresholds) = nn_tools.TabulateModelPerformanceForROC(l_model, xDevDoc, yDev)
# print(modelFPRs)
# print(modelFNRs)
# print(thresholds)


    
