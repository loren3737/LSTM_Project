# import time
import numpy
import torch 
import nn_model, nn_recurrent_model, nn_tools
import KaggleWord2VecUtility
from gensim.models import Word2Vec
from gen_Doc2Vec import generate_doc2vec
from gen_Word2Vev import generate_word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import pandas as pd
from KaggleVectorize import getAvgFeatureVecs, getCleanReviews, getDocFeatureVec, getFeatureList, getSentimentArray
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

### CONFIGURATION ###################
generate_embedding_word = False
generate_embedding_doc = False
word2vec_name = "300features_40minwords_10context"
doc2vec_name = "doc2vec_300features_40minwords_10context"
NUM_FEATURES = 100
#####################################

print()
print("Sentiment Analysis - Neural Network using Word2Vec")
print()

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_B = pd.read_csv( "dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
dataset_D = pd.read_csv( "dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
dataset_E = pd.read_csv( "dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_A = pd.read_csv( "NLP-Project/dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_B = pd.read_csv( "NLP-Project/dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_D = pd.read_csv( "NLP-Project/dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_E = pd.read_csv( "NLP-Project/dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_B = pd.read_csv( "../../dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_D = pd.read_csv( "../../dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
# dataset_E = pd.read_csv( "../../dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
datasets = [dataset_A, dataset_D, dataset_E]

# Step 2 Generate or Load Word2Vec/Doc2Vec
if generate_embedding_word:
    print("CREATING Word2Vec Model")
    word2vec_model = generate_word2vec(word2vec_name, datasets)
else:
    try:
        print("LOADING Word2Vec Model")
        word2vec_model = Word2Vec.load(word2vec_name)
    except Exception as ex:
        print(ex)
        print("Couldn't load Word2Vec, BUILDING...")
        word2vec_model = generate_word2vec(word2vec_name, datasets)

if generate_embedding_doc:
    print("CREATING Doc2Vec Model")
    doc2vec_model = generate_doc2vec(doc2vec_name, datasets)
else:
    try:
        print("LOADING Doc2Vec Model")
        doc2vec_model = Doc2Vec.load(doc2vec_name)
    except Exception as ex:
        print(ex)
        print("Could't load Doc2Vec, BUILDING...")
        doc2vec_model = generate_doc2vec(doc2vec_name, datasets)
        
# Step 3 Gather Features
print("FEATURIZING")

# Labels 
# yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"][0:200]])
yTrain = getSentimentArray(dataset_A["sentiment"], useSmall=None)
yTrain_list = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = getSentimentArray(dataset_B["sentiment"], useSmall=None)
yDev_list = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])

# Get word2vec
xTrain = getFeatureList(getCleanReviews(dataset_A, useSmall=None), word2vec_model, NUM_FEATURES, num_splits=4)
xDev = getFeatureList(getCleanReviews(dataset_B, useSmall=None), word2vec_model, NUM_FEATURES, num_splits=4)

# Step 4 Training NN model
MODEL_PATH = "recurrent_model.pt"
doTraining = True
if doTraining:
    print("TRAINING recurrent nn model")
    r_model = nn_recurrent_model.RNN(input_size=NUM_FEATURES)
    r_model.train(xTrain, yTrain, learning_rate = 0.01, epochs=4)
    torch.save(r_model,  MODEL_PATH)
else:
    r_model = torch.load(MODEL_PATH)

# Step 5 Evaluate performance
print()
print("EVALUATION")
yTrainingPredicted = r_model.predict(xTrain)
training_accuracy = nn_tools.Accuracy(yTrain_list, yTrainingPredicted)
print("Training Accuracy: " + str(training_accuracy))

yValidatePredicted = r_model.predict(xDev)
dev_accuracy = nn_tools.Accuracy(yDev_list, yValidatePredicted)
print("Development Accuracy: " + str(dev_accuracy))

# Step 6 Generate ROC curve
(modelFPRs, modelFNRs, thresholds) = nn_tools.TabulateModelPerformanceForROC(r_model, xDev, yDev_list)
