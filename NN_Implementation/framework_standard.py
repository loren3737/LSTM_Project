import time
import numpy
import torch 
import nn_model
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
yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])

# Get word2vec
# xTrain = getAvgFeatureVecs(getCleanReviews(dataset_A), word2vec_model, NUM_FEATURES)
# xTrain = torch.tensor(xTrain)

# Get doc2vec
xTrainDoc = getDocFeatureVec(getCleanReviews(dataset_A), doc2vec_model, NUM_FEATURES)
xTrainDoc = torch.tensor(xTrainDoc)

# Step 4 Training NN model
MODEL_PATH = "feedforward_model.pt"
doTraining = False
if doTraining:
    print("TRAINING nn model")
    l_model = nn_model.NeuralNetwork(input_nodes=NUM_FEATURES)
    l_model.train_model_persample(xTrainDoc, yTrain)
    torch.save(l_model, MODEL_PATH)
else:
    l_model = torch.load(MODEL_PATH)

# Step 5 Evaluate performance
print()
print("EVALUATION")
yTrainingPredicted = l_model.predict(xTrainDoc)
training_accuracy = nn_tools.Accuracy(yTrain, yTrainingPredicted)
print("Training Accuracy: " + str(training_accuracy))

xDevDoc = getDocFeatureVec(getCleanReviews(dataset_B), doc2vec_model, NUM_FEATURES)
xDevDoc = torch.tensor(xDevDoc)
yValidatePredicted = l_model.predict(xDevDoc)
dev_accuracy = nn_tools.Accuracy(yDev, yValidatePredicted)
print("Development Accuracy: " + str(dev_accuracy))

# Step 6 Generate ROC curve
(modelFPRs, modelFNRs, thresholds) = nn_tools.TabulateModelPerformanceForROC(l_model, xDevDoc, yDev)
print(modelFPRs)
print(modelFNRs)
print(thresholds)
