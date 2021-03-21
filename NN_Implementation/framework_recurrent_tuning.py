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


def ExecuteTuning(runSpecification, datasets, xTrainRaw, yTrain, yTrainList, xDevRaw, yDev, yDevList):
        startTime = time.time()

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

        # Step 4 Training NN model
        # MODEL_PATH = "ff_" + str(runSpecification['num_features']) + "_" + str(runSpecification['context']) + ".pt"
        MODEL_PATH = "rnn" + "_" + str(runSpecification['num_features']) + "_" + str(runSpecification['num_splits']) + "_" + str(runSpecification['hidden_size'])
        
        try:
            r_model = torch.load(MODEL_PATH)
        except:
            r_model = nn_recurrent_model.RNN(input_size=runSpecification['num_features'], hidden_size=runSpecification['hidden_size'],)
            r_model.train(xTrain, yTrain, learning_rate = 0.01, epochs=10)
            torch.save(r_model,  MODEL_PATH)

        # Step 5 Evaluate performance
        yTrainingPredicted = r_model.predict(xTrain)
        training_accuracy = nn_tools.Accuracy(yTrainList, yTrainingPredicted)

        yValidatePredicted = r_model.predict(xDev)
        dev_accuracy = nn_tools.Accuracy(yDevList, yValidatePredicted)
        
        runSpecification['train_accuracy'] = training_accuracy
        runSpecification['dev_accuracy'] = dev_accuracy

        endTime = time.time()
        runSpecification['runtime'] = endTime - startTime
        
        return runSpecification

# Step 1 Load in data
print("LOADING data from CSV")
dataset_A = pd.read_csv( "dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
dataset_B = pd.read_csv( "dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
dataset_D = pd.read_csv( "dataset/processed/D.tsv", header=0, delimiter="\t", quoting=3 )
dataset_E = pd.read_csv( "dataset/processed/E.tsv", header=0, delimiter="\t", quoting=3 )
datasets = [dataset_A, dataset_D, dataset_E]

yTrain = getSentimentArray(dataset_A["sentiment"], useSmall=None)
yTrain_list = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = getSentimentArray(dataset_B["sentiment"], useSmall=None)
yDev_list = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])

# yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
# yDev = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])
xTrainRaw = getCleanReviews(dataset_A)
xDevRaw = getCleanReviews(dataset_B)

# Step 2 Define sweep range
evaluationRunSpecifications = []

# for nf in [100, 200, 300, 350, 400]:
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = nf
#     runSpecification['context'] = 5
#     runSpecification['num_splits'] = 4
#     runSpecification['hidden_size'] = 40
#     runSpecification['optimizing'] = 'num_features'

#     evaluationRunSpecifications.append(runSpecification)

# for context in range(1,11):
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = 350
#     runSpecification['context'] = context
#     runSpecification['num_splits'] = 4
#     runSpecification['hidden_size'] = 40
#     runSpecification['optimizing'] = 'context'

#     evaluationRunSpecifications.append(runSpecification)

for hs in [5, 10, 15, 20, 25,30]:
    runSpecification = {}

    runSpecification['model'] = 'Standard Neural Network'
    runSpecification['num_features'] = 400
    runSpecification['context'] = 5
    runSpecification['num_splits'] = 4
    runSpecification['hidden_size'] = hs
    runSpecification['optimizing'] = 'hidden_size'

    evaluationRunSpecifications.append(runSpecification)

for ns in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    runSpecification = {}

    runSpecification['model'] = 'Standard Neural Network'
    runSpecification['num_features'] = 400
    runSpecification['context'] = 5
    runSpecification['num_splits'] = ns
    runSpecification['hidden_size'] = 20
    runSpecification['optimizing'] = 'num_splits'

    evaluationRunSpecifications.append(runSpecification)

evaluations = Parallel(n_jobs=12)(delayed(ExecuteTuning)(runSpec, datasets, xTrainRaw, yTrain, yTrain_list, xDevRaw, yDev, yDev_list) for runSpec in evaluationRunSpecifications)
# evaluations = [ ExecuteTuning(runSpec, datasets, xTrainRaw, yTrain, yTrain_list, xDevRaw, yDev, yDev_list) for runSpec in evaluationRunSpecifications ]

for evaluation in evaluations:
    print(evaluation)

nf_train_accuracy = []
nf_dev_accuracy = []
nf_series = []

ct_train_accuracy = []
ct_dev_accuracy = []
ct_series = []

hs_train_accuracy = []
hs_dev_accuracy = []
hs_series = []

ns_train_accuracy = []
ns_dev_accuracy = []
ns_series = []

for evaluation in evaluations:
    if evaluation["optimizing"] == "num_features":
        nf_train_accuracy.append(evaluation['train_accuracy'])
        nf_dev_accuracy.append(evaluation['dev_accuracy'])
        nf_series.append(evaluation['num_features'])
    elif evaluation["optimizing"] == "context":
        ct_train_accuracy.append(evaluation['train_accuracy'])
        ct_dev_accuracy.append(evaluation['dev_accuracy'])
        ct_series.append(evaluation['context'])
    elif evaluation["optimizing"] == "hidden_size":
        hs_train_accuracy.append(evaluation['train_accuracy'])
        hs_dev_accuracy.append(evaluation['dev_accuracy'])
        hs_series.append(evaluation['hidden_size'])
    elif evaluation["optimizing"] == "num_splits":
        ns_train_accuracy.append(evaluation['train_accuracy'])
        ns_dev_accuracy.append(evaluation['dev_accuracy'])
        ns_series.append(evaluation['num_splits'])
 

print("number features")
print(nf_train_accuracy)
print(nf_dev_accuracy)
print(nf_series)
print()

print("context")
print(ct_train_accuracy)
print(ct_dev_accuracy)
print(ct_series)

print("hidden_size")
print(hs_train_accuracy)
print(hs_dev_accuracy)
print(hs_series)
print()

print("num_splits")
print(ns_train_accuracy)
print(ns_dev_accuracy)
print(ns_series)




