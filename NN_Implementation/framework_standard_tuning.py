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



def ExecuteTuning(runSpecification, datasets, xTrainRaw, yTrain, xDevRaw, yDev):
        startTime = time.time()

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

        # Step 4 Training NN model
        # MODEL_PATH = "ff_" + str(runSpecification['num_features']) + "_" + str(runSpecification['context']) + ".pt"
        MODEL_PATH = "ff2" + "_" + str(runSpecification['h1']) + "_" + str(runSpecification['h2']) + "_" + str(runSpecification['learning_rate'])
        doTraining = False
        # try:
        #     l_model = torch.load(MODEL_PATH)
        # except:
        #     l_model = nn_model.NeuralNetwork(input_nodes=runSpecification['num_features'], layer1=runSpecification["h1"], layer2=runSpecification['h2'])
        #     l_model.train_model_persample(xTrainDoc, yTrain, learning_rate=runSpecification["learning_rate"])
        #     torch.save(l_model, MODEL_PATH)

        l_model = nn_model.NeuralNetwork(input_nodes=runSpecification['num_features'], layer1=runSpecification["h1"], layer2=runSpecification['h2'])
        l_model.train_model_persample(xTrainDoc, yTrain, learning_rate=runSpecification["learning_rate"])
        torch.save(l_model, MODEL_PATH)
        
        # Step 5 Evaluate performance
        yTrainingPredicted = l_model.predict(xTrainDoc)
        training_accuracy = nn_tools.Accuracy(yTrain, yTrainingPredicted)

        yValidatePredicted = l_model.predict(xDevDoc)
        dev_accuracy = nn_tools.Accuracy(yDev, yValidatePredicted)
        
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

yTrain = torch.Tensor([y_val for y_val in dataset_A["sentiment"]])
yDev = torch.Tensor([y_val for y_val in dataset_B["sentiment"]])
xTrainRaw = getCleanReviews(dataset_A)
xDevRaw = getCleanReviews(dataset_B)

# Step 2 Define sweep range
evaluationRunSpecifications = []

# for num_features in [100, 150, 200, 250, 300]:
# for num_features in [350, 400, 450]:
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = num_features
#     runSpecification['context'] = 5
#     runSpecification['optimizing'] = 'num_features'
            
#     evaluationRunSpecifications.append(runSpecification)

# for context in range(4,5):
# # for context in range(1,2):    
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = 150
#     runSpecification['context'] = context
#     runSpecification['optimizing'] = 'context'
            
#     evaluationRunSpecifications.append(runSpecification)

# for h1 in [40, 30, 25, 20, 15, 10]:    
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = 350
#     runSpecification['context'] = 10
#     runSpecification['h1'] = h1
#     runSpecification['h2'] = 10
#     runSpecification['learning_rate'] = 1
#     runSpecification['optimizing'] = 'h1'
#     evaluationRunSpecifications.append(runSpecification)

# for h2 in [10, 8, 6, 4]:    
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = 350
#     runSpecification['context'] = 10
#     runSpecification['h1'] = 14
#     runSpecification['h2'] = h2
#     runSpecification['learning_rate'] = 1
#     runSpecification['optimizing'] = 'h2'
#     evaluationRunSpecifications.append(runSpecification)

# for lr in [1, 0.5, 0.1, 0.05, 0.01]:
# # for lr in [1]:    
#     runSpecification = {}

#     runSpecification['model'] = 'Standard Neural Network'
#     runSpecification['num_features'] = 350
#     runSpecification['context'] = 10
#     runSpecification['h1'] = 14
#     runSpecification['h2'] = 10
#     runSpecification['learning_rate'] = lr
#     runSpecification['optimizing'] = 'learning_rate'

#     evaluationRunSpecifications.append(runSpecification)

 
runSpecification = {}

runSpecification['model'] = 'Standard Neural Network'
runSpecification['num_features'] = 350
runSpecification['context'] = 10
runSpecification['h1'] = 10
runSpecification['h2'] = 10
runSpecification['learning_rate'] = 0.1
runSpecification['optimizing'] = 'learning_rate'

evaluationRunSpecifications.append(runSpecification)

evaluations = Parallel(n_jobs=12)(delayed(ExecuteTuning)(runSpec, datasets, xTrainRaw, yTrain, xDevRaw, yDev) for runSpec in evaluationRunSpecifications)
# evaluations = [ ExecuteTuning(runSpec, datasets, xTrainRaw, yTrain, xDevRaw, yDev) for runSpec in evaluationRunSpecifications ]

for evaluation in evaluations:
    print(evaluation)

nf_train_accuracy = []
nf_dev_accuracy = []
nf_series = []

ct_train_accuracy = []
ct_dev_accuracy = []
ct_series = []

h1_train_accuracy = []
h1_dev_accuracy = []
h1_series = []

h2_train_accuracy = []
h2_dev_accuracy = []
h2_series = []

lr_train_accuracy = []
lr_dev_accuracy = []
lr_series = []

for evaluation in evaluations:
    if evaluation["optimizing"] == "num_features":
        nf_train_accuracy.append(evaluation['train_accuracy'])
        nf_dev_accuracy.append(evaluation['dev_accuracy'])
        nf_series.append(evaluation['num_features'])
    elif evaluation["optimizing"] == "context":
        ct_train_accuracy.append(evaluation['train_accuracy'])
        ct_dev_accuracy.append(evaluation['dev_accuracy'])
        ct_series.append(evaluation['context'])
    elif evaluation["optimizing"] == "h1":
        h1_train_accuracy.append(evaluation['train_accuracy'])
        h1_dev_accuracy.append(evaluation['dev_accuracy'])
        h1_series.append(evaluation['h1'])
    elif evaluation["optimizing"] == "h2":
        h2_train_accuracy.append(evaluation['train_accuracy'])
        h2_dev_accuracy.append(evaluation['dev_accuracy'])
        h2_series.append(evaluation['h2'])
    elif evaluation["optimizing"] == "learning_rate":
        lr_train_accuracy.append(evaluation['train_accuracy'])
        lr_dev_accuracy.append(evaluation['dev_accuracy'])
        lr_series.append(evaluation['learning_rate'])
    

print("number features")
print(nf_train_accuracy)
print(nf_dev_accuracy)
print(nf_series)
print()

print("context")
print(ct_train_accuracy)
print(ct_dev_accuracy)
print(ct_series)

print("h1")
print(h1_train_accuracy)
print(h1_dev_accuracy)
print(h1_series)
print()

print("h2")
print(h2_train_accuracy)
print(h2_dev_accuracy)
print(h2_series)

print("lr")
print(lr_train_accuracy)
print(lr_dev_accuracy)
print(lr_series)
print()



