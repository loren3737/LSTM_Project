from tqdm import tqdm

# Baseline metrics

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected values. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def AccuracyFromTensors(y, yPredicted):
    correct = (yPredicted == y).float()
    return correct.sum() / len(correct)

def Precision(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    confusion = ConfusionMatrix(y,yPredicted)
    TP = confusion[1][1]
    FP = confusion[0][1]

    #Avoid division by 0
    if TP == 0 and FP == 0:
        return None
    else:
        return TP/(TP + FP)

def Recall(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    confusion = ConfusionMatrix(y,yPredicted)
    TP = confusion[1][1]
    FN = confusion[1][0]
    
    #Avoid division by 0
    if TP == 0 and FN == 0:
        return None
    else:
        return TP/(TP + FN)

def FalseNegativeRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    confusion = ConfusionMatrix(y,yPredicted)
    TP = confusion[1][1]
    FN = confusion[1][0]

    #Avoid division by 0
    if TP == 0 and FN == 0:
        return None
    else:
        return FN/(TP + FN)

def FalsePositiveRate(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    confusion = ConfusionMatrix(y,yPredicted)
    TN = confusion[0][0]
    FP = confusion[0][1]

    #Avoid division by 0
    if TN == 0 and FP == 0:
        return None
    else:
        return FP/(TN + FP)

def ConfusionMatrix(y, yPredicted):
    # This function should return: [[<# True Negatives>, <# False Positives>], [<# False Negatives>, <# True Positives>]]
    #  Hint: writing this function first might make the others easier...
    __CheckEvaluationInput(y, yPredicted)
    
    #Start counters for confusion matrix
    TN = 0
    FP = 0
    FN = 0
    TP = 0

    #For each element classify confusion
    for i in range(len(yPredicted)):
        if y[i] == 0:
            if yPredicted[i] == 0:
                TN += 1
            else:
                FP += 1
        else:
            if yPredicted[i] == 0:
                FN += 1
            else:
                TP += 1

    #Return the Matrix
    return [TN,FP],[FN,TP]
    
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
    pointsToEvaluate = 100
    thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []

    try:
        for set_threshold in thresholds:
            FPRs.append(FalsePositiveRate(yValidate, model.predict(xValidate, threshold=set_threshold)))
            FNRs.append(FalseNegativeRate(yValidate, model.predict(xValidate, threshold=set_threshold)))
    except NotImplementedError:
        raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

    return (FPRs, FNRs, thresholds)


def TabulateModelPerformanceForROCFromProbabilityEstimates(yValidate, probability_estimates):
    pointsToEvaluate = 100
    thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
    FPRs = []
    FNRs = []

    for set_threshold in tqdm(thresholds):
        predictions = (probability_estimates >= set_threshold).float()
        FPRs.append(FalsePositiveRate(yValidate, predictions))
        FNRs.append(FalseNegativeRate(yValidate, predictions))

    return (FPRs, FNRs, thresholds)
