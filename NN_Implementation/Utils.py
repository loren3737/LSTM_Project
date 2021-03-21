import numpy as np
import torch

def createInputBatch(words, word2vec_model, num_features):
    """
    Creates an input batch (of size 1) from the specified list of words
    """
    
    # batch size of 1
    result = np.zeros((1, len(words), num_features), dtype="float32")
    
    for i, word in enumerate(words):
        # leave OOV words as zeroes
        if word in word2vec_model:
            result[0][i] = word2vec_model[word]

    return torch.from_numpy(result)
