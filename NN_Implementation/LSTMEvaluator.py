import collections
import json
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nn_lstm_attempt_1 import LSTM
import nn_tools
import pandas as pd
import torch
import torchtext
from tqdm import tqdm

MODEL_FILE_PATH = "../../nlpModels/64hidden_1layer_0dropout_bidirectional_Word2Vec.pt"
WORD2VEC_VECTORS_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_w2vformat.txt"
WORD2VEC_WORD_FREQ_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_word_freq.json"
BATCH_SIZE = 64
NUM_FEATURES = 300
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0
BIDIRECTIONAL = True

ERROR_ANALYSIS = False


def getAllProbabilityEstimates(lstm, iterator):
    result = torch.Tensor()
    lstm.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, text_lengths = batch.review

            predictions = lstm(text, text_lengths).squeeze()
            result = torch.cat((result, predictions))

    return result


if __name__ == '__main__':
    text = torchtext.data.Field(tokenize=KaggleWord2VecUtility.review_to_wordlist, batch_first=True, include_lengths=True)
    label = torchtext.data.LabelField(dtype=torch.float, batch_first=True)

    fields = [(None, None), ("sentiment", label), ("review", text)]

    # load datasets
    print("LOADING data from CSV")
    train_data = torchtext.data.TabularDataset(path="../../dataset/processed/A.tsv", format="tsv", fields=fields, skip_header=True)
    dev_data = torchtext.data.TabularDataset(path="../../dataset/processed/B.tsv", format="tsv", fields=fields, skip_header=True)
    dev_data_pd = pd.read_csv( "../../dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )
    test_data = torchtext.data.TabularDataset(path="../../dataset/processed/C.tsv", format="tsv", fields=fields, skip_header=True)

    print("LOADING Word2Vec Model")
    vectors = torchtext.vocab.Vectors(WORD2VEC_VECTORS_FILE_PATH)
    with open(WORD2VEC_WORD_FREQ_FILE_PATH) as word_freq_file:
        word_counter = collections.Counter(json.load(word_freq_file))
    vocab = torchtext.vocab.Vocab(word_counter, vectors=vectors)
    text.vocab = vocab

    label.build_vocab(train_data)

    # load iterator
    # don't sort for error analysis
    dev_iterator = torchtext.data.BucketIterator(dev_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.review), train=False, sort=not ERROR_ANALYSIS, sort_within_batch=not ERROR_ANALYSIS)
    test_iterator = torchtext.data.BucketIterator(test_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.review), train=False, sort_within_batch=True)

    true_labels = torch.Tensor()
    for batch in dev_iterator:
        true_labels = torch.cat((true_labels, batch.sentiment))

    true_labels_test = torch.Tensor()
    for batch in test_iterator:
        true_labels_test = torch.cat((true_labels_test, batch.sentiment))

    if ERROR_ANALYSIS:
        # verify that the reviews are in their original order, so that we can identify false negatives/positives by index
        true_labels_list = true_labels.int().tolist()
        sentiments_list = dev_data_pd["sentiment"].tolist()
        print ("True labels:", true_labels_list)
        print ("Sentiments:", sentiments_list)
        print ("Reviews are in original order:", true_labels_list == sentiments_list)

    # load the model
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, len(text.vocab), enforce_sorted=not ERROR_ANALYSIS)
    lstm.load_state_dict(torch.load(MODEL_FILE_PATH))

    # get predictions
    probability_estimates = getAllProbabilityEstimates(lstm, dev_iterator)
    probability_estimates_test = getAllProbabilityEstimates(lstm, test_iterator)

    rounded_predictions = torch.round(probability_estimates)

    if ERROR_ANALYSIS:
        print ("Confusion matrix:", nn_tools.ConfusionMatrix(true_labels, rounded_predictions))

        # flip for false positives
        with open("LSTMFalseNegatives.txt", mode="w") as file:
            print ("False negatives")
            for i in range(len(true_labels)):
                if true_labels[i] == 1 and rounded_predictions[i] == 0:
                    file.write(dev_data_pd["review"][i] + "\n\n")
    else:
        # accuracies
        print ("Dev set accuracy:", nn_tools.Accuracy(true_labels, rounded_predictions))
        if input("Display test set accuracy? y/n ").lower() == "y":
            print ("Test set accuracy:", nn_tools.Accuracy(true_labels_test, torch.round(probability_estimates_test)))

        # get ROC data
        FPRs, FNRs, thresholds = nn_tools.TabulateModelPerformanceForROCFromProbabilityEstimates(true_labels, probability_estimates)
        print ("FPRs:", FPRs)
        print ("FNRs:", FNRs)
        print ("Thresholds:", thresholds)
