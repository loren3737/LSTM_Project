import collections
import json
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nn_lstm_attempt_1 import LSTM
import nn_tools
import torch
import torchtext
from tqdm import tqdm

### CONFIGURATION ###################
BATCH_SIZE = 64
NUM_FEATURES = 300
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0
BIDIRECTIONAL = False

USE_WORD2VEC = True
WORD2VEC_VECTORS_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_w2vformat.txt"
WORD2VEC_WORD_FREQ_FILE_PATH = "../../word2vecModels/300features_40minwords_10context_word_freq.json"
#####################################


def trainOneEpoch(lstm, iterator, optimizer, loss_function):
    total_loss = 0
    total_count = 0
    lstm.train()
    for batch in tqdm(iterator):
        text, text_lengths = batch.review

        predictions = lstm(text, text_lengths).squeeze()

        loss = loss_function(predictions, batch.sentiment)

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

        batch_size = text.size()[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    print ("Average train set loss:", total_loss / total_count)


def evaluate(lstm, iterator, loss_function, classification_threshold=0.5):
    total_accuracy = 0
    total_loss = 0
    total_count = 0
    lstm.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            text, text_lengths = batch.review

            predictions = lstm(text, text_lengths).squeeze()

            loss = loss_function(predictions, batch.sentiment)

            # TODO use classification threshold
            rounded_predictions = torch.round(predictions)

            accuracy = nn_tools.AccuracyFromTensors(batch.sentiment, rounded_predictions)

            batch_size = text.size()[0]
            total_accuracy += accuracy.item() * batch_size
            total_loss += loss.item() * batch_size
            total_count += batch_size

    print ("Average loss:", total_loss / total_count)
    
    return total_accuracy / total_count


if __name__ == '__main__':
    model_file_path = input("Model file path? ")

    # preprocess data
    text = torchtext.data.Field(tokenize=KaggleWord2VecUtility.review_to_wordlist, batch_first=True, include_lengths=True)
    label = torchtext.data.LabelField(dtype=torch.float, batch_first=True)

    labeled_fields = [(None, None), ("sentiment", label), ("review", text)]
    unlabeled_fields = [(None, None), ("review", text)]

    # load datasets
    print("LOADING data from CSV")
    train_data = torchtext.data.TabularDataset(path="../../dataset/processed/A.tsv", format="tsv", fields=labeled_fields, skip_header=True)
    dev_data = torchtext.data.TabularDataset(path="../../dataset/processed/B.tsv", format="tsv", fields=labeled_fields, skip_header=True)
    dataset_D = torchtext.data.TabularDataset(path="../../dataset/processed/D.tsv", format="tsv", fields=unlabeled_fields, skip_header=True)
    dataset_E = torchtext.data.TabularDataset(path="../../dataset/processed/E.tsv", format="tsv", fields=unlabeled_fields, skip_header=True)
    print ("Training example:", vars(train_data.examples[0]))


    # prep input and output sequences
    if USE_WORD2VEC:
        print("LOADING Word2Vec Model")
        
        vectors = torchtext.vocab.Vectors(WORD2VEC_VECTORS_FILE_PATH)

        with open(WORD2VEC_WORD_FREQ_FILE_PATH) as word_freq_file:
            word_counter = collections.Counter(json.load(word_freq_file))

        vocab = torchtext.vocab.Vocab(word_counter, vectors=vectors)

        text.vocab = vocab
    else:
        # use GloVe pretrained embeddings
        print ("LOADING GloVe pretrained embeddings")
        text.build_vocab(train_data, dataset_D, dataset_E, min_freq=40, vectors = "glove.6B.300d")
        
    label.build_vocab(train_data)

    print ("Size of text vocab:", len(text.vocab))
    print ("Size of label vocab:", len(label.vocab))
    print ("Commonly used words:", text.vocab.freqs.most_common(10))
    # print ("Word dictionary:", text.vocab.stoi)

    # load iterators
    train_iterator, dev_iterator = torchtext.data.BucketIterator.splits((train_data, dev_data), batch_size=BATCH_SIZE, sort_key=lambda x: len(x.review), sort_within_batch=True)


    print("TRAINING LSTM model")
    
    # instantiate model
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, len(text.vocab))

    # initialize pretrained embeddings from word2vec or GloVe
    lstm.embedding.weight.data.copy_(text.vocab.vectors)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(lstm.parameters())
    loss_function = torch.nn.BCELoss()

    epoch = 0
    best_dev_set_accuracy = 0
    dev_set_accuracy = evaluate(lstm, dev_iterator, loss_function)
    print ("Epoch:", epoch)
    print ("Dev set accuracy:", dev_set_accuracy)
    while epoch < 5 or dev_set_accuracy >= best_dev_set_accuracy:
        # save the best model
        if dev_set_accuracy >= best_dev_set_accuracy:
            best_dev_set_accuracy = dev_set_accuracy
            torch.save(lstm.state_dict(), model_file_path)

        trainOneEpoch(lstm, train_iterator, optimizer, loss_function)
        epoch += 1
        dev_set_accuracy = evaluate(lstm, dev_iterator, loss_function)
        print ("Epoch:", epoch)
        print ("Dev set accuracy:", dev_set_accuracy)


    print("EVALUATION")

    # reload the best model
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL, len(text.vocab))
    lstm.load_state_dict(torch.load(model_file_path))

    print ("Final train set accuracy:", evaluate(lstm, train_iterator, loss_function))
    print ("Final dev set accuracy:", evaluate(lstm, dev_iterator, loss_function))
