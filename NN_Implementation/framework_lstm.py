from gensim.models import Word2Vec
import KaggleVectorize
from nn_lstm import LSTM
import nn_tools
import pandas as pd
import torch
from tqdm import tqdm
import Utils

### CONFIGURATION ###################
WORD2VEC_NAME = "../../word2vecModels/300features_40minwords_10context"
NUM_FEATURES = 300
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = False
LEARNING_RATE = 1
#####################################


def evaluate(lstm, word2vec_model, clean_reviews, sentiments, classification_threshold=0.5):
    lstm.eval()
    with torch.no_grad():
        sentiments_predicted = [0 if lstm(Utils.createInputBatch(clean_review, word2vec_model, NUM_FEATURES)).item() < classification_threshold else 1 for clean_review in tqdm(clean_reviews)]
    return nn_tools.Accuracy(sentiments, sentiments_predicted)


def trainOneEpoch(lstm, word2vec_model, clean_reviews, sentiments, optimizer, loss_function):
    total_loss = 0
    lstm.train()
    for i in tqdm(range(len(sentiments))):
        sentiment_predicted = lstm(Utils.createInputBatch(clean_reviews[i], word2vec_model, NUM_FEATURES))
        loss = loss_function(sentiment_predicted, torch.Tensor([[sentiments[i]]]))

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

        total_loss += loss.item()

    print ("Average loss:", total_loss / len(sentiments))


if __name__ == '__main__':
    # Step 1 Load in data
    print("LOADING data from CSV")
    dataset_A = pd.read_csv( "../../dataset/processed/A.tsv", header=0, delimiter="\t", quoting=3 )
    dataset_B = pd.read_csv( "../../dataset/processed/B.tsv", header=0, delimiter="\t", quoting=3 )

    # Step 2 Load Word2Vec
    print("LOADING Word2Vec Model")
    word2vec_model = Word2Vec.load(WORD2VEC_NAME)

    # Step 3 Clean reviews
    print ("CLEANING reviews")
    # stopwords may be useful since the LSTM processes one word at a time
    clean_reviews_A = KaggleVectorize.getCleanReviews(dataset_A, remove_stopwords=False, useSmall=True)
    clean_reviews_B = KaggleVectorize.getCleanReviews(dataset_B, remove_stopwords=False)

    #Step 4 Training LSTM model
    print("TRAINING LSTM model")
    lstm = LSTM(NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL)
    optimizer = torch.optim.SGD(lstm.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.BCELoss()

    epoch = 0
    print ("Epoch:", epoch)
    print ("Dev set accuracy:", evaluate(lstm, word2vec_model, clean_reviews_B, dataset_B["sentiment"]))
    # TODO automate convergence
    user_input = str(input("Continue training? y/n: "))
    while user_input.lower() == "y":
        trainOneEpoch(lstm, word2vec_model, clean_reviews_A, dataset_A["sentiment"][:200], optimizer, loss_function)
        epoch += 1

        print ("Epoch:", epoch)
        print ("Dev set accuracy:", evaluate(lstm, word2vec_model, clean_reviews_B, dataset_B["sentiment"]))
        if epoch % 5 == 0:
            user_input = str(input("Continue training? y/n"))
