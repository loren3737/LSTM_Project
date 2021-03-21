from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import nltk.data
import pandas as pd

# Set values for various parameters. We may wanna tune these
MIN_WORD_COUNT = 40   # Minimum word count
NUM_WORKERS = 4       # Number of threads to run in parallel
DOWNSAMPLING = 1e-3   # Downsample setting for frequent words

def generate_word2vec(model_name, dataset_list, NUM_FEATURES, CONTEXT):

    sentences = [] 

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for i, dataset in enumerate(dataset_list):
        print("Parsing sentences from dataset " + str(i + 1))
        for review in dataset["review"]:
            sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    print ("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=NUM_WORKERS, \
                size=NUM_FEATURES, min_count = MIN_WORD_COUNT, \
                window = CONTEXT, sample = DOWNSAMPLING, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # You can load the model later using Word2Vec.load()
    model.save(model_name)

    return model