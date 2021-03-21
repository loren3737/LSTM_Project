from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from KaggleWord2VecUtility import KaggleWord2VecUtility
import logging
import nltk.data
import pandas as pd

# Set values for various parameters. We may wanna tune these
MIN_WORD_COUNT = 40   # Minimum word count
NUM_WORKERS = 4       # Number of threads to run in parallel

def generate_doc2vec(model_name, dataset_list, NUM_FEATURES=100, CONTEXT=5):

    print(model_name)

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the training sets into clean docs
    #
    # flatten each list of sentences from a review into a single list of words
    docs = []  # Initialize an empty list of docs

    for i, dataset in enumerate(dataset_list):
        print("Parsing sentences from dataset " + str(i + 1))
        for review in dataset["review"]:
            sentences = KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
            docs.append([word for sentence in sentences for word in sentence])

    # Initialize and train the model
    print ("Training Doc2Vec model...")
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(documents, workers=NUM_WORKERS, \
                vector_size=NUM_FEATURES, min_count = MIN_WORD_COUNT, \
                window = CONTEXT, seed=1)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # You can load the model later using Doc2Vec.load()
    print(model_name)
    model.save(model_name)

    return model
