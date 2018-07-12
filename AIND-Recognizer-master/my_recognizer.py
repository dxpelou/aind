import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # for index, word_X_length in test_set.get_all_Xlengths().items():

    for word_id in range(0, len(test_set.get_all_Xlengths())):
        word_log_likely = {}
        current_word_feature_lists_sequences, current_sequences_length = test_set.get_item_Xlengths(word_id)

        for word, model in models.items():
            try:
                # score = model.score(word_X_length[0], word_X_length[1])
                score = model.score(current_word_feature_lists_sequences, current_sequences_length)
                word_log_likely[word] = score
            except :
                pass

        probabilities.append(word_log_likely)

        # max = np.max(word_log_likely)
        max2 = None
        try:
            max2 = max(word_log_likely, key=word_log_likely.get)
        except ValueError:
            print("None")
            return ([],[])
        guesses.append(max2)
    return probabilities, guesses
