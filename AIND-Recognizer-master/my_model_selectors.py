import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    The parameters of a hidden Markov model are of two types, transition probabilities and emission probabilities
    (also known as output probabilities). The transition probabilities control the way the hidden state at time t is
    chosen given the hidden state at time t−1.
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        scores = []

        for n in range (self.min_n_components, self.max_n_components + 1):

            try:
                model = self.base_model(n)
                log_likely = model.score(self.X, self.lengths)
            except Exception as e :
                #print(e)
                continue

            no_data_points = np.sum(self.lengths)
            no_transition = n * (n - 1)
            no_free = (2 * n * no_data_points)
            no_initial = n - 1

            no_params = no_transition + no_free + no_initial;

            scores.append((-2 * log_likely) + (no_params * np.log(n)))

        try:
            max_index = np.argmin(scores)  # find index of max log likeyhood in array
        except:
            return None

        states = max_index + 1 + self.min_n_components  # calculate the actual number of states
        return self.base_model(states)






class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = []
        dic_scores = []
        this_log_likely = None

        #get other words
        other_words = [word for word in filter(lambda word: word != self.this_word, self.words)]

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                this_log_likely = model.score(self.X, self.lengths)
                models.append((model, this_log_likely))

            except:
                continue

            other_log_likelies = []
            for word in other_words:
                try:
                    log_likely = model.score(self.hwords[word][0], self.hwords[word][1])
                except:
                    continue

                other_log_likelies.append(log_likely)

            dic = log_likely - np.mean(other_log_likelies)
            dic_scores.append(dic)
        try:
            i = np.argmax(dic_scores)
        except:
            return None
        states = i + self.min_n_components + 1
        return self.base_model(states)




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

        More complex model are suspected of having higher logarithmic likely hoods, so therefore it is insuficient to
        just use the state with the highest log likey hood to find the optimal number of states for a word. Instead
        we used Cross Validation (CV). Cross validation
        is a process used to find best number of states by using different combinations of data to train and test
        your hmm and then finds the best number of states by finding the max of the average of the log likeyly hood
        produced for the different combination of training and test data.

        To be specific the process is as follows:
            1. split
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        sequence_length = len(self.sequences)
        sequence_folder = KFold(n_splits = 3)
        scores = []

        #iterate over the number of states between your min and max
        for no_states in range(self.min_n_components, self.max_n_components + 1):
            if (sequence_length < 3): # not enough data to apply scikit learns KFold().split()
                try: # not all modela can be trained or provide a valid log likely hood
                    model = self.base_model(no_states)
                    log_likely = model.score(self.X, self.lengths)
                except :
                    continue
                scores.append(log_likely)
            else:
                log_likely_list = []
                for train_i, test_i in sequence_folder.split(self.sequences): # get 3 different combinations of train and test data
                    self.X, self.lengths = combine_sequences(train_i, self.sequences)
                    comb_test_X, comb_test_lengths = combine_sequences(test_i, self.sequences)
                    try:# not all modela can be trained or provide a valid log likely hood
                        model = self.base_model(no_states)
                        log_likely = model.score(comb_test_X, comb_test_lengths)
                    except :
                        continue
                    log_likely_list.append(log_likely)

                log_likely_avg = np.mean(log_likely_list) # find average of log likely hoods produced by all combinations
                scores.append(log_likely_avg)
        if len(scores) == 0 : return None
        max_index = np.argmax(scores) # find index of max log likeyhood in array
        states = max_index + 1 + self.min_n_components # calculate the actual number of states

        return self.base_model(states)