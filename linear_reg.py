import perceptron
import numpy as np
import math
import copy
import pdb

"""
Notes:
"""

def generate_weights(weight_gen, partition=None, dims=100, precision=2):
    """
    Generate random weights in [-1, 1]. The positive weights are grouped at
    the head of the vector followed by the negative weights.
    """

    max_precision = math.pow(10, precision)

    def gen_weights_with_precision(num_weights):
        """ Generate weights with the requested precision. """

        weights = np.array([np.abs(weight_gen()) for i in range(num_weights)])
        weight_norm = weights / np.sum(weights)
        weight_norm_trunc = np.asarray(weight_norm * max_precision, dtype=int)
        sum_except_last = np.sum(weight_norm_trunc[:-1])
        weight_norm_trunc[-1] = max_precision - sum_except_last
        return weight_norm_trunc

    # Partition the weights.
    if partition is None:
        num_pos = min(1 + int(np.random.uniform() * dims), dims - 1)
        num_neg = (dims - num_pos)
        partition = (num_pos, num_neg)
    else:
        assert(len(partition) is 2)
        dims = sum(partition)
    # Generate weights.
    pos_weights = gen_weights_with_precision(partition[0])
    neg_weights = -1 * gen_weights_with_precision(partition[1])
    # Combine weights.
    weights = np.hstack([pos_weights, neg_weights])
    return weights / max_precision
 
class MatchmakerLinReg(object):
    """ The matchmaker tries to infer the preferences of the dater. """

    def __init__(self, dims=100):
        self._examples = []
        self._w_estimate = np.zeros(dims)
        self._alpha = 0.005

    def new_client(self, examples):
        """
        The Matchmaker works with one client at a time. Pass an initial list
        of examples so that the matchmaker can begin its search.

        """
        dims = len(examples[0][0])
        self._w_estimate = np.zeros(dims)
        self._examples = copy.copy(examples)
        # Contraint that weights sum to zero.
        for mul in np.arange(-1.0, 1.1, 0.25):
            self.rate_example(np.ones(dims) * mul, 0.)


    def find_date(self, method=None, **kwargs):
        """ Create a date. """
        pdb.set_trace()

    def get_examples(self):
        return self._examples
    def get_w_estimate(self):
        return self._w_estimate
    examples = property(get_examples, doc="Dates retured with rate_example().")
    w_estimate = property(get_w_estimate, doc="Currently estimated weights.")

def iterate_matches_lin_reg(matchmaker, dater, iterations):
    """ Show the dater some dates and track scores. """

    scores = []
    for date in range(iterations):
        d = matchmaker.find_date()
        r = dater.rate_a_date(d)
        matchmaker.rate_example(d, r)
        scores.append(dater.rate_a_date(matchmaker.find_date()))
    return scores

