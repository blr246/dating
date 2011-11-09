import numpy as np
import math
import copy

"""
Notes:
20111110 -- Tried with different weights:
  * Single positive and negative weight shuffled: the perceptron finds the
    max easily. Not a good strategy for the Dater.
  * All weight distributed evenly: tends to perform better than random
    weights. This is not a good strategy for the Dater.
  * Tried weights using Beta distribution and params
      (.5, .5), (2., 5.), (2., 2.)
    but nothing seems to perform better than Gauss. It seems the same, which
    I suspect has to do with the very small precision offered by 100 dims and
    2 decimal places of precision.
  * Tried weights for Gamma with k = 2, theta = 2. Same results.
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
 
class Dater(object):
    """ The dater will rate the candidate dates. """

    # Dimension of preference vector.
    DIMS = 100
    # Number of candidates that the dater will rate a priori.
    INITIAL_CANDIDATES = 20

    @classmethod
    def random_binary_candidate(cls):
        """ Generate a random binary candidate vector. """

        ones = int(max(np.random.uniform() * cls.DIMS, 1))
        zeros = cls.DIMS - ones;
        d = np.asarray(np.hstack([np.ones(ones), np.zeros(zeros)]), dtype=int)
        return d, ones, zeros

    def __init__(self, w=None):
        gauss_weights = lambda: np.random.normal(loc=.1, scale=.1)
        self._w = generate_weights(gauss_weights, dims=Dater.DIMS) \
                  if w is None else w
        np.random.shuffle(self._w)
        # Generate initial candidates.
        self._initial_survey = []
        for candidate_idx in range(Dater.INITIAL_CANDIDATES):
            # Don't make an ideal candidate or a dup.
            while True:
                d, ones, zeros = self.random_binary_candidate()
                np.random.shuffle(d)
                r = self.rate_a_date(d)
                candidates = [e[0] for e in self._initial_survey]
                if all((d != c).any() for c in candidates) and r != 1:
                    break
            self._initial_survey.append((d, r))

    def rate_a_date(self, d):
        """ Date is a DIMS length vector. """

        return int((np.dot(self.w, d) * Dater.DIMS) + 0.5) / float(Dater.DIMS)

    def get_w(self):
        return self._w
    def get_initial_survey(self):
        return self._initial_survey
    w = property(get_w, doc="The dater's preference vector.")
    initial_survey = property(get_initial_survey,
                              doc="Initial list of scored candidates.")


class Matchmaker(object):
    """ The matchmaker tries to infer the preferences of the dater. """

    def __init__(self):
        self._examples = []
        self._w_estimate = np.zeros(Dater.DIMS)
        self._alpha = 0.01

    def new_client(self, examples):
        """
        The Matchmaker works with one client at a time. Pass an initial list
        of examples so that the matchmaker can begin its search.

        """

        #self._w_estimate = generate_weights(np.random.uniform, dims=Dater.DIMS)
        #np.random.shuffle(self._w_estimate)
        self._w_estimate = np.zeros(Dater.DIMS)
        self._examples = copy.copy(examples)
        # Contraint that weights sum to zero.
        for mul in np.arange(-1.0, 1.1, 0.25):
            self.rate_example(np.ones(Dater.DIMS) * mul, 0.)

    def find_date(self):
        """ Create a date. """

        max_date = np.asarray(self._w_estimate >= 0.005, dtype=int)
        # Check same as any previous
        dates = [e[0] for e in self.examples]
        if any((max_date == d).all() for d in dates):
            print "date was same so mutating"
            noise = np.random.normal(size=Dater.DIMS)
            mutate = np.asarray(np.abs(noise) > 1.0, dtype=int)
            max_date = (max_date + mutate) % 2
        return max_date

    def rate_example(self, d, r, iterations=100):
        """ Rate example d with the actual Dater preference r. """

        self._examples.append((d, r))
        permutation = range(len(self.examples))
        for iteration in range(iterations):
            np.random.shuffle(permutation)
            # Iterate all examples.
            for date, rate in self.examples:
                pred = np.dot(self.w_estimate, date)
                error = rate - pred
                w_new = self._alpha * error * date
                self._w_estimate = self._w_estimate + w_new

    def get_examples(self):
        return self._examples
    def get_w_estimate(self):
        return self._w_estimate
    examples = property(get_examples, doc="Dates retured with rate_example().")
    w_estimate = property(get_w_estimate, doc="Currently estimated weights.")


def iterate_matches(matchmaker, dater, iterations):
    """ Show the dater some dates. """

    for date in range(iterations):
        print "date number", date
        d = matchmaker.find_date()
        r = dater.rate_a_date(d)
        matchmaker.rate_example(d, r)
    scores = [e[1] for e in matchmaker.examples]
    return scores

