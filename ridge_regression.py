import perceptron
import numpy as np
import copy

class Matchmaker(object):
    def __init__(self, gamma=0.001):
        self._gamma = gamma

    def new_client(self, examples):
        """
        The Matchmaker works with one client at a time. Pass an initial list
        of examples so that the matchmaker can begin its search.
        """

        dims = len(examples[0][0])
        self._examples = copy.deepcopy(examples)
        # Create observation matrix.
        self._A = np.array([e[0] for e in examples])
        self._b = np.array([np.array([e[1]]) for e in examples])
        # Add examples that weights sum to zero.
        for mul in np.arange(-1.0, 1.1, 0.25):
            e = np.ones(dims) * mul
            self._A = np.vstack([self._A, e])
            self._b = np.vstack([self._b, np.array([0.])])
            self._examples.append(e)
        # Create regularization matrix.
        self._Gamma = np.identity(self._A.shape[1]) * self._gamma
        # Solve for w.
        M = np.dot(self._A.T, self._A) + np.dot(self._Gamma.T, self._Gamma)
        self._w_estimate = np.dot(np.linalg.inv(M),
                                  np.dot(self._A.T, self._b)).reshape((dims,))
  
    def find_date(self):
        """ Create a date.  """

        dims = len(self.w_estimate)
        w_est_date = np.asarray(self._w_estimate > 0.0,
                                dtype=int).reshape((dims,))
        mk_date = lambda: w_est_date
        # Guarantee uniqueness.
        dates = [e[0] for e in self.examples]
        date_not_unique = lambda date: any((date == d).all() for d in dates)
        while True:
            # Get a date.
            date = mk_date()
            # Check same as any previous
            if date_not_unique(date):
                noise = np.random.normal(size=dims)
                mutate = np.asarray(np.abs(noise) > 2., dtype=int)
                date = (date + mutate) % 2
                if not date_not_unique(date):
                    return date
            else:
                return date

    def rate_example(self, d, r, iterations=400):
        """ Rate example d with the actual Dater preference r. """

        self._examples.append((d, r))
        num_example = len(self.examples)
        self._A = np.vstack([self._A, d])
        self._b = np.vstack([self._b, np.array([r])])
        # Solve for w.
        M = np.dot(self._A.T, self._A) + np.dot(self._Gamma.T, self._Gamma)
        self._w_estimate = np.dot(np.linalg.inv(M), np.dot(self._A.T, self._b))

    def get_examples(self):
        return self._examples
    def get_w_estimate(self):
        return self._w_estimate
    examples = property(get_examples, doc="Dates retured with rate_example().")
    w_estimate = property(get_w_estimate, doc="Currently estimated weights.")


