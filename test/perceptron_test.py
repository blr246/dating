import unittest
import perceptron
import numpy as np
import sys
sys.path.append('..')

class GenerateWeightsTest(unittest.TestCase):
    def runTest(self):
        """ Test that weights are generated according to the problem spec. """

        def parition_and_sum_weights(weights):
            """
            Partiion weights into positive and negative sets and return
            sum of the sets.
            """

            pos = np.array([w for w in weights if w >= 0])
            neg = np.array([w for w in weights if w <  0])
            return np.sum(pos), np.sum(neg)

        ITERATIONS = 1000

        # Random uniform partion and weights.
        for iteration in range(ITERATIONS):
            w = perceptron.generate_weights(np.random.uniform)
            sum_pos, sum_neg = parition_and_sum_weights(w)
            self.assertAlmostEqual( 1., sum_pos)
            self.assertAlmostEqual(-1., sum_neg)

        # Random uniform partion and gaussian weights.
        for iteration in range(ITERATIONS):
            w = perceptron.generate_weights(np.random.normal)
            sum_pos, sum_neg = parition_and_sum_weights(w)
            self.assertAlmostEqual( 1., sum_pos)
            self.assertAlmostEqual(-1., sum_neg)


if __name__ == '__main__':
    unittest.main()

