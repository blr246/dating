import perceptron
import sys

def __main__(argv):
    """ Dater's main method. Write a weights file and exit. """

    # Optional command line parameter for weights filename.
    if len(argv) < 2:
        print "Usage: {0} DIMS [FILENAME]".format(argv[0])
        return -1
    dims = eval(argv[1])
    filename = argv[2] if len(argv) > 2 else 'weights.txt'
    # Make a dater and write the weights.
    D = perceptron.Dater(dims=dims)
    weight_lines = perceptron.Serialization.dater_to_lines(D)
    weight_file = open(filename, 'w')
    weight_file.writelines(weight_lines)
    weight_file.close()
    return 0

if __name__ == "__main__":
    __main__(sys.argv)

