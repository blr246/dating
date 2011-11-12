import perceptron
import numpy as np
import socket
import sys

def __main__(argv):
    """ Matchmaker's main method. Connect to server and guess dates. """

    # Collect positional parameters.
    if len(argv) is not 4:
        print "Usage: {0} HOST PORT DIMENSIONS".format(argv[0])
        return -1
    else:
        hostname = argv[1]
        port = int(argv[2])
        dims = int(argv[3])

    # Connect. Let exceptions happen so that we can see trace.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))

    # Read examples from the server.
    str_examples = s.recv(20480)
    print str_examples
    examples_lines = str_examples.split('\n')[1:-2]
    examples = [perceptron.Serialization.scored_date_from_string(l)
                for l in examples_lines]

    # Create matchmaker and introduce to the dater.
    print "Introducing dater to matchmaker with {0} initial candidates."\
          .format(len(examples))
    M = perceptron.Matchmaker()
    M.new_client(examples)

    # Create dates until limit reached or perfect match found.
    while True:
        d = M.find_date()
        str_d = perceptron.Serialization.date_to_string(d)
        s.send(str_d + '\n')
        str_rated = s.recv(4096)
        print str_rated
        if str_rated.lower().find("bye") >= 0:
            print "Server sent \"Bye\" message."
            break
        str_rated_lines = str_rated.split('\n')
        d_rate, s_rate = perceptron.Serialization.scored_date_from_string(
            str_rated_lines[0])
        d_check = np.asarray(d, dtype=float) == d_rate
        if not d_check.all():
            print "WARN : Got a different date from server."
            print d_check
            print d
            print d_rate
        M.rate_example(d_rate, s_rate)

    return 0

if __name__ == "__main__":
    __main__(sys.argv)

