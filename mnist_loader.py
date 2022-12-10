import pickle as cPickle
import numpy as np
import gzip, os
curr_dir = os.path.dirname( os.path.abspath( __file__ ) )

def load_data():
    with gzip.open(curr_dir+'/mnist.pkl.gz') as f:
        training_data, validation_data, test_data = cPickle.load(f, encoding = 'latin1')
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d  = load_data()
    
    training_inputs   = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results  = [vectorized_result(y) for y in tr_d[1]]
    training_data     = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data   = list(zip(validation_inputs, va_d[1]))

    test_inputs       = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data         = list(zip(test_inputs, te_d[1]))

    training_data, test_data = np.array(training_data, dtype=object), np.array(test_data, dtype=object)
    training_data = np.concatenate( [training_data, test_data], axis=0, dtype=object )
    return training_data, validation_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
