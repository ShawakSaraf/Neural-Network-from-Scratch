import mnist_loader
import Multi_Layer_Perceptron as mlp
from Multi_Layer_Perceptron import MLP
import numpy as np
from pprint import pprint
import time

train_data, val_data, test_data = mnist_loader.load_data_wrapper()
model = MLP( layers=[784, 30, 10] )
	
if 1:
	evaluation_cost, evaluation_accuracy, \
	training_cost, training_accuracy = \
	model.SGD( 
		train_data,
		epochs                      = 50,
		batch_size                  = 32,
		learning_rate               = 0.1,
		lmbda                       = 5.0,
		evaluation_data             = val_data,
		save                        = False,
		continue_LastSaved          = False,
		monitor_evaluation_cost     = True,
		monitor_evaluation_accuracy = True,
		monitor_training_cost       = True,
		monitor_training_accuracy   = True
	)

# print( "Prediction: ", mlp.Predict( test_data[19][0] ) )
# print( "Ground Truth: ", test_data[19][1] )
print( mlp.model_accuracy() )