import mnist_loader
import Multi_Layer_Perceptron as mlp
from Multi_Layer_Perceptron import MLP, SGD, CrossEntropyCost, Sigmoid
import numpy as np

train_data, val_data, test_data = mnist_loader.load_data_wrapper()
model = MLP( 
	layers        = [784, 30, 10],
	optimizer     = SGD,
	cost          = CrossEntropyCost,
	activation    = Sigmoid,
	learning_rate = 0.1,
	lmbda         = 5.0,
)
	
if 1:
	evaluation_cost, evaluation_accuracy, \
	training_cost, training_accuracy = \
	model.fit(
		train_data,
		evaluation_data             = val_data,
		epochs                      = 50,
		batch_size                  = 32,
		save                        = False,
		continue_LastSaved          = True,
		monitor_evaluation_cost     = True,
		monitor_evaluation_accuracy = True,
		monitor_training_cost       = True,
		monitor_training_accuracy   = True
	)

print( mlp.model_accuracy() )
print( "Prediction: ", mlp.predict( test_data[19][0] ) )
print( "Ground Truth: ", test_data[19][1] )
