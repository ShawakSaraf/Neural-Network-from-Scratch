import mnist_loader
import MultiLayerPerceptron as mlp
from MultiLayerPerceptron import MLP, Monitor
from Optimizer import SGD, CrossEntropyCost, Sigmoid
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

monitor = Monitor(
		val_cost       = True,
		val_accuracy   = True,
		train_cost     = True,
		train_accuracy = True
	)

if 1:
	model.fit(
		train_data     = train_data[:1000],
		val_data       = val_data,
		epochs         = 5,
		batch_size     = 32,
		save_model     = False,
		continue_train = True,
		monitor        = monitor
	)
	evaluation_cost, evaluation_accuracy, \
	training_cost, training_accuracy = monitor.history()

print( "Prediction: ", mlp.predict( test_data[19][0] ) )
print( "Ground Truth: ", test_data[19][1] )
