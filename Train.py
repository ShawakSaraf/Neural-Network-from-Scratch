from statistics import mode
from regex import F
import mnist_loader
import MultiLayerPerceptron as mlp
from MultiLayerPerceptron import MLP, Monitor
from Optimizer import SGD, CrossEntropyLoss, Sigmoid
import numpy as np

train_data, val_data, test_data = mnist_loader.load_data_wrapper()
train_data, test_data = np.array(train_data, dtype=object), np.array(test_data, dtype=object)
train_data = np.concatenate( [train_data, test_data], axis=0, dtype=object )

model = MLP( 
	layers        = [784, 30, 10],
	optimizer     = SGD,
	loss          = CrossEntropyLoss,
	activation    = Sigmoid,
	learning_rate = 0.1,
	lmbda         = 5.0,
)

monitor = Monitor(
		training   = True,
		validation = True,
		save       = False,
	)

if 1:
	model.fit(
		train_data     = train_data[:10000],
		val_data       = val_data[:1000],
		epochs         = 5,
		batch_size     = 32,
		continue_train = False,
		monitor        = monitor
	)
	evaluation_cost, evaluation_accuracy, \
	training_cost, training_accuracy = monitor.history()

model = mlp.load_model()
print( "Prediction: ", model.predict( val_data[140][0] ) )
print( "Ground Truth: ", val_data[140][1] )
