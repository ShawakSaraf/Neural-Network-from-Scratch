import mnist_loader
import Model as mlp
from Model import MLP, Monitor
from Optimizer import SGD, CrossEntropyLoss, Sigmoid
import numpy as np

"""
This script imports several classes, including MLP, SGD, CrossEntropyLoss, and Sigmoid, 
classes that are used to define a neural network. 
It then creates an instance of the MLP class with specified parameters, and creates an instance of 
the Monitor class with specified parameters to update certain matrices while the model is being trained. 
The MLP model is then trained using the fit method with specified parameters, 
and then the trained model is loaded using mlp.load_model(). 
Finally, the loaded model is used to make a prediction on a specific input and the ground truth value 
for that input is printed to the console.
"""

train_data, val_data = mnist_loader.load_data_wrapper()

model = MLP( 
	layers        = [ 784, 30, 10 ],
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
n_preds = 10
np.random.shuffle(val_data)

print( f'{"Prediction":15s} {"Ground Truth"}' )
for i in range(n_preds):
	print( f'{"":5s}{ str(model.predict( val_data[i][0] )):16s} {val_data[i][1]}' )
