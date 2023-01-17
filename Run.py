"""
Imports several classes, including MLP, SGD, CrossEntropyLoss, and Sigmoid, 
classes that are used to define a neural network.
"""
import mnist_loader
import Model as mlp
from Model import MLP, Monitor
from Optimizer import SGD, CrossEntropyLoss, Sigmoid
import numpy as np
import matplotlib.pyplot as plt

train_data, val_data = mnist_loader.load_data_wrapper() # Loading the MNIST data.

"""
Creating a Multi-Layer Perceptron (MLP) model with 784 input units, 30 hidden units, and 10 output units.
The optimizer used is Stochastic Gradient Descent (SGD), the loss function is Cross-Entropy Loss, and the activation function is Sigmoid.
The learning rate is set to 0.1 and the regularization parameter (lmbda) is set to 5.0.
"""
model = MLP( 
    layers        = [ 784, 30, 10 ],
    optimizer     = SGD,
    loss          = CrossEntropyLoss,
    activation    = Sigmoid,
    learning_rate = 0.1,
    lmbda         = 5.0,
)

# Monitors the relevant model metrics while training.
monitor = Monitor(
    training   = True,
    validation = True,
    save       = False,
)

"""
Train the model and store the training history in the evaluation_cost, 
evaluation_accuracy, training_cost, and training_accuracy lists.
"""
if 0:
    model.fit(
      train_data     = train_data[:10000],
      val_data       = val_data[:1000],
      epochs         = 5,
      batch_size     = 32,
      continue_train = True,
      monitor        = monitor
    )
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = monitor.history()

model = mlp.load_model() # Loading the previously saved model.

"""
Number of data use for prediction from the validation data, we've not show the 
validation data to the model while training so it's ok to use it for prediction.
"""
n_preds = 10 
np.random.shuffle(val_data) # Shuffling the data to get diffrent elements every time.

fig, axes = plt.subplots(1,n_preds, figsize=(10,2))

print( f'{"Prediction":15s} {"Ground Truth"}' )
for i,ax in enumerate(axes.flat):
	ax.imshow( np.reshape(val_data[i][0], (28,28)), cmap="Greys" )
	ax.get_yaxis().set_visible(False)

	prediciton = 'Pred: ' + str(model.predict( val_data[i][0] )) if i==0 else str(model.predict( val_data[i][0] ))
	ax.set_xlabel( prediciton, labelpad=18, size=15 )
	print( f'{"":5s}{ str(model.predict( val_data[i][0] )):16s} {val_data[i][1]}' )

plt.show()