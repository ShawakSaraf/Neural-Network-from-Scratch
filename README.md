<h1 align=center>Building a Neural Network from Scratch</h1>

<p align="center">
	<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/></a>
	<a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white"/></a>
	<a href="https://colab.research.google.com/drive/1B9Uu9A-O6efN8_oqYGYmU8iiD-JK80dt"><img src="https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=for-the-badge&logo=Google-Colab&logoColor=white"/></a>
	
</p>

You know the rite of passage everyone has to go through when they first set foot into the world of programming.  
The iconic, print("Hello World!").

This is that but for the world of Neural Networks.

Let's first go through what's inside the repository.    
Dependencies - Python 3 or above, NumPy, Json, Pickle, Gzip, and Tqdm.

• Model.py and Optimizer.py consist of the model architecture and the training algorithm.  
• Run.py is where you can train the model and do inference.

Everything else is just trained model variables, and a script to load training data from the mnist.pkl.gz file.

This neural network is competent in tasks such as image classification (as shown below), and can serve as a good starting point for more advanced deep learning projects.
 
https://user-images.githubusercontent.com/74816223/206852868-7cf65399-7504-4640-910c-1b9d97597dab.mp4

I used the famous MNIST dataset for training, which consists of 70,000 images of handwritten digits (0-9) and their corresponding labels.

By implementing a neural network using only the Numpy library, I was able to understand the core principles and mathematics of how a neural network works without the complexities of modern machine learning frameworks. Numpy is a powerful library for numerical computing in Python, but it does not provide high-level abstractions for building and training neural networks. As a result, I had a lot of fun implementing the core components of neural networks from scratch.

This small project came into existence when I first started doing research about neural networks and found the absolutely incredible book, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen, and the amazing YouTube series, ["Neural Networks"](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3blue1brown. They did such an amazing job of explaining the magic behind these networks in easy-to-follow language. And most importantly in the book, Michael Nielsen provided the code, so you don't get stuck. I used it as a reference to do my own implementation of neural networks.

I've also written a blog, ["Understanding Neural Networks"](https://lookingisnotenough.com/UnderstandingNeuralNetworks), where I try to build an intuitive understanding of the inner workings of neural networks without getting bogged down into technical details.  
Have a read if this interests you.  
Another way to play with the code is the ["Google Colab Notebook"](https://colab.research.google.com/drive/1B9Uu9A-O6efN8_oqYGYmU8iiD-JK80dt) I've made.

# Usage
1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Run the script Run.py to train and test the model.
## Data
The MNIST dataset is provided in mnist.pkl.gz file and loaded using the mnist_loader module. The dataset is split into a training set and a validation set, which are loaded as follows:

```python
train_data, val_data = mnist_loader.load_data_wrapper()
````
## Model
The neural network is implemented using the MLP class. It is initialized with the following parameters:
```
layers        : a list of integers representing the number of neurons in each layer.
optimizer     : the optimizer to use for training, such as SGD or Adam.
loss          : the loss function to use, such as CrossEntropyLoss or MeanSquaredError.
activation    : the activation function to use, such as Sigmoid or ReLU.
learning_rate : the learning rate for the optimizer.
lmbda         : the regularization parameter.
```
For example:

```python
model = MLP( 
	layers        = [ 784, 30, 10 ],
	optimizer     = SGD,
	loss          = CrossEntropyLoss,
	activation    = Sigmoid,
	learning_rate = 0.1,
	lmbda         = 5.0,
)
```
## Training
To train the model, use the fit method with the following parameters:

```
train_data    : the training data in the form of a list of tuples (x, y), where x is the input and y is the label.
val_data      : the validation data in the same format as train_data.
epochs        : the number of epochs to train for.
batch_size    : the size of the mini-batches to use for training.
continue_train: a boolean indicating whether to continue training from a previously saved model.
monitor       : a Monitor object that tracks the progress of training and validation.
```
For example:

```python
monitor = Monitor(
	training   = True,
	validation = True,
	save       = False,
)

model.fit(
	train_data     = train_data[:10000],
	val_data       = val_data[:1000],
	epochs         = 5,
	batch_size     = 32,
	continue_train = False,
	monitor        = monitor
)
```
## Evaluation
To evaluate the model, use the history method of the Monitor object to retrieve the cost and accuracy for both training and validation.

```python
evaluation_cost, evaluation_accuracy, \
training_cost, training_accuracy = monitor.history()
```
## Prediction
To make a prediction for a single input x, use the predict method after training the model or load the pre-trained model.
```python
model = mlp.load_model()
print( "Prediction: ", model.predict( val_data[140][0] ) )
print( "Ground Truth: ", val_data[140][1]  
# Here I'm using validation data for prediction.  
```
