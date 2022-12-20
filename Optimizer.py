import numpy as np

class SGD():
	"""
	SGD (short for Stochastic Gradient Descent), represents an optimization algorithm 
	that can be used to train a neural network model.
	The __call__ method of the SGD class performs a single iteration of training, 
	which includes iterating over the training data in mini-batches, 
	making predictions using the forward propagation method, and updating the model's weights and biases 
	using the back-propagation method. 
	The forward_propagation and back_propagation methods are helper methods that implement the 
	forward and backward passes of the training algorithm, respectively. 
	The SGD class is initialized with a reference to the MLP model that it will be optimizing.
	"""
	def __init__( self, model = None ):
		self.model         = model
		self.learning_rate = model.learning_rate
		self.lmbda         = model.lmbda
		self.loss          = model.loss
		self.activation    = model.activation
		self.num_layers    = model.num_layers

	def __call__( self, batch, num_data ):
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		"""
		This loop calculates the gradient of loss with respect to all variables, 
		sums them up and stores in nabla_Wm nabla_B lists,
		"""
		for inputs, labels in batch:
			a  = inputs 	# Activation
			As = [inputs] 	# list to store all the Activation matrix, layer by layer
			Zs = []			# list to store all the z vectors, layer by layer

			self.forward_propagation( a, As, Zs )
			delta_Nabla_W, delta_Nabla_B = self.back_propagation(As, Zs, labels)

			# Summing up all the changes in weights amd biases
			nabla_W = [ nw + d_nw for nw, d_nw in zip( nabla_W, delta_Nabla_W ) ]
			nabla_B = [ nb + d_nb for nb, d_nb in zip( nabla_B, delta_Nabla_B ) ]

		# Taking steps towards the lowest loss function
		# lmbda, regularization parameter
		regularization  = 1 - ( self.learning_rate * ( self.lmbda / num_data ) )
		nw_coef = self.learning_rate / len(batch)

		self.model.weights = [
			( regularization * w ) - ( nw_coef * nw )
			for w, nw in zip( self.model.weights, nabla_W )
		]

		self.model.biases = [ 
			b - ( self.learning_rate / len(batch) ) * nb
			for b, nb in zip( self.model.biases, nabla_B ) 
		]
	
	def forward_propagation( self, a, As=None, Zs=None  ):
		"""
		This code defines a forward_propagation algorithm. 
		It is called during the training of out neural network.

		The method takes three arguments:

		a  : a matrix of input activations
		As : a list to store the activation matrices, layer by layer
		Zs : a list to store the z vectors, layer by layer

		The method uses the weights and biases of the model to perform a 
		forward propagation through the network. 
		This means that the input activations "a" are passed through the network layer by layer, 
		and the output activations are returned by the method.

		In addition to returning the output activations, 
		the method also appends the intermediate activation matrices 
		and z vectors to the As and Zs lists if they are provided. 
		These lists will be used later in the back-propagation step of training.
		"""
		for W, b in zip( self.model.weights, self.model.biases ):
			z = np.dot( W, a ) + b
			if Zs != None : Zs.append(z)

			a = self.activation(z)
			if As != None : As.append(a)
		return a
	
	def back_propagation( self, As, Zs, labels ):
		"""
		This function is implementing back-propagation, a method for training a neural network. 
		The back-propagation algorithm is used to calculate the gradient of the loss function with respect to 
		the weights and biases of the network. This is used to update the weights and biases to reduce the loss 
		and improve the performance of the network on a given task.

		It takes three arguments: 
		As		 : a list to store the activation matrices, layer by layer
		Zs		 : a list to store the z vectors, layer by layer , where z = (a * w) + b
		labels : a numpy array of target values for the network.

		First the method initializes two lists, nabla_W and nabla_B, to store the gradients of the loss function 
		with respect to the weights and biases of the network, respectively. 
		These are initialized to be arrays of zeros with the same shape as the weight and bias arrays of the network.

		Next, it calculates the gradient of the loss with respect to the output of the network, 
		and stores the result in delta_error. It then uses this to calculate the gradient of the loss with respect to 
		the final layer's weights and biases, and stores these in nabla_W and nabla_B, respectively.

		It then loops through the layers of the network in reverse order (from the second-to-last layer to the first layer), 
		and uses the previously calculated gradients to compute the gradients of the loss with respect to the weights and biases 
		of each layer.

		Finally, the method returns the calculated gradients in nabla_W and nabla_B.
		"""
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		delta_error = self.loss.delta( As[-1], labels )
		nabla_W[-1] = np.dot( delta_error, As[-2].transpose() )
		nabla_B[-1] = delta_error

		# Backwards pass / Back-propagation
		for l in range(2, self.num_layers):
			delta_error = np.dot( self.model.weights[-l + 1].transpose(), delta_error ) * self.activation.prime( Zs[-l] )
			nabla_W[-l] = np.dot( delta_error, As[-l - 1].transpose() )
			nabla_B[-l] = delta_error

		return nabla_W, nabla_B

class CrossEntropyLoss():
	"""
	CrossEntropyLoss implements the cross-entropy loss function for a binary classification task.
	It measures the difference between the predicted probability distribution over classes and 
	the true probability distribution over classes.
	"""
	@staticmethod
	def func(a, y):
		"""
		The func method takes two arguments, a and y, and calculates the cross-entropy loss for a given prediction a 
		and true label y. This is done by summing the cross-entropy loss for each example in the batch of data.
		"""
		return np.sum( np.nan_to_num( -y * np.log(a) - (1-y) * np.log(1-a) ) )

	@staticmethod
	def delta(a, y):
		"""
		The delta method calculates the gradient of the cross-entropy loss with respect to the output of the network. 
		It simply returns the difference between the predicted probabilities (a) and the true labels (y). 
		This is used in the back-propagation algorithm to calculate the gradient of the loss with respect to the 
		weights and biases of the network.
		"""
		return ( a - y )
	
class Sigmoid():
	"""
	The sigmoid function is often used as the activation function for neurons in a neural network. 
	It maps a real-valued input to a value between 0 and 1, which can be interpreted as a probability.
	"""
	def __call__( self, z ):
		"""
		This method is called when an instance of the class is called as a function, 
		and it calculates the sigmoid function for the given input z.
		"""
		return 1.0 / ( 1.0 + np.exp(-z) )

	def prime( self, z ):
		"""
		The prime method calculates the derivative of the sigmoid function for the given input z. 
		This is used in the back-propagation algorithm to calculate the gradient of the loss with respect to the inputs to each neuron in the network.
		"""
		return self( z ) * ( 1.0 - self( z ) )
		