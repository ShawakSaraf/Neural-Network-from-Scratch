import numpy as np

# Stochastic gradient decent optimizer
class SGD():
	def __init__( self, model = None ):
		self.learning_rate = model.learning_rate
		self.lmbda         = model.lmbda
		self.loss          = model.loss
		self.activation    = model.activation
		self.num_layers    = model.num_layers
		self.model         = model

	def __call__( self, batch, num_data ):
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		for inputs, labels in batch:
			a  = inputs 	# Activation
			As = [inputs] 	# list to store all the Activatin matrix, layer by layer
			Zs = []			# list to store all the z vectors, layer by layer
			self.feed_forward( a, As, Zs )

			delta_Nabla_W, delta_Nabla_B = self.back_propogation(As, Zs, labels)

			# Summing up all the changes in weights amd biases
			nabla_W = [ nw + d_nw for nw, d_nw in zip( nabla_W, delta_Nabla_W ) ]
			nabla_B = [ nb + d_nb for nb, d_nb in zip( nabla_B, delta_Nabla_B ) ]

		# Taking steps towards the lowest loss function
		# lmbda, regularization parameter
		w_coef  = 1 - self.learning_rate * self.lmbda / num_data
		nm_coef = self.learning_rate / len(batch)

		self.model.weights = [
			( w_coef * w ) - ( nm_coef * nw )
			for w, nw in zip( self.model.weights, nabla_W )
		]

		self.model.biases = [ 
			b - ( self.learning_rate / len(batch) ) * nb
			for b, nb in zip( self.model.biases, nabla_B ) 
		]
	
	def feed_forward( self, a, As=None, Zs=None  ):
		"""
		Propogates the inputs forward into the network
		"""
		for W, b in zip( self.model.weights, self.model.biases ):
			z = np.dot( W, a ) + b
			if Zs != None : Zs.append(z)

			a = self.activation(z)
			if As != None : As.append(a)
		return a
	
	def back_propogation( self, As, Zs, labels ):
		"""
		Calculates the gradient of loss function with respect to each variable by back propogation
		and returns gradients. 
		"""
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		delta_error = self.loss.delta( As[-1], labels )
		nabla_W[-1] = np.dot( delta_error, As[-2].transpose() )
		nabla_B[-1] = delta_error

		# Backwards pass / Back-Propogation
		for l in range(2, self.num_layers):
			delta_error = np.dot( self.model.weights[-l + 1].transpose(), delta_error ) * self.activation.prime( Zs[-l] )
			nabla_W[-l] = np.dot( delta_error, As[-l - 1].transpose() )
			nabla_B[-l] = delta_error

		return nabla_W, nabla_B

class CrossEntropyLoss():
	@staticmethod
	def func(a, y):
		return np.sum( np.nan_to_num( -y * np.log(a) - (1-y) * np.log(1-a) ) )

	@staticmethod
	def delta(a, y):
		return ( a - y )
	
class Sigmoid():
	def __call__( self, z ):
		return 1.0 / ( 1.0 + np.exp(-z) )

	def prime( self, z ):
		return self( z ) * ( 1.0 - self( z ) )

class RELU():
	def __call__( self, z ):
		return np.where( z > 0.0, z, 0.0 )

	def prime( self, z ):
		return np.where(z > 0.0, 1.0, 0.0)