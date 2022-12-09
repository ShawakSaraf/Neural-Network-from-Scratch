# type: ignore
from matplotlib.style import available
import numpy as np
import random, time, json, sys, os
from tqdm import tqdm

curr_dir = os.path.dirname( os.path.abspath( __file__ ) )
trained_model_file = curr_dir+r'\TrainedModelData.sav'

class CrossEntropyCost( object ):
	@staticmethod
	def func(a, y):
		return np.sum( np.nan_to_num( -y*np.log(a) - (1-y) * np.log(1-a) ) )

	@staticmethod
	def delta(a, y):
		return ( a - y )

class Sigmoid( object ):
	def __call__( self, z ):
		return 1.0 / ( 1.0 + np.exp(-z) )

	def prime( self, z ):
		return self( z ) * ( 1.0 - self( z ) )

# Stochastic gradient decent
class SGD():
	def __init__(self, model=None ):
		self.learning_rate = model.learning_rate
		self.lmbda         = model.lmbda
		self.cost          = model.cost
		self.activation    = model.activation
		self.num_layers    = model.num_layers
		self.model         = model

	def __call__( self, batch, num_data ):
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		for inputs, labels in batch:
			delta_Nabla_W, delta_Nabla_B = self.propagate(inputs, labels)

			# Summing up all the changes in weights amd biases
			nabla_W = [ nw + d_nw for nw, d_nw in zip( nabla_W, delta_Nabla_W ) ]
			nabla_B = [ nb + d_nb for nb, d_nb in zip( nabla_B, delta_Nabla_B ) ]

		# Taking steps towards the lowest cost function
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

	def propagate( self, inputs, labels ):
		"""
		Propogates the inputs forward, calculates the gradient of loss function
		with respect to each variable by back propogation and returns it. 
		"""
		nabla_W = [ np.zeros( w.shape ) for w in self.model.weights ]
		nabla_B = [ np.zeros( b.shape ) for b in self.model.biases ]

		a  = inputs 	# Activation
		As = [inputs] 	# list to store all the Activatin matrix, layer by layer
		Zs = []			# list to store all the z vectors, layer by layer

		# Forward pass
		for W, b in zip( self.model.weights, self.model.biases ):
			z = np.dot( W, a ) + b
			Zs.append(z)
			a = self.activation(z)
			As.append(a)

		# Backwards pass / Back-Propogation
		delta_error = self.cost.delta( As[-1], labels )
		nabla_W[-1] = np.dot( delta_error, As[-2].transpose() )
		nabla_B[-1] = delta_error

		for l in range(2, self.num_layers):
			delta_error = np.dot( self.model.weights[-l + 1].transpose(), delta_error ) * self.activation.prime( Zs[-l] )
			nabla_W[-l] = np.dot( delta_error, As[-l - 1].transpose() )
			nabla_B[-l] = delta_error

		return ( nabla_W, nabla_B )

# A Multi-Layers Perceptron model
class MLP( object ):
	def __init__( self,
		layers        = [784,30,10],
		optimizer     = SGD,
		cost          = CrossEntropyCost,
		activation    = Sigmoid,
		learning_rate = 0.1,
		lmbda         = 5.0,
	):
		self.num_layers    = len(layers)
		self.layers        = layers
		self.cost          = cost()
		self.activation    = activation()
		self.learning_rate = learning_rate
		self.lmbda         = lmbda
		self.optimizer     = optimizer( model=self )
		self.weights       = [ 
			np.random.randn(y, x) / np.sqrt(x)
			for x, y in zip( layers[:-1], layers[1:] )
		]
		self.biases	= [ np.random.randn(y, 1) for y in layers[1:] ]
	
	def fit(
		self, 
		training_data               = None,
		evaluation_data             = None,
		epochs                      = 10,
		batch_size                  = 32,
		save                        = False,
		continue_LastSaved          = False,
		monitor_evaluation_cost     = False,
		monitor_evaluation_accuracy = False,
		monitor_training_cost       = False,
		monitor_training_accuracy   = False
	):
		num_data    = len( training_data )
		maxAccuracy = 0;
		max_w       = []
		max_b       = []

		evaluation_cost, evaluation_accuracy = [], []
		training_cost, training_accuracy = [], []

		if evaluation_data :
			n_test = len( evaluation_data )

		if ( continue_LastSaved ):
			with open(trained_model_file, "r") as f:
				data = json.load(f)
			self.weights = [np.array(w) for w in data["weights"]]
			self.biases  = [np.array(b) for b in data["biases"]]
			print( "Last saved, Hidden neurons: ", len(self.weights[0]), ", ", data["accuracy"] )
	
		epochs_progress_bar = tqdm( range(epochs), ncols=85, position=0 )
		stagnantCount = 0
		for j in epochs_progress_bar:
			epochs_progress_bar.set_description(f'Epoch {j+1}')
			start_T = time.time()
			random.shuffle( training_data )
			# Creating batches out of training data
			batches = [
				training_data[ k : k + batch_size ]
				for k in range( 0, num_data, batch_size )
			]

			batches_progress_bar = tqdm(batches, desc=f'Batch', ncols=75, position=1, ascii=False, leave=False)
			# print( "old weights average: ", np.average( self.weights ) )
			for batch in batches_progress_bar:
				self.optimizer( batch, num_data )
				# print( "new weights average: ", np.average( self.weights ) )

			if ( save ): self.save(trained_model_file, maxAccuracy)

			if monitor_training_cost:
				cost = self.total_cost(training_data)
				training_cost.append(cost)
				tqdm.write("{}: Training cost	: {}".format(j+1,cost))

			if monitor_training_accuracy:
				accuracy = self.accuracy(training_data, convert=True)
				training_accuracy.append(accuracy)
				tqdm.write("   Training accuracy 	: {} / {},	{}%".format( accuracy, num_data, round((accuracy/num_data)*100, 2)))

			if monitor_evaluation_cost:
				cost = self.total_cost(evaluation_data, convert=True)
				evaluation_cost.append(cost)
				tqdm.write("   Evaluation cost 	: {}".format(cost))

			if monitor_evaluation_accuracy:
				accuracy = self.accuracy(evaluation_data)
				evaluation_accuracy.append(accuracy)
				tqdm.write("   Evaluation accuracy 	: {} / {},		{}%".format(
					accuracy, len(evaluation_data), round((accuracy/len(evaluation_data))*100, 2), stagnantCount))
			if ( accuracy/n_test*100 > maxAccuracy ):
				maxAccuracy = accuracy/n_test*100
				max_w, max_b = self.weights, self.biases
			tqdm.write("")

		if ( save ):
			self.weights, self.biases = max_w, max_b
			self.save(trained_model_file, maxAccuracy)
		print(  maxAccuracy)
		return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

	def feed_forward( self, a ):
		for b, w in zip(self.biases, self.weights):
			a = self.activation( np.dot( w, a ) + b )
		return a

	def accuracy(self, data, convert=False):
		if convert:
		    results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
		               for (x, y) in tqdm(data, desc="Accuracy", ncols=75, leave=False, position=1)]
		else:
		    results = [(np.argmax(self.feed_forward(x)), y)
		                for (x, y) in tqdm(data, desc="Accuracy", ncols=75, leave=False, position=1)]
		return sum(int(x == y) for (x, y) in results)

	def total_cost(self, data, convert=False):
		cost = 0.0
		for x, y in tqdm(data, desc="Cost", ncols=75, leave=False, position=1):
		    a = self.feed_forward(x)
		    if convert:
		    	y = vectorized_result(y)
		    cost += self.cost.func(a, y)/len(data)
		cost += 0.5*( self.optimizer.lmbda / len( data ) )*sum(
		    np.linalg.norm(w)**2 for w in self.weights)
		return cost

	def save(self, trained_model_file, accuracy):
		data = {
	        "layers"       : self.layers,
	        "weights"      : [ w.tolist() for w in self.weights ],
	        "biases"       : [ b.tolist() for b in self.biases ],
	        "optimizer"    : str(self.optimizer.__name__),
	        "cost"         : str(self.cost.__name__),
	        "activation"   : str(self.activation.__name__),
	        "learning_rate": self.learning_rate,
	        "lmbda"        : self.lmbda,
	        "accuracy"     : accuracy,
			}
		with open(trained_model_file, "w") as f:
			json.dump(data, f)

def load_model( trained_model_file ):
	with open(trained_model_file, "r") as f:
		data = json.load(f)

	optimizer  = getattr(sys.modules[__name__], data["optimizer"])
	cost       = getattr(sys.modules[__name__], data["cost"])
	activation = getattr(sys.modules[__name__], data["activation"])
	model      = MLP( 
		data["layers"],
		optimizer     = optimizer,
		cost          = cost,
		activation    = activation,
		learning_rate = data["learning_rate"],
		lmbda         = data["lmbda"],
	)

	model.weights = [np.array(w) for w in data["weights"]]
	model.biases 	= [np.array(b) for b in data["biases"]]
	return model

def predict(x):
	a = x
	net = load_model(trained_model_file)
	for W, b in zip( net.weights, net.biases):
		z = np.dot(W, a) + b
		a = 1.0 / (1.0 + np.exp(-z))
	return ( np.argmax(a) )

def model_accuracy():
	with open(trained_model_file, "r") as f:
		data = json.load(f)
	return data["accuracy"]

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e