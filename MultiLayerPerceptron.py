# type: ignore
import numpy as np
import random, time, json, sys, os
from tqdm import tqdm
from Optimizer import SGD, CrossEntropyCost, Sigmoid

curr_dir = os.path.dirname( os.path.abspath( __file__ ) )
trained_model_file = curr_dir+r'\TrainedModelData.sav'

# A Multi-Layers Perceptron model
class MLP( object ):
	def __init__( 
			self,
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
			train_data     = None,
			val_data       = None,
			epochs         = 10,
			batch_size     = 32,
			save_model     = False,
			continue_train = False,
			monitor        = None
		):
		num_data    = len( train_data )
		maxAccuracy = 0;
		max_w       = []
		max_b       = []

		if val_data :
			n_test = len( val_data )

		if ( continue_train ):
			with open(trained_model_file, "r") as f:
				data = json.load(f)
			self.weights = [np.array(w) for w in data["weights"]]
			self.biases  = [np.array(b) for b in data["biases"]]
			print( "Last saved, Hidden neurons: ", len( self.weights[0] ), ", ", data["accuracy"] )
	
		epochs_progress_bar = tqdm( range(epochs), ncols=85, position=0 )
		for j in epochs_progress_bar:
			epochs_progress_bar.set_description(f'Epoch {j+1}')
			start_T = time.time()
			random.shuffle( train_data )
			# Creating batches out of training data
			batches = [
				train_data[ k : k + batch_size ]
				for k in range( 0, num_data, batch_size )
			]

			batches_progress_bar = tqdm(batches, desc=f'Batch', ncols=75, position=1, ascii=False, leave=False)
			for batch in batches_progress_bar:
				self.optimizer( batch, num_data )

			if ( save_model ): 
				self.save_model( maxAccuracy )

			if ( monitor != None ):
				accuracy = monitor( self, j, num_data, train_data, val_data )

			if ( accuracy/n_test*100 > maxAccuracy ):
				maxAccuracy = accuracy/n_test*100
				max_w, max_b = self.weights, self.biases
			
			epoch_time =  round( time.time() - start_T, 2 )
			tqdm.write( "   " + str( epoch_time ) + "s\n" )

		if ( save_model ):
			self.weights, self.biases = max_w, max_b
			self.save_model( maxAccuracy )
		print( "Max accuracy achieved: ", maxAccuracy)
		# return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

	def save_model( self, accuracy ):
		data = {
	        "layers"       : self.layers,
	        "weights"      : [ w.tolist() for w in self.weights ],
	        "biases"       : [ b.tolist() for b in self.biases ],
	        "optimizer"    : str(self.optimizer.__class__.__name__),
	        "cost"         : str(self.cost.__class__.__name__),
	        "activation"   : str(self.activation.__class__.__name__),
	        "learning_rate": self.learning_rate,
	        "lmbda"        : self.lmbda,
	        "accuracy"     : accuracy,
			}
		with open(trained_model_file, "w") as f:
			json.dump(data, f)

class Monitor():
	def __init__( 
		self, 
		val_cost       = False,
		val_accuracy   = False,
		train_cost     = False,
		train_accuracy = False
	):
		self.val_cost       = val_cost
		self.val_accuracy   = val_accuracy
		self.train_cost     = train_cost
		self.train_accuracy = train_accuracy
		self.evaluation_cost, self.evaluation_accuracy = [], []
		self.training_cost, self.training_accuracy = [], []
	
	def __call__( self, model, epoch, num_data, training_data, evaluation_data ):
		self.model = model
		if self.train_cost:
				cost = self.total_cost(training_data)
				self.training_cost.append(cost)
				tqdm.write( "{}: Training cost	: {}".format(epoch+1, round( cost, 2 ) ) )

		if self.train_accuracy:
			accuracy = self.accuracy(training_data, convert=True)
			self.training_accuracy.append(accuracy)
			tqdm.write("   Training accuracy 	: {}%".format( round( ( accuracy/num_data ) * 100, 2 ) ))

		if self.val_cost:
			cost = self.total_cost(evaluation_data, convert=True)
			self.evaluation_cost.append(cost)
			tqdm.write( "   Evaluation cost 	: {}".format( round( cost, 2 ) ) )

		if self.val_accuracy:
			accuracy = self.accuracy(evaluation_data)
			self.evaluation_accuracy.append(accuracy)
			tqdm.write("   Evaluation accuracy 	: {}%".format(
				accuracy, len(evaluation_data), round( ( accuracy / len( evaluation_data ) )*100, 2 ) ))

		return accuracy

	def history(self):
		return self.evaluation_cost, self.evaluation_accuracy, \
			self.training_cost, self.training_accuracy
		
	def feed_forward( self, a ):
		for b, w in zip( self.model.biases, self.model.weights ):
			a = self.model.activation( np.dot( w, a ) + b )
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
		    cost += self.model.cost.func(a, y)/len(data)
		cost += 0.5*( self.model.optimizer.lmbda / len( data ) )*sum(
		    np.linalg.norm(w)**2 for w in self.model.weights)
		return cost

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