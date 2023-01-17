import numpy as np
from tqdm.auto import tqdm
from Optimizer import SGD, CrossEntropyLoss, Sigmoid
import time, json, sys, os

curr_dir = os.path.dirname( os.path.abspath( __file__ ) )
trained_model_file = curr_dir+r'\TrainedModelData.sav'

class MLP( object ):
	"""
	A basic Multi-Layers Perceptron model, which represents a basic neural network model. 
	The class constructor ( __init__ ) takes several parameters that specify the model's configuration, 
	such as the number of layers in the network, the type of optimizer and loss function to use, and the learning rate. 
	The fit method trains the model on a given dataset, while the predict method uses the trained model 
	to make inference on new data.
	The MLP class also includes several helper methods such as __make_batches which divides the training data 
	into batches, and __load_variables which loads previously saved weights and biases for the model.
	"""
	def __init__( 
			self,
			layers        = [784,30,10],
			optimizer     = SGD,
			loss          = CrossEntropyLoss,
			activation    = Sigmoid,
			learning_rate = 0.1,
			lmbda         = 5.0,
		):
		self.num_layers    = len(layers)
		self.layers        = layers
		self.loss          = loss()
		self.activation    = activation()
		self.learning_rate = learning_rate
		self.lmbda         = lmbda
		self.optimizer     = optimizer( model=self )
		self.weights       = [ 
			np.random.randn(y, x) / np.sqrt(x)
			for x, y in zip( layers[:-1], layers[1:] )
		]
		self.biases	= [ np.random.randn(y, 1) for y in layers[1:] ]
		self.is_last_epoch = False
		self.num_train_data = 0
		self.epoch_start_time = 0
	
	def fit(
			self, 
			train_data     = None,
			val_data       = None,
			epochs         = 10,
			batch_size     = 32,
			continue_train = False,
			monitor        = None
		):
		self.num_train_data = len( train_data )

		if ( continue_train ):
			self.__load_viriables()
	
		epochs_progress_bar = tqdm( range(epochs), position=0 )
		for epoch in epochs_progress_bar:
			"""
			This is the training loop for our neural network. 
			It iterates through a number of epochs (passes over the training data) and within each epoch, 
			it splits the training data into batches and trains the model on each batch using the specified optimizer.
			For each epoch, it also calls the monitor function to record the training and validation metrics.
			The epochs_progress_bar and batches_progress_bar are using the tqdm library to show progress bars for the training loop.
			"""
			self.epoch_start_time   = time.time()
			self.is_last_epoch = True if epoch+1 == epochs else False
			epochs_progress_bar.set_description( f'Epoch {epoch+1}' )
			batches = self.__make_batches( train_data, batch_size )

			batches_progress_bar = tqdm(batches, desc=f'Batch', position=1, ascii=False, leave=False)
			for batch in batches_progress_bar:
				self.optimizer( batch, self.num_train_data )

			if ( monitor != None ):
				monitor( self, epoch, self.epoch_start_time, self.num_train_data, train_data, val_data )
			
	def __load_viriables(self):
		with open(trained_model_file, "r") as f:
			data = json.load(f)
		self.weights = [np.array(w) for w in data["weights"]]
		self.biases  = [np.array(b) for b in data["biases"]]
		print( "Last saved, Hidden neurons: ", len( self.weights[0] ), ", ", data["accuracy"] )
	
	def __make_batches( self, data, batch_size ):
		np.random.shuffle( data )
		# Creating batches out of training data
		batches = [
			data[ k : k + batch_size ]
			for k in range( 0, self.num_train_data, batch_size )
		]
		return batches

	def predict(self, x):
		a = self.optimizer.forward_propagation( x )
		return ( np.argmax(a) )
		
class Monitor():
	"""
	This class monitors the performance of the model. 
	It has attributes for tracking whether training and validation data should be monitored, 
	as well as whether the current model parameters should be saved if they yield the best performance so far. 
	The Monitor class also has methods for computing the total loss and accuracy on a given dataset, 
	and for printing the current values of these metrics. 
	When the __call__ method of the Monitor instance is invoked, it updates the relevant metrics and prints them.
	"""
	def __init__( self, training=False, validation=True, save=False ):
		self.training   = training
		self.validation = validation
		self.save       = save

		self.evaluation_loss, self.evaluation_accuracy = [], []
		self.training_loss, self.training_accuracy = [], []
		self.maxAccuracy = 0;
		self.max_w       = []
		self.max_b       = []
		self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy = 0,0,0,0
	
	def __call__( self, model, epoch, epoch_start_time, num_train_data, train_data, val_data ):
		self.model = model
		self.is_val_data = val_data != None
		
		if self.training:
			self.train_loss = self.total_loss(train_data)
			self.training_loss.append(self.train_loss)
	
			self.train_accuracy = self.accuracy(train_data, convert=True)
			self.training_accuracy.append(self.train_accuracy)

		if self.is_val_data & self.validation:
			self.num_val_data = len( val_data )
			self.val_loss = self.total_loss(val_data, convert=True)
			self.evaluation_loss.append(self.val_loss)

			self.val_accuracy = self.accuracy(val_data)
			self.evaluation_accuracy.append(self.val_accuracy)
		
		self.print_metrics( epoch, epoch_start_time, num_train_data )

	def print_metrics( self, epoch, epoch_start_time, num_train_data ):
		"""
		Prints the relevant training metrics to the console.
		"""
		loss       = round( self.train_loss, 2 ) if self.training  else 0
		accuracy   = round( ( self.train_accuracy/num_train_data ), 2 ) if self.training  else 0
		v_loss     = round( self.val_loss, 2 ) if self.is_val_data & self.validation  else 0
		v_accuracy = round( ( self.val_accuracy/self.num_val_data ), 2 ) if self.is_val_data & self.validation  else 0
		epoch_time = str( round( time.time() - epoch_start_time, 2 ) ) + "s"
		if self.training & self.is_val_data & self.validation:
			tqdm.write( 
				f'{str(epoch+1):3s}: {"loss"}: {str(loss):4s} — {"accuracy"}: {str(accuracy):4s} — {"val_loss"}: {str(v_loss):4s} — {"val_accuracy"}: {str(v_accuracy):4s} — {epoch_time}' 
			)
			
		elif self.training & ( (self.is_val_data == False) or (self.validation == False) ):
			tqdm.write( 
				f'{str(epoch+1):3s}: {"loss"}: {str(loss):4s} — {"accuracy"}: {str(accuracy):4s} — {epoch_time}' 
			)

		elif self.is_val_data & self.validation & (self.training == False):
			tqdm.write( 
				f'{str(epoch+1):3s}: {"val_loss"}: {str(v_loss):4s} — {"val_accuracy"}: {str(v_accuracy):4s} — {epoch_time}' 
			)

		if ( self.val_accuracy > self.maxAccuracy ):
			self.maxAccuracy = self.val_accuracy
			self.max_w, self.max_b = self.model.weights, self.model.biases

		if ( self.save ): 
			self.save_model()
			if ( self.model.is_last_epoch ):
				self.model.weights, self.model.biases = self.max_w, self.max_b
				self.save_model()

	def save_model( self ):
		"""
		Saves the model variables.
		"""
		data = {
			"layers"       : self.model.layers,
			"weights"      : [ w.tolist() for w in self.model.weights ],
			"biases"       : [ b.tolist() for b in self.model.biases ],
			"optimizer"    : str( self.model.optimizer.__class__.__name__ ),
			"loss"         : str( self.model.loss.__class__.__name__ ),
			"activation"   : str( self.model.activation.__class__.__name__ ),
			"learning_rate": self.model.learning_rate,
			"lmbda"        : self.model.lmbda,
			"accuracy"     : self.maxAccuracy,
		}
		with open(trained_model_file, "w") as f:
			json.dump(data, f)

	def history(self):
		return self.evaluation_loss, self.evaluation_accuracy, \
			self.training_loss, self.training_accuracy

	def accuracy(self, data, convert=False):
		"""
		The accuracy method takes a list of data as input, where each data point consists of input values and 
		corresponding labels. The convert parameter determines whether the labels in the data should be converted 
		to their index in the label vector (for example, the label vector [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 
		would be converted to 2). 
		The method computes the accuracy of the model's predictions on the input data by comparing the 
		model's predicted label with the true label for each data point. The predicted labels are obtained by 
		using the model's forward_propagation method on the input values. 
		The method returns the percentage of correct predictions.
		"""
		results = [
			( np.argmax( self.model.optimizer.forward_propagation( x ) ), np.argmax( y ) )
			for x, y in tqdm( data, desc="Accuracy", leave=False, position=1 )
		] if convert else [
			( np.argmax( self.model.optimizer.forward_propagation( x ) ), y )
			for x, y in tqdm( data, desc="Accuracy", leave=False, position=1 )
		]
		return sum( int( x == y ) for ( x, y ) in results )

	def total_loss(self, data, convert=False):
		"""
		This method calculates the average loss across all the examples in the dataset.
		It also includes a regularization term (lambda) to penalize large weights.
		"""
		loss = 0.0
		for x, y in tqdm(data, desc="Cost", leave=False, position=1):
			a = self.model.optimizer.forward_propagation(x)
			y = vectorized_result(y) if convert else y
			loss += self.model.loss.func(a, y)/len(data)

		loss += 0.5*( self.model.optimizer.lmbda / len( data ) ) * \
			sum( np.linalg.norm(w)**2 for w in self.model.weights )
		return loss

def load_model():
	"""
	Returns a trained model.
	"""
	with open(trained_model_file, "r") as f:
		data = json.load(f)

	optimizer  = getattr(sys.modules[__name__], data["optimizer"])
	loss       = getattr(sys.modules[__name__], data["loss"])
	activation = getattr(sys.modules[__name__], data["activation"])
	model      = MLP( 
		data["layers"],
		optimizer     = optimizer,
		loss          = loss,
		activation    = activation,
		learning_rate = data["learning_rate"],
		lmbda         = data["lmbda"],
	)

	model.weights = [np.array(w) for w in data["weights"]]
	model.biases  = [np.array(b) for b in data["biases"]]
	return model

def model_accuracy():
	"""
	Returns the trained model accuracy.
	"""
	with open(trained_model_file, "r") as f:
		data = json.load(f)
	return data["accuracy"]

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e