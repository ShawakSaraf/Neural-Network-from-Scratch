# type: ignore
import numpy as np
import random, time, json, sys, os
from tqdm import tqdm
from Optimizer import SGD, CrossEntropyLoss, Sigmoid

curr_dir = os.path.dirname( os.path.abspath( __file__ ) )
trained_model_file = curr_dir+r'\TrainedModelData.sav'

# A Multi-Layers Perceptron model
class MLP( object ):
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
	
	def fit(
			self, 
			train_data     = None,
			val_data       = None,
			epochs         = 10,
			batch_size     = 32,
			continue_train = False,
			monitor        = None
		):
		num_data    = len( train_data )

		if ( continue_train ):
			with open(trained_model_file, "r") as f:
				data = json.load(f)
			self.weights = [np.array(w) for w in data["weights"]]
			self.biases  = [np.array(b) for b in data["biases"]]
			print( "Last saved, Hidden neurons: ", len( self.weights[0] ), ", ", data["accuracy"] )
	
		epochs_progress_bar = tqdm( range(epochs), ncols=85, position=0 )
		for epoch in epochs_progress_bar:
			self.is_last_epoch = True if epoch+1 == epochs else False
			start_time = time.time()
			epochs_progress_bar.set_description(f'Epoch {epoch+1}')
			random.shuffle( train_data )
			# Creating batches out of training data
			batches = [
				train_data[ k : k + batch_size ]
				for k in range( 0, num_data, batch_size )
			]

			batches_progress_bar = tqdm(batches, desc=f'Batch', ncols=75, position=1, ascii=False, leave=False)
			for batch in batches_progress_bar:
				self.optimizer( batch, num_data )

			if ( monitor != None ):
				monitor( self, epoch, start_time, num_data, train_data, val_data )

	def predict(self, x):
		a = self.optimizer.feed_forward( x )
		return ( np.argmax(a) )
		
class Monitor():
	def __init__( self, training=False, validation=True, save=False ):
		self.training   = training
		self.validation = validation
		self.save = save

		self.evaluation_loss, self.evaluation_accuracy = [], []
		self.training_loss, self.training_accuracy = [], []
		self.maxAccuracy = 0;
		self.max_w       = []
		self.max_b       = []
	
	def __call__( self, model, epoch, start_time, num_data, train_data, val_data ):
		self.model = model
		train_loss, train_accuracy, val_loss, val_accuracy = 0,0,0,0
		if self.training:
			train_loss = self.total_loss(train_data)
			self.training_loss.append(train_loss)

			train_accuracy = self.accuracy(train_data, convert=True)
			self.training_accuracy.append(train_accuracy)

		if self.validation:
			val_loss = self.total_loss(val_data, convert=True)
			self.evaluation_loss.append(val_loss)

			val_accuracy = self.accuracy(val_data)
			self.evaluation_accuracy.append(val_accuracy)

		loss       = round( train_loss, 2 ) if self.training  else 0
		accuracy   = round( ( train_accuracy/num_data ), 2 ) if self.training  else 0
		v_loss     = round( val_loss, 2 ) if self.validation  else 0
		v_accuracy = round( ( val_accuracy/len( val_data ) ), 2 ) if self.validation  else 0
		epoch_time = str( round( time.time() - start_time, 2 ) ) + "s"
		if self.training & self.validation:
			tqdm.write( 
				f'{str(epoch+1):3s}: {"loss"}: {str(loss):4s} — {"accuracy"}: {str(accuracy):4s} — {"val_loss"}: {str(v_loss):4s} — {"val_accuracy"}: {str(v_accuracy):4s} — {epoch_time}' 
			)
			
		elif self.training & (self.validation == False):
			tqdm.write( 
				f'{str(epoch+1):3s}: {"loss"}: {str(loss):4s} — {"accuracy"}: {str(accuracy):4s} — {epoch_time}' 
			)

		elif self.validation & (self.training == False):
			tqdm.write( 
				f'{str(epoch+1):3s}: {"val_loss"}: {str(v_loss):4s} — {"val_accuracy"}: {str(v_accuracy):4s} — {epoch_time}' 
			)

		if ( val_accuracy > self.maxAccuracy ):
			self.maxAccuracy = val_accuracy
			self.max_w, self.max_b = self.model.weights, self.model.biases

		if ( self.save ): 
			self.save_model()
			if ( self.model.is_last_epoch ):
				self.model.weights, self.model.biases = self.max_w, self.max_b
				self.save_model()

	def save_model( self ):
		data = {
			"layers"       : self.model.layers,
			"weights"      : [ w.tolist() for w in self.model.weights ],
			"biases"       : [ b.tolist() for b in self.model.biases ],
			"optimizer"    : str(self.model.optimizer.__class__.__name__),
			"loss"         : str(self.model.loss.__class__.__name__),
			"activation"   : str(self.model.activation.__class__.__name__),
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
		results = [
			( np.argmax( self.model.optimizer.feed_forward( x ) ), np.argmax( y ) )
			for x, y in tqdm( data, desc="Accuracy", ncols=75, leave=False, position=1 )
		] if convert else [
			( np.argmax( self.model.optimizer.feed_forward( x ) ), y )
			for x, y in tqdm( data, desc="Accuracy", ncols=75, leave=False, position=1 )
		]
		return sum( int( x == y ) for ( x, y ) in results )

	def total_loss(self, data, convert=False):
		loss = 0.0
		for x, y in tqdm(data, desc="Cost", ncols=75, leave=False, position=1):
			a = self.model.optimizer.feed_forward(x)
			y = vectorized_result(y) if convert else y
			loss += self.model.loss.func(a, y)/len(data)

		loss += 0.5*( self.model.optimizer.lmbda / len( data ) ) * \
			sum( np.linalg.norm(w)**2 for w in self.model.weights )
		return loss

def load_model():
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
	with open(trained_model_file, "r") as f:
		data = json.load(f)
	return data["accuracy"]

def vectorized_result(j):
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e