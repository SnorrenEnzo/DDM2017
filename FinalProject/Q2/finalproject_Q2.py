import numpy as np
from astropy.table import Table

def loadData(dloc = 'Tables/'):
	"""
	Load the train and test data
	"""
	#load the train data
	train_d = Table().read(dloc + 'PhotoZFileA.vot')
	keys = train_d.keys()
	# print(keys)
	Xtrain = np.array([train_d['mag_r'], train_d['u-g'], train_d['g-r'], train_d['r-i'], train_d['i-z']]).T
	Ytrain = np.array(train_d['z_spec'])

	#load the train data
	test_d = Table().read(dloc + 'PhotoZFileB.vot')
	keys = test_d.keys()
	# print(keys)
	Xtest = np.array([test_d['mag_r'], test_d['u-g'], test_d['g-r'], test_d['r-i'], test_d['i-z']]).T
	Ytest = np.array(test_d['z_spec'])

	return Xtrain, Xtest, Ytrain, Ytest

def error(z_phot, z_spec):
	"""
	Compute the error.

	Input:
		z_phot, z_spec (float arrays): the photometric (predicted) and spectroscopic redshifts.

	Output:
		error (float): the error on the prediction.
	"""
	return np.median(np.abs(z_spec - z_phot) / (1 + z_spec))

def linear(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply linear regression
	"""
	from sklearn.linear_model import LinearRegression

	print('\nLinear regression:')

	#apply the linear model
	reg = LinearRegression(n_jobs = -1).fit(Xtrain, Ytrain)
	print('R^2: {0}'.format(reg.score(Xtrain, Ytrain)))

	#find the training error
	prediction = reg.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = reg.predict(Xtest)
	Etest = error(prediction, Ytest)
	print('Test error: {0}'.format(Etest))

def ridge(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply ridge regression
	"""
	from sklearn.linear_model import Ridge

	print('\nRidge regression:')

	alpharange = np.linspace(0.001, 0.10, 100)
	errorlist = np.zeros(len(alpharange))

	#hyperparameter tuning
	for alpha, i in zip(alpharange, range(len(alpharange))):
		reg = Ridge(alpha = alpha).fit(Xtrain, Ytrain)
		prediction = reg.predict(Xtrain)
		errorlist[i] = error(prediction, Ytrain)

	bestloc = np.argmin(errorlist)
	print('Best alpha: {0}'.format(alpharange[bestloc]))
	print('Corresponding error: {0}'.format(errorlist[bestloc]))

	#apply ridge regression
	reg = Ridge(alpha = alpharange[bestloc]).fit(Xtrain, Ytrain)
	print('R^2: {0}'.format(reg.score(Xtrain, Ytrain)))

	#find the training error
	prediction = reg.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = reg.predict(Xtest)
	Etest = error(prediction, Ytest)
	print('Test error: {0}'.format(Etest))

def lasso(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply Lasso regression
	"""
	from sklearn.linear_model import Lasso

	print('\nLasso regression:')

	alpharange = np.logspace(-10, 0, 100)
	errorlist = np.zeros(len(alpharange))

	#hyperparameter tuning
	for alpha, i in zip(alpharange, range(len(alpharange))):
		reg = Lasso(alpha = alpha).fit(Xtrain, Ytrain)
		prediction = reg.predict(Xtrain)
		errorlist[i] = error(prediction, Ytrain)

	bestloc = np.argmin(errorlist)
	print('Best alpha: {0}'.format(alpharange[bestloc]))
	print('Corresponding error: {0}'.format(errorlist[bestloc]))

	#apply ridge regression
	reg = Lasso(alpha = alpharange[bestloc]).fit(Xtrain, Ytrain)
	print('R^2: {0}'.format(reg.score(Xtrain, Ytrain)))

	#find the training error
	prediction = reg.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = reg.predict(Xtest)
	Etest = error(prediction, Ytest)
	print('Test error: {0}'.format(Etest))

def RF(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply a random forest
	"""
	from sklearn.ensemble import RandomForestRegressor
	print('\nRandom Forest:')

	clf = RandomForestRegressor(n_estimators=500, n_jobs=-1).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))

def Adaboost(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply the adaboost algorithm
	"""
	from sklearn.ensemble import AdaBoostRegressor
	print('\nAdaboost:')

	clf = AdaBoostRegressor(n_estimators=1000).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))


def Bagging(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply the extra trees regressor
	"""
	from sklearn.ensemble import BaggingRegressor
	print('\nBagging regressor:')

	clf = BaggingRegressor(n_estimators=100, n_jobs=-1).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))

def ExtraTrees(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply the extra trees regressor
	"""
	from sklearn.ensemble import ExtraTreesRegressor
	print('\nExtra trees regressor:')

	clf = ExtraTreesRegressor(n_estimators=100, n_jobs=-1).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))

def SVM(Xtrain, Ytrain, Xtest, Ytest, kerneltype = 'rbf'):
	"""
	Apply a support vector machine regressor
	"""
	from sklearn.svm import SVR
	print('\nSVM regressor:')

	clf = SVR(epsilon = 0.02, kernel = kerneltype).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))

def NearestNeighbour(Xtrain, Ytrain, Xval, Yval, n_folds = 5, k_best = None):
	"""
	Apply a nearest neighbour regressor. Find the optimal number of neighbours
	with cross validation and grid search.
	"""
	import matplotlib.pyplot as plt
	from sklearn.neighbors import KNeighborsRegressor
	import seaborn as sns

	sns.set(font = 'Latin Modern Roman',rc = {'legend.frameon': True})
	print('\nNearest neighbour regressor:')

	#find the best number of neighbours if none is given
	if k_best == None:
		from sklearn.model_selection import KFold
		kf = KFold(n_splits = n_folds)
		#the range of values of the amount of neighbours to test
		k_range = np.arange(3, 100, 1)
		#the array which will store the error of each number of neighbours
		errorlist = np.zeros(len(k_range))	
		#array storing the median error
		med_err = np.zeros(len(k_range))

		print('Finding the best number of neighbours...')
		for k, i in zip(k_range, np.arange(len(k_range))):
			le = []
			#apply 4 fold cross validation
			for train_i, test_i in kf.split(Xtrain, Ytrain):
				#split the data into a train and test set 
				Xtr, Xte = Xtrain[train_i], Xtrain[test_i]
				Ytr, Yte = Ytrain[train_i], Ytrain[test_i]
				model = KNeighborsRegressor(n_neighbors = k, n_jobs = -1).fit(Xtr, Ytr)

				# lhscore = model.score(Xte, Yte)
				err = error(model.predict(Xte), Yte)
				
				le = np.append(le, err)
				
			errorlist[i] = np.mean(le)
			med_err[i] = np.median(le)

			print('Iteration {0}, n_neighbours {1}, mean error: {2}'.format(i, k, errorlist[i]))

		#find the best amount of neighbours. Usually this is 3
		k_best = k_range[np.argmin(errorlist)]

		print('Best number of neighbours using median error: {0}'.format(k_range[np.argmin(med_err)]))
		print('Best number of neighbours using mean error: {0}'.format(k_range[np.argmin(errorlist)]))

		plt.plot(k_range, med_err, label = 'Median error')
		plt.plot(k_range, errorlist, label = 'Mean error')
		plt.xlabel('Number of neighbours')
		plt.ylabel('Error')
		plt.title('Nearest neighbour error for different number of neighbours')
		plt.legend(loc = 'best')
		plt.savefig('Nearest_neighbours_median_mean_error.pdf', dpi = 300)
		plt.close()

	print(f'Number of neighbours: {k_best}')

	model = KNeighborsRegressor(n_neighbors = k_best, n_jobs = -1).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(model.score(Xtrain, Ytrain)))

	#find the training error
	prediction = model.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the validation set error
	prediction = model.predict(Xval)
	Etrain = error(prediction, Yval)
	print('Validation error: {0}'.format(Etrain))


#############################################################################
# Functions for the hand written neural network. This code was written by me 
# for the Coursera course on neural networks. Therefore, it will have many
# similarities with code written by others for this course.
#############################################################################

# np.random.seed(1)

def sigmoid(Z):
	return 1 / (1 + np.exp(-Z)), Z

def relu(Z):
	return np.maximum(Z, 0), Z

def tanh(Z):
	return 6*np.tanh(Z), Z

def sigmoid_backward(dA, cache):
	s = 1 / (1 + np.exp(-cache))
	return dA * s * (1 - s)

def relu_backward(dA, cache):
	dZ = np.array(dA, copy = True)
	dZ[cache <= 0] = 0
	return dZ

def tanh_backward(dA, cache):
	return dA * 6 * (1 / np.cosh(cache)**2)

def initialize_parameters_deep(layer_dims):
	"""
	Arguments:
	layer_dims -- python array (list) containing the dimensions of each layer in our network
	
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""
	
	np.random.seed(3)
	parameters = {}
	L = len(layer_dims)			# number of layers in the network

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		
		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

		
	return parameters

def linear_forward(A, W, b):
	"""
	Implement the linear part of a layer's forward propagation.

	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	
	Z = np.dot(W, A) + b
	
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python dictionary containing "linear_cache" and "activation_cache";
			 stored for computing the backward pass efficiently
	"""
	
	if activation == "sigmoid":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	elif activation == "tanh":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = tanh(Z)
	
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache

def L_model_forward(X, parameters, activation, final_activation):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	
	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
				every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
				the cache of linear_sigmoid_forward() (there is one, indexed L-1)
	"""

	caches = []
	A = X
	L = len(parameters) // 2				  # number of layers in the neural network
	
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters['W{0}'.format(l)], parameters['b{0}'.format(l)], activation)
		caches.append(cache)
	
	# Implement LINEAR -> RELU as the final activation unit. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A, parameters['W{0}'.format(L)], parameters['b{0}'.format(L)], final_activation)

	caches.append(cache)
	
	assert(AL.shape == (1,X.shape[1]))
			
	return AL, caches

def compute_cost(AL, Y):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	
	m = Y.shape[1]

	# Compute loss from aL and y.
	cost = -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
	
	cost = np.squeeze(cost)	  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost

def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, A_prev.T) / m
	db = np.mean(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(W.T, dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	elif activation == "tanh":
		dZ = tanh_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activation, final_activation):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# Initializing the backpropagation
	dAL = AL - Y
	
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
	current_cache = caches[L - 1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, final_activation)
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+2)], current_cache, activation)
		grads["dA" + str(l + 1)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads

def update_parameters(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of L_model_backward
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	"""
	
	L = len(parameters) // 2 # number of layers in the neural network

	# Update rule for each parameter. Use a for loop.
	for l in range(L):
		parameters['W{0}'.format(l+1)] = parameters['W{0}'.format(l+1)] - learning_rate * grads['dW{0}'.format(l+1)]
		parameters['b{0}'.format(l+1)] = parameters['b{0}'.format(l+1)] - learning_rate * grads['db{0}'.format(l+1)]
	return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0001, num_iterations = 3000, print_cost=False, activation = 'relu', final_activation = 'tanh'):#lr was 0.009
	"""
	Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
	
	Arguments:
	X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
	Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
	layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
	learning_rate -- learning rate of the gradient descent update rule
	num_iterations -- number of iterations of the optimization loop
	print_cost -- if True, it prints the cost every 100 steps
	
	Returns:
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""

	import matplotlib.pyplot as plt

	# np.random.seed(1)
	costs = []						 # keep track of cost
	
	# Parameters initialization.
	parameters = initialize_parameters_deep(layers_dims)
	
	# Loop (gradient descent)
	for i in range(0, num_iterations):

		# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
		AL, caches = L_model_forward(X, parameters, activation, final_activation)
		
		# Compute cost.
		cost = compute_cost(AL, Y)
	
		# Backward propagation.
		grads = L_model_backward(AL, Y, caches, activation, final_activation)
 
		# Update parameters.
		parameters = update_parameters(parameters, grads, learning_rate)
				
		# Print the cost every 100 training example
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
			
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	
	return parameters

Xtrain, Xtest, Ytrain, Ytest = loadData()

print(f'Shape of training data: {Xtrain.shape}')

# linear(Xtrain, Ytrain, Xtest, Ytest)
# ridge(Xtrain, Ytrain, Xtest, Ytest)
# lasso(Xtrain, Ytrain, Xtest, Ytest)

# RF(Xtrain, Ytrain, Xtest, Ytest)
# Adaboost(Xtrain, Ytrain, Xtest, Ytest)
# Bagging(Xtrain, Ytrain, Xtest, Ytest)
# ExtraTrees(Xtrain, Ytrain, Xtest, Ytest)
# SVM(Xtrain, Ytrain, Xtest, Ytest, kerneltype = 'linear')
NearestNeighbour(Xtrain, Ytrain, Xtest, Ytest)

'''
#run the Neural Network
layers_dims = [5, 8, 6, 3, 1]
parameters = L_layer_model(Xtrain.T, Ytrain[:,None].T, layers_dims, num_iterations = 1000, print_cost = True, activation = 'relu')

#now input the test set and see the results
AL, cache = L_model_forward(Xtest.T, parameters, 'relu', 'tanh')

print(AL[:100])

print('Neural network error: {0}'.format(error(AL[0], Ytest)))
'''