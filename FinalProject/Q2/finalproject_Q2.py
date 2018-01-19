import numpy as np
from astropy.table import Table

dloc = 'Tables/'

def error(z_phot, z_spec):
	"""
	Compute the error.

	Input:
		z_phot, z_spec (float arrays): the photometric (predicted) and spectroscopic redshifts.

	Output:
		error (float): the error on the prediction.
	"""
	return np.median(np.abs(z_spec - z_phot) / (1 + z_spec))

def linear(Xtrain, Ytrain):
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

def ridge(Xtrain, Ytrain):
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

def lasso(Xtrain, Ytrain):
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

def RF(Xtrain, Ytrain, Xtest, Ytest):
	"""
	Apply a random forest
	"""
	from sklearn.ensemble import RandomForestRegressor
	print('\nRandom Forest:')

	clf = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(Xtrain, Ytrain)
	print('Accuracy: {0}'.format(clf.score(Xtrain, Ytrain)))

	#find the training error
	prediction = clf.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

	#find the test error
	prediction = clf.predict(Xtest)
	Etrain = error(prediction, Ytest)
	print('Test error: {0}'.format(Etrain))


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

# linear(Xtrain, Ytrain)
# ridge(Xtrain, Ytrain)
# lasso(Xtrain, Ytrain)

RF(Xtrain, Ytrain, Xtest, Ytest)
