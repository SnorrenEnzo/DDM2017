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
	lin_model = LinearRegression(n_jobs = -1).fit(Xtrain, Ytrain)
	print('R^2: {0}'.format(lin_model.score(Xtrain, Ytrain)))

	#find the training error
	prediction = lin_model.predict(Xtrain)
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
		lin_model = Ridge(alpha = alpha).fit(Xtrain, Ytrain)
		prediction = lin_model.predict(Xtrain)
		errorlist[i] = error(prediction, Ytrain)

	bestloc = np.argmin(errorlist)
	print('Best alpha: {0}'.format(alpharange[bestloc]))
	print('Corresponding error: {0}'.format(errorlist[bestloc]))

	#apply ridge regression
	lin_model = Ridge(alpha = alpharange[bestloc]).fit(Xtrain, Ytrain)
	print('R^2: {0}'.format(lin_model.score(Xtrain, Ytrain)))

	#find the training error
	prediction = lin_model.predict(Xtrain)
	Etrain = error(prediction, Ytrain)
	print('Training error: {0}'.format(Etrain))

#load the train data
train_d = Table().read(dloc + 'PhotoZFileA.vot')
keys = train_d.keys()
print(keys)
Xtrain = np.array([train_d['mag_r'], train_d['u-g'], train_d['g-r'], train_d['r-i'], train_d['i-z']]).T
Ytrain = train_d['z_spec']

linear(Xtrain, Ytrain)
ridge(Xtrain, Ytrain)


