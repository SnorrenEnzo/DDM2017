import numpy as np
from astropy.table import Table
from sklearn.linear_model import LinearRegression

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

#load the train data
train_d = Table().read(dloc + 'PhotoZFileA.vot')
keys = train_d.keys()
print(keys)
Xtrain = np.array([train_d['mag_r'], train_d['u-g'], train_d['g-r'], train_d['r-i'], train_d['i-z']]).T
print(Xtrain[0])
Ytrain = train_d['z_spec']

#apply the linear model
lin_model = LinearRegression(n_jobs = -1).fit(Xtrain, Ytrain)

print(lin_model.score(Xtrain, Ytrain))

#find the training error
prediction = lin_model.predict(Xtrain)
E_train = error(prediction, train_d['z_spec'])

print(E_train)

