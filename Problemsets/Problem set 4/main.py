import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def Q1():
	df = pd.read_csv('x-vs-y-for-PCA.csv')

	#(a): standardization
	x = (df['x'] - np.mean(df['x']))/np.std(df['x'])
	y = (df['y'] - np.mean(df['y']))/np.std(df['y'])

	#(b): calculating the covariance matrix
	n = (len(x) - 1)
	Ex = np.sum(x) / n
	Ey = np.sum(y) / n
	C = np.sum((x - Ex) * (y - Ey)) / n

	print('Self calculated covariance:', C)
	C = np.cov(x, y)
	print('Covariance matrix from numpy:\n', C)

	#(c)
	w, v = np.linalg.eig(C)
	print('\nEigenvalues:', w)
	print('Eigenvectors:\n', v)

	#(d)
	plt.figure()
	ax = plt.axes()
	#plt.scatter(x, y, alpha = 0.6)
	plt.scatter(df['x'], df['y'], alpha = 0.6)
	for vec in v:
		ax.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.5, fc='k', ec='k')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('Q1d.png', dpi = 200, bbox_inches = 'tight')
	plt.close()
	
	#(e)
	ecp = np.zeros((len(x), 2))
	for i in range(len(x)):
		ecp[i] = np.matmul(v.T, np.array([x[i], y[i]]).T)
	#print(ecp)

	plt.figure()
	plt.scatter(ecp[:,0], ecp[:,1], alpha = 0.6)
	plt.savefig('Q1e.png', dpi = 200, bbox_inches = 'tight')
	plt.close()

	#(f)
	PC1 = np.zeros(len(x))
	for i in range(len(x)):
		PC1[i] = np.matmul(v[0].T, np.array([x[i], y[i]]).T)

	x_proj = PC1 * v[0][0]
	y_proj = PC1 * v[0][1]
	
	#undo standardization
	x_proj = x_proj * np.std(df['x']) + np.mean(df['x'])
	y_proj = y_proj * np.std(df['y']) + np.mean(df['y'])
	
	plt.figure()
	plt.scatter(x_proj, y_proj, alpha = 0.6)
	plt.savefig('Q1f.png', dpi = 200, bbox_inches = 'tight')
#	plt.close()
	plt.show()
	
Q1()
