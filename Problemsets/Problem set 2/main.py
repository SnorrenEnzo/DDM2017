import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def question2():
	import sqlite3 as lite

	con = lite.connect("ThirteenDatasets.db")
	#cur = con.cursor()
	
	#cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
	#print(cur.fetchall())
	
	t = pd.read_sql_query('SELECT * FROM Set02', con)
	
		# Set style of scatterplot
	plt.rc('text', usetex=True)
	sns.set_context("notebook", font_scale=1.1)
	sns.set_style("ticks")
	
	# Create scatterplot of dataframe
	sns.lmplot('x', # Horizontal axis
		         'y', # Vertical axis
		         data=t, # Data source
		         fit_reg=True) # Don't fix a regression line
		          # S marker size
	
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
	
	plt.figure()
	
	x = t['x']
	print(x[0:10])
	sns.distplot(x[0:10])
	
	plt.show()
	
def pickle_from_file(fname):
	import pickle

	fh = open(fname, 'rb')
	data = pickle.load(fh)
	fh.close()
	return data
	
	
def lnprob(theta, x, y, yerr):
	"""
	The likelihood to include in the MCMC.
	"""
	def lnL(theta, x, y, yerr):
		a, b = theta
		model = b * x + a
		inv_sigma2 = 1.0/(yerr**2)
		return -0.5*(np.sum((y-model)**2*inv_sigma2))
	
	def lnprior(theta):
		a, b = theta
		if -5.0 < a < 5.0 and -10.0 < b < 10.0:
			return 0.0
		return -np.inf
	
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnL(theta, x, y, yerr)
	
def findMaxLikelyhood(data):
	a = np.linspace(-5, 5, num = 100)
	b = np.linspace(-5, 5, num = 100)
	
	likelyhood = np.zeros((len(a), len(b)))
	
	for i in np.arange(len(a)):
		print(i)
		for j in np.arange(len(b)):
			likelyhood[i][j] = lnprob((a[i], b[j]), data['x'], data['y'], data['sigma'])
		
	loc = np.argwhere(likelyhood == np.max(likelyhood))
	print(loc)
	return a[loc[0][0]], b[loc[0][1]]

def question3():
	import emcee
	
	d = pickle_from_file('points_example1.pkl')
	
	print(findMaxLikelyhood(d))
	
	plt.rc('text', usetex=True)
	sns.set_context("notebook", font_scale=1.1)
	sns.set_style("ticks")
	
	sns.lmplot('x', 'y', data = pd.DataFrame.from_dict(d), fit_reg = False)
	plt.show()
	

def main():

	question3()
	
if __name__ == "__main__":
	main()
