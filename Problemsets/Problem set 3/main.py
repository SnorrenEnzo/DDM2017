from astropy.table import Table
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


def applyKDE(t, key, bwrange, title, savename = None, xlabel = 'Xlabel'):
	import matplotlib.colors as colors
	import matplotlib.cm as cmx
	import seaborn as sns
	sns.set()
	#t = Table().read('joint-bh-mass-table.csv')
	
	X_plot = np.linspace(np.min(t[key]) - 1, np.max(t[key]) + 1, num = 1000)[:, np.newaxis]
	X = t[key][:, np.newaxis]
	
	plt.scatter(X[:,0], np.zeros(len(X[:,0])), marker = 'x', color = 'black')
	
	#different values for the bandwidth
	#bwrange = np.arange(1, 8)
	
	#plot many lines using colors from a color map, with MAX the amount of lines
	jet = plt.get_cmap('jet') 
	cNorm  = colors.Normalize(vmin=0, vmax=len(bwrange))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	
	for bw, i in zip(bwrange, np.arange(len(bwrange))):
		kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(X)
	
		log_dens = kde.score_samples(X_plot)
		
		#for plotting in many different colours
		colorVal = scalarMap.to_rgba(i)	
	
		plt.plot(X_plot, np.exp(log_dens), label = 'Bandwidth: {0}'.format(round(bw, 2)), color = colorVal)
	
	
	plt.xlabel(xlabel)
	plt.ylabel('Probability')
	plt.legend(loc = 'best')
	
	plt.title(title)
	if savename is not None:
		plt.savefig(savename)
	plt.show()
	
	"""
	The best bandwidth seems to be 6 Msun. This is not necessarily the best estimate.
	"""
	
def question1b(t, key, bwrange):
	import matplotlib.colors as colors
	import matplotlib.cm as cmx
	import seaborn as sns
	sns.set()
	
	t = Table().read('joint-bh-mass-table.csv')
	
	X_plot = np.linspace(np.min(t['MBH']) - 4, np.max(t['MBH']) + 4, num = 1000)[:, np.newaxis]
	X = t['MBH'][:, np.newaxis]
	
	#plt.scatter(X[:,0], np.zeros(len(X[:,0])), marker = 'x', color = 'black')
	
	#different values for the bandwidth
	bwrange = np.arange(1, 10, 0.1)
	
	#plot many lines using colors from a color map, with MAX the amount of lines
	#jet = plt.get_cmap('jet') 
	#cNorm  = colors.Normalize(vmin=0, vmax=len(bwrange))
	#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	
	#set the number of folds
	kf = KFold(n_splits = 5)
	
	likelyhood = np.zeros(len(bwrange))
	
	for bw, i in zip(bwrange, np.arange(len(bwrange))):
		lh = []
		for train_i, test_i in kf.split(X):
			Xtrain, Xtest = X[train_i], X[test_i]
			kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(Xtrain)
	
			log_dens = kde.score_samples(Xtrain)
			lhscore = kde.score(Xtest)
			
			#print('Bandwidth: {0}, Likelyhood: {1}'.format(bw, lhscore))
			
			lh = np.append(lh, lhscore)
			
		likelyhood[i] = np.mean(lh)
			
	print('Highest likelyhood ({0}) at bandwidth = {1}'.format(round(np.max(likelyhood), 2), bwrange[np.argmax(likelyhood)]))
	
	plt.plot(bwrange, likelyhood, color = 'black', alpha = 0.8, label = 'Likelyhood')
	plt.scatter(bwrange[np.argmax(likelyhood)], np.max(likelyhood), marker = 'x', s = 100, color = 'orange', label = 'Maximum likelyhood')
	
	plt.xlabel('Bandwidth [$10^6$ M$_{\odot}$]')
	plt.ylabel('Likelyhood')
	plt.legend(loc = 'best')
	
	plt.title('Black hole mass density distribution')
	#plt.savefig('Blackhole-kde-bandwidth-likelyhood.svg')
	plt.show()
	
def question2():
	import matplotlib.colors as colors
	import matplotlib.cm as cmx
	import seaborn as sns
	sns.set()
	
	t = Table().read('pulsar_masses.vot')
	
	print(len(t['Mass']))
	
	X_plot = np.linspace(np.min(t['Mass']) - 1, np.max(t['Mass']) + 1, num = 1000)[:, np.newaxis]
	X = t['Mass'][:, np.newaxis]
	
	#plt.scatter(X[:,0], np.zeros(len(X[:,0])), marker = 'x', color = 'black')
	
	#different values for the bandwidth
	bwrange = np.linspace(0.03, 1.5, num = 40)
	
	#plot many lines using colors from a color map, with MAX the amount of lines
	#jet = plt.get_cmap('jet') 
	#cNorm  = colors.Normalize(vmin=0, vmax=len(bwrange))
	#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	
	#set the number of folds
	kf = KFold(n_splits = 5)
	
	likelyhood = np.zeros(len(bwrange))
	
	print('Finding the best bandwidth...')
	for bw, i in zip(bwrange, np.arange(len(bwrange))):
		lh = []
		for train_i, test_i in kf.split(X):
			Xtrain, Xtest = X[train_i], X[test_i]
			kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(Xtrain)
	
			log_dens = kde.score_samples(Xtrain)
			lhscore = kde.score(Xtest)
			
			#print('Bandwidth: {0}, Likelyhood: {1}'.format(bw, lhscore))
			
			lh = np.append(lh, lhscore)
			
		likelyhood[i] = np.mean(lh)
	
	bestbandwidth = bwrange[np.argmax(likelyhood)]
	print('Highest likelyhood ({0}) at bandwidth = {1}'.format(round(np.max(likelyhood), 2), bestbandwidth))
	
	'''
	plt.plot(bwrange, likelyhood, color = 'black', alpha = 0.8, label = 'Likelyhood')
	plt.scatter(bwrange[np.argmax(likelyhood)], np.max(likelyhood), marker = 'x', s = 100, color = 'orange', label = 'Maximum likelyhood')
	
	plt.xlabel('Bandwidth [M$_{\odot}$]')
	plt.ylabel('Likelyhood')
	plt.legend(loc = 'best')
	
	plt.title('Neutron star mass density distribution')
	plt.savefig('Neutronstar-kde-bandwidth-likelyhood.svg')
	plt.show()
	'''
	print('Obtaining probabilities')
	#using this found bandwidth, calculate the probability of the mass ranges
	kde = KernelDensity(bandwidth = bestbandwidth, kernel = 'gaussian').fit(X)
	#obtain the density
	dens = np.exp(kde.score_samples(X_plot))
	
	print('\n\nA\n')
	inner1 = 1.8
	stepsize = np.abs(X_plot[:,0][1] - X_plot[:,0][2])
	prob1 = np.sum(dens[X_plot[:,0] > inner1] * stepsize)
	print('Probability of finding a neutron star with M > {0}: {1}'.format(inner1, prob1))
	print('\n\nB\n')
	
	inner2 = 1.36
	outer2 = 2.26
	prob2 = np.sum(dens[(X_plot[:,0] > inner2) * (X_plot[:,0] < outer2)] * stepsize)
	print('Probability of finding a neutron star with {0} < M < {1}: {2}'.format(inner2, outer2, prob2))
	
	inner3 = 0.86
	outer3 = 1.36
	prob3 = np.sum(dens[(X_plot[:,0] > inner3) * (X_plot[:,0] < outer3)] * stepsize)
	print('Probability of finding a neutron star with {0} < M < {1}: {2}'.format(inner3, outer3, prob3))
	
	print('Probability of finding this bindary: {0}'.format(prob2 * prob3))
	
def pickle_from_file(fname):
	"""
	Load a pickle file 
	"""
	import pickle

	fh = open(fname, 'rb')
	data = pickle.load(fh)
	fh.close()
	return data
	
def question3():
	import seaborn as sns
	sns.set()
	print('---------\nPlease use Python 2!\n---------')
	
	data = pickle_from_file('mysterious-peaks.pkl')
		          
	X = data[:, np.newaxis]

	X_plot = np.linspace(np.min(data) - 1, np.max(data) + 1, num = 1000)[:, np.newaxis]
	
	kde = KernelDensity(bandwidth = 0.2, kernel = 'gaussian').fit(X)
	
	log_dens = kde.score_samples(X_plot)
	
	plt.plot(X_plot, np.exp(log_dens), color = 'b')
	plt.scatter(data, np.zeros(len(data)) - np.random.rand(len(data))/10., marker = 'x', color = 'black')
	
	plt.xlabel('Data value')
	plt.ylabel('Probability')
	
	plt.xlim((np.min(X_plot) - 1, np.max(X_plot) + 1))
	plt.ylim((-0.1 - 0.01, 0.15))
	
	plt.savefig('Question3.svg')
	plt.show()
	
def question4():
	from astroML.datasets import fetch_great_wall
	X = fetch_great_wall()
	
	bw = 5 #bandwidth for the KDE

	# Create the grid on which to evaluate the results
	ratio = 50./125.
	sizefactor = 250 #default = 125
	Nx = int(ratio * sizefactor)
	Ny = int(sizefactor)
	
	xmin, xmax = (-375, -175)
	ymin, ymax = (-300, 200)
	
	xgrid = np.linspace(xmin, xmax, Nx)
	ygrid = np.linspace(ymin, ymax, Ny)
	mesh = np.meshgrid(xgrid, ygrid)
	
	tmp = map(np.ravel, mesh)
	Xgrid = np.vstack(tmp).T
	
	def Qa():
			#Make KDEs for the different kernels
		kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(X)
		# Evaluate the KDE on the grid
		log_dens = kde.score_samples(Xgrid)
		dens1 = X.shape[0] * np.exp(log_dens).reshape((Ny, Nx))
	
		kde = KernelDensity(bandwidth = bw, kernel = 'tophat').fit(X)
		# Evaluate the KDE on the grid
		log_dens = kde.score_samples(Xgrid)
		dens2 = X.shape[0] * np.exp(log_dens).reshape((Ny, Nx))
	
		kde = KernelDensity(bandwidth = bw, kernel = 'exponential').fit(X)
		# Evaluate the KDE on the grid
		log_dens = kde.score_samples(Xgrid)
		dens3 = X.shape[0] * np.exp(log_dens).reshape((Ny, Nx))
	
		kde = KernelDensity(bandwidth = bw, kernel = 'epanechnikov').fit(X)
		# Evaluate the KDE on the grid
		log_dens = kde.score_samples(Xgrid)
		dens4 = X.shape[0] * np.exp(log_dens).reshape((Ny, Nx))

		plt.figure(figsize = (12. * 2./5., 12))
		#plt.imshow(dens1, cmap=plt.get_cmap('hot'))
		plt.scatter(X.T[0], X.T[1], edgecolor = 'none', s = 2, color = 'black')
		plt.axis('equal')
		plt.xlim((-375, -175))
		plt.ylim((-300, 200))
		plt.title('Great Wall galaxies')
		plt.xlabel('Width [Mly]')
		plt.ylabel('Height [Mly]')
		plt.savefig('great-wall_raw.svg', bbox_inches = 'tight')
		#plt.show()
		plt.close()
	
		fig, ((x1, x2), (x3, x4)) = plt.subplots(2,2, figsize = (6.5, 12))
		x1.imshow(dens1, interpolation = 'nearest', cmap = plt.get_cmap('hot'))
		x1.set_title("Gaussian KDE")
		x1.set_xlabel('Width [Mly]')
		x1.set_ylabel('Height [Mly]')
		x2.imshow(dens2, interpolation = 'nearest', cmap = plt.get_cmap('hot'))	
		x2.set_title("Tophat KDE")
		x2.set_xlabel('Width [Mly]')
		x2.set_ylabel('Height [Mly]')
		x3.imshow(dens3, interpolation = 'nearest', cmap = plt.get_cmap('hot'))
		x3.set_title("Exponential KDE")
		x3.set_xlabel('Width [Mly]')
		x3.set_ylabel('Height [Mly]')
		x4.imshow(dens4, interpolation = 'nearest', cmap = plt.get_cmap('hot'))
		x4.set_title("Epanechnikov KDE")
		x4.set_xlabel('Width [Mly]')
		x4.set_ylabel('Height [Mly]')
	
		plt.savefig('Question4a.svg', bbox_inches = 'tight')
		print('Best kernel: exponential')
	
		plt.show()
		
	def Qb():
		print('Starting cross validation...')
		
		#different values for the bandwidth
		bwrange = np.linspace(1, 20, 20)

		#set the number of folds
		kf = KFold(n_splits = 10)
	
		likelyhood = np.zeros(len(bwrange))
	
		print('Finding the best bandwidth...')
		for bw, i in zip(bwrange, np.arange(len(bwrange))):
			print('{0} of {1}'.format(i + 1, len(bwrange)))
			lh = []
			for train_i, test_i in kf.split(X):
				Xtrain, Xtest = X[train_i], X[test_i]
				kde = KernelDensity(bandwidth = bw, kernel = 'exponential').fit(Xtrain)
	
				log_dens = kde.score_samples(Xtrain)
				lhscore = kde.score(Xtest)
			
				#print('Bandwidth: {0}, Likelyhood: {1}'.format(bw, lhscore))
			
				lh = np.append(lh, lhscore)
			
			likelyhood[i] = np.mean(lh)
	
		bestbandwidth = bwrange[np.argmax(likelyhood)]
		print('Highest likelyhood ({0}) at bandwidth = {1}'.format(round(np.max(likelyhood), 2), bestbandwidth))
		
		plt.plot(bwrange, likelyhood, color = 'black', alpha = 0.8, label = 'Likelyhood')
		plt.scatter(bwrange[np.argmax(likelyhood)], np.max(likelyhood), marker = 'x', s = 100, color = 'orange', label = 'Maximum likelyhood')
	
		plt.xlabel('Bandwidth [Mly]')
		plt.ylabel('Likelyhood')
		plt.legend(loc = 'best')
	
		plt.title('Great wall KDE bandwidth likelyhood')
		plt.savefig('GreatWall-KDE-bandwidth-likelyhood.svg', bbox_inches = 'tight')
		plt.show()
		
		
		#show the KDE with the highest likelyhood badwidth
		kde = KernelDensity(bandwidth = bestbandwidth, kernel = 'exponential').fit(X)
		# Evaluate the KDE on the grid
		log_dens = kde.score_samples(Xgrid)
		bestdens = X.shape[0] * np.exp(log_dens).reshape((Ny, Nx))
		
		plt.figure()
		plt.imshow(bestdens, interpolation = 'nearest', cmap = plt.get_cmap('hot'))
		plt.title('Great Wall KDE with bandwidth = {0}'.format(round(bestbandwidth, 2)))
		plt.xlabel('Width [Mly]')
		plt.ylabel('Height [Mly]')
		plt.savefig('GreatWall-KDE-best-bandwidth.svg', bbox_inches = 'tight')
		plt.show()
		
	Qb()
	

def main():
	'''
	applyKDE(Table().read('pulsar_masses.vot'), 'Mass', 
						[0.1807692], 
						'Neutron star mass density distribution', 
						xlabel = 'Neutron star mass [M$_{\odot}$]',
						savename = 'Neutronstar-kde-best-bandwidth.svg')
	'''
	
	question4()
	
if __name__ == "__main__":
	main()
