from astropy.table import Table
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
sns.set()
plt.rc('text', usetex=True)


def question1a(t, key, bwrange, title, savename = 'fig.svg', xlabel = 'Xlabel'):
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
	plt.savefig(savename)
	plt.show()
	
	"""
	The best bandwidth seems to be 6 Msun. This is not necessarily the best estimate.
	"""
	
def question1b(t, key, bwrange):
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
	t = Table().read('pulsar_masses.vot')
	
	print(len(t['Mass']))
	
	X_plot = np.linspace(np.min(t['Mass']) - 1, np.max(t['Mass']) + 1, num = 1000)[:, np.newaxis]
	X = t['Mass'][:, np.newaxis]
	
	#plt.scatter(X[:,0], np.zeros(len(X[:,0])), marker = 'x', color = 'black')
	
	#different values for the bandwidth
	bwrange = np.linspace(0.5, 5, num = 40)
	
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
	
	plt.xlabel('Bandwidth [M$_{\odot}$]')
	plt.ylabel('Likelyhood')
	plt.legend(loc = 'best')
	
	plt.title('Black hole mass density distribution')
	#plt.savefig('Neutronstar-kde-bandwidth-likelyhood.svg')
	plt.show()

def main():

	question1a(Table().read('pulsar_masses.vot'), 'Mass', 
						np.linspace(0.05, 1.5, num = 7), 
						'Black hole mass density distribution', 
						savename = 'Neutronstar-kde-bandwidth.svg', 
						xlabel = 'Black hole mass [$10^6$ M$_{\odot}$]')
	
if __name__ == "__main__":
	main()
