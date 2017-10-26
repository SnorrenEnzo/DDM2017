from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
sns.set()


def question1():
	t = Table().read('joint-bh-mass-table.csv')
	
	X_plot = np.linspace(np.min(t['MBH']) - 4, np.max(t['MBH']) + 4, num = 1000)[:, np.newaxis]
	X = t['MBH'][:, np.newaxis]
	
	print(X)
	
	plt.scatter(X[:,0], np.zeros(len(X[:,0])), marker = 'x', color = 'black')
	
	#different values for the bandwidth
	bwrange = np.arange(1, 8)
	
	#plot many lines using colors from a color map, with MAX the amount of lines
	jet = plt.get_cmap('jet') 
	cNorm  = colors.Normalize(vmin=0, vmax=len(bwrange))
	scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	
	for bw, i in zip(bwrange, np.arange(len(bwrange))):
		kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(X)
	
		log_dens = kde.score_samples(X_plot)
		
		#for plotting in many different colours
		colorVal = scalarMap.to_rgba(i)	
	
		plt.plot(X_plot, np.exp(log_dens), label = 'Bandwidth: {0}'.format(bw), color = colorVal)
	
	
	plt.xlabel('Black hole mass [M$_{sun}$]')
	plt.ylabel('Probability')
	plt.legend(loc = 'best')
	
	plt.title('Black hole mass density distribution')
	plt.savefig('Blackhole-kde-bandwidth.svg')
	plt.show()
	
	"""
	The best bandwidth seems to be 6 Msun. This is not necessarily the best estimate.
	"""

def main():

	question1()
	
if __name__ == "__main__":
	main()
