#for making a file list
import glob


def makeFileList(mypath, extension):
	"""
	Makes list of files with certain file extension
	
	Input: mypath (str), extension (str): for example .fits
	
	Returns: str array (location of files)
	"""
	objectfiles = glob.glob(mypath + '*' + extension)
	objectfiles = np.sort(objectfiles)
	
	return objectfiles

def readFits(filename, index):
	"""
	Load a fits table
	"""
	from astropy.io import fits 

	hlist = fits.open(filename)
	header = hlist[index].header

	#For tables
	data = hlist[1].data
	
	return data