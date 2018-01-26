#for making a file list
import glob
import numpy as np
from astropy.table import Table
import sqlite3 as lite
import pandas as pd
#for checking if something is nan
import math

#name of the database
db_name = "survey_database.db"

def makeFileList(mypath, extension):
	"""
	Makes list of files with certain file extension
	
	Input: mypath (str), extension (str): for example .fits
	
	Returns: str array (location of files)
	"""
	objectfiles = glob.glob(mypath + '*' + extension)
	objectfiles = np.sort(objectfiles)
	
	return objectfiles

def createDB():
	"""
	Create the database with the required columns
	"""
	
	#the keys of the three tables of the database
	#note that ID = identifier for every image (= catalogue)
	posdata_keys = np.array(['StarID','FieldID','Ra','Dec','X','Y'])
	fieldinfo_keys = np.array(['ID','FieldID', 'Filename', 'MJD','Airmass','Exptime','Filter'])
	fluxdata_keys = np.array(['StarID','ID','Flux1','dFlux1','Flux2', 'dFlux2','Flux3','dFlux3','Mag1','dMag1','Mag2','dMag2','Mag3','dMag3','Class'])
	#the names of the tables
	tablenames = np.array(['PosData','FieldInfo','FluxData'])
	
	#check if the database exists
	if len(glob.glob(db_name)) > 0:
		return {tablenames[0]:posdata_keys, tablenames[1]:fieldinfo_keys, tablenames[2]:fluxdata_keys}, tablenames
		
	print('Creating new database with name "{0}"...'.format(db_name))
		
	con = lite.connect(db_name)
	cur = con.cursor()

	#create the table for the positional data
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT, 
	FieldID INT,
	Ra FLOAT, 
	Dec FLOAT, 
	X FLOAT, 
	Y FLOAT)""".format(tablenames[0])
	cur.execute(createtable) #run command
	
	
	#create the table for the field info like date and exposure time
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(ID INT,
	FieldID INT, 
	Filename varchar(255),
	MJD DOUBLE,
	Airmass FLOAT,
	Exptime INT,
	Filter varchar(5))""".format(tablenames[1])
	cur.execute(createtable) #run command
	
	#create the table for the fluxes and the magnitudes
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT,
	ID INT, 
	Flux1 FLOAT,
	dFlux1 FLOAT,
	Flux2 FLOAT,
	dFlux2 FLOAT,
	Flux3 FLOAT,
	dFlux3 FLOAT,
	Mag1 FLOAT,
	dMag1 FLOAT,
	Mag2 FLOAT,
	dMag2 FLOAT,
	Mag3 FLOAT,
	dMag3 FLOAT,
	Class INT
	)""".format(tablenames[2])
	cur.execute(createtable) #run command
	
	con.close()
	
	return {tablenames[0]:posdata_keys, tablenames[1]:fieldinfo_keys, tablenames[2]:fluxdata_keys}, tablenames
	
def fillTable(dbn, data, keys, tablename):
	"""
	Fill a table in a database
	"""
	
	#open the database
	con = lite.connect(dbn)
	with con:
		cur = con.cursor()

		#determine the number of keys
		nkeys = len(keys)

		#fill the table
		for i in np.arange(len(data[keys[0]])):
			insertcommand = "INSERT INTO {0} VALUES(".format(tablename)
			
			#add all the values from the data array row
			for key, j in zip(keys, np.arange(nkeys)):
				#check if there is a nan in the row. If so, we have to adjust the 
				#insert command
				if type(data[key][i]) != np.str_ and math.isnan(data[key][i]):
					insertcommand += "NULL"
				#check if we have to add quotations around the data
				elif type(data[key][i]) == np.str_:
					insertcommand += "'" + data[key][i] + "'"
				else:
					insertcommand += str(data[key][i])
				
				#add commas to seperate the different values
				if j < (nkeys - 1):
					insertcommand += ","
				
			#close the command with a bracket
			insertcommand += ")"
			
			cur.execute(insertcommand)
			
def fillDataBase(dataloc = 'Tables/'):
	"""
	Fill the database with data from fits tables in the folder 'Tables/'
	"""
		
	allkeys, tablenames = createDB()

	#load the csv
	csv_data = Table.read('Tables/file_info_for_problem.csv')
	#fill the 'FieldInfo' table with the data from the csv
	fillTable(db_name, csv_data, allkeys['FieldInfo'], 'FieldInfo')

	#make a list of the filenames of the fits files
	filelist = makeFileList(dataloc, '.fits')
	flistlength = len(filelist)
	#loop over all the found filenames
	for fname, i in zip(filelist, np.arange(flistlength)):
		print('Loading file {0}/{1}...'.format(i+1, flistlength))

		tabledata = Table.read(fname)

		#find the field ID
		fID = int(fname[len(dataloc):].split('-')[1])

		#find the filter
		filt = fname[len(dataloc):].split('-')[2].split('.')[0]

		#find the image ID using the fieldID and filter
		iID = csv_data['ID'][(csv_data['Filter'] == filt) * (csv_data['FieldID'] == fID)][0]

		#add the image ID to the table
		tabledata['ID'] = np.tile([iID], len(tabledata[allkeys['PosData'][0]]))
		#add Field ID to the table
		tabledata['FieldID'] = np.tile([fID], len(tabledata[allkeys['PosData'][0]]))

		#insert the data in the PosData and FluxData tables
		fillTable(db_name, tabledata, allkeys['PosData'], 'PosData')
		fillTable(db_name, tabledata, allkeys['FluxData'], 'FluxData')

def makeKDE(data, names, query_id):
	"""
	Make a KDE plot of the different columns in the SQL data
	"""
	import matplotlib.pyplot as plt
	import seaborn as sns

	sns.set(font = 'Latin Modern Roman',rc = {'legend.frameon': True})

	#the bandwidth to be used
	bandwidth = 0.2

	#rename the columns of the dataframe
	data.columns = names
	#remove all none or nan values
	data = data.dropna()

	# plt.hist(np.array(data_dict['J'])[0])
	for n in names:
		sns.kdeplot(np.array(data[n]), bw = bandwidth, label = n)

	plt.legend(loc = 'best', title = 'Filter', shadow = True)
	plt.title('KDE plot of database filters with bandwidth = {0}'.format(bandwidth))
	plt.xlabel('Magnitude')
	plt.ylabel('Probability')
	plt.savefig('{0}_visualization.pdf'.format(query_id), dpi = 300)
	plt.show()

def makeHistogram(data, names, query_id):
	"""
	Make a histogram of the given SQL output rows
	"""
	import matplotlib.pyplot as plt
	import seaborn as sns

	sns.set(font = 'Latin Modern Roman',rc = {'legend.frameon': True})

	#rename the columns of the dataframe
	data.columns = names
	#remove all none or nan values
	data = data.dropna()

	sns.distplot(data[names[1]], kde = False, color = 'g')
	# plt.hist(data[names[1]])

	# plt.legend(loc = 'best', title = 'Filter', shadow = True)
	if query_id == 'R2':
		plt.title('Histogram of J - H')
		plt.xlim((1.4, 2.25))
	# plt.yscale('log')
		plt.xlabel('J - H')
		plt.ylabel('Number of stars')
	elif query_id == 'R3':
		plt.xlim((-6000, 250000))
		plt.title('Histogram of Ks flux')
		plt.xlabel('Flux')
		plt.ylabel('Number of stars')
	plt.savefig('{0}_visualization.pdf'.format(query_id), dpi = 300)
	plt.show()

def saveResults(data, query_id):
	"""
	Save the results of a query to a csv file.

	Input:
		data (pandas dataframe): a dataframe containing the data to be saved.\n
		query_id (str): the name of the query.
	"""
	data.to_csv('./Query_results/{0}_results.csv'.format(query_id), index = False)

def R1():
	"""
	Test query R1
	"""
	print('\nQuery 1:')
	#open the database
	con = lite.connect(db_name)
	with con:
		
		query1 = """
				SELECT i.ID, COUNT(f.StarID)
				FROM FluxData f JOIN FieldInfo i 
				ON f.ID = i.ID
				WHERE f.Flux1/f.dFlux1 > 5 AND f.Class = -1
				GROUP BY i.ID HAVING MJD BETWEEN 56800 AND 57300
				"""

		#run the query
		data = pd.read_sql(query1, con)

		print(data)

		#save the data to a csv file
		saveResults(data, 'R1')

def R2():
	"""
	Test query R2
	"""
	#open the database
	con = lite.connect(db_name)
	with con:

		query1 = """
				SELECT j.StarID, j.Mag1 - h.Mag1
				FROM (SELECT f.Mag1, f.StarID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'J') j
				JOIN (SELECT f.Mag1, f.StarID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'H') h
				ON j.StarID = h.StarID
				WHERE j.Mag1 - h.Mag1 > 1.5
				GROUP BY j.StarID
				"""
		
		#run the query
		data = pd.read_sql(query1, con)

		makeHistogram(data, ['StarID', 'J - H'], 'R2')

def R3():
	"""
	Test query R3
	"""
	print('Warning: this query might take very long')
	#open the database
	con = lite.connect(db_name)
	with con:

		query1 = """
				SELECT f.StarID, f.Flux1
				FROM FluxData f 
				JOIN FieldInfo i ON f.ID = i.ID
				WHERE i.Filter = 'Ks' AND ABS(
					f.Flux1 - (
					SELECT AVG(f2.Flux1)
					FROM FluxData f2
					JOIN FieldInfo i2 ON f2.ID = i2.ID
					WHERE f.StarID = f2.StarID AND i2.Filter = 'Ks'
					)) > (20 * f.dFlux1)
				GROUP BY f.StarID
				"""

		#run the query
		data = pd.read_sql(query1, con)

		#save the data to a csv file
		saveResults(data, 'R3')

		#it is also possible to load the results from a file, as the query takes
		#very long to run
		# data = pd.read_csv('./Query_results/R3_results.csv')
		
		makeHistogram(data, ['StarID', 'Ks flux'], 'R3')

def R4(fieldid = 1):
	"""
	Test query R4. Catalogue = image
	"""
	#open the database
	con = lite.connect(db_name)
	with con:

		query1 = """
				SELECT ID
				FROM FieldInfo
				WHERE FieldID = {0}
				""".format(fieldid)

		
		#run the query
		data = pd.read_sql(query1, con)

		print(data)

		#save the data to a csv file
		saveResults(data, 'R4')

def R5(fieldid = 1):
	"""
	Test query R5. The Ks magnitudes are averaged per field.

	Input:
		fieldid (int): ID of the field of which the Y, Z, J, H and Ks magnitudes
		should be obtained.
	"""
	#names of the filters
	names = ['Y', 'Z', 'J', 'H', 'Ks']

	#open the database
	con = lite.connect(db_name)
	with con:

		query1 = """
				SELECT Y.Mag1, Z.Mag1, J.Mag1, H.Mag1, Ks.AvgMag1
				FROM (
					SELECT f.Mag1, f.StarID, i.FieldID, f.Class
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'Y' AND f.Flux1/f.dFlux1 > 30) Y
				LEFT JOIN (
					SELECT f.Mag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'Z' AND f.Flux1/f.dFlux1 > 30
				) Z On Y.StarID = Z.StarID AND Y.FieldID = Z.FieldID
				LEFT JOIN (
					SELECT f.Mag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'J' AND f.Flux1/f.dFlux1 > 30
				) J On Y.StarID = J.StarID AND Y.FieldID = J.FieldID
				LEFT JOIN (
					SELECT f.Mag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'H' AND f.Flux1/f.dFlux1 > 30
				) H On Y.StarID = H.StarID AND Y.FieldID = H.FieldID
				LEFT JOIN (
					SELECT AVG(f.Mag1) AS AvgMag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'Ks' AND f.Flux1/f.dFlux1 > 30
					GROUP BY f.StarID, i.ID
				) Ks On Y.StarID = Ks.StarID AND Y.FieldID = Ks.FieldID
				WHERE Y.FieldID = {0} AND Y.Class = -1
				""".format(fieldid)

		#run the query
		data = pd.read_sql(query1, con)

		makeKDE(data, names, 'R5')

def loadYJHdata(SN = 10):
	"""
	Load the Y - J and J - H colours of all the stars in the database with
	SN > 8.

	Input:
		SN (int): signal to noise threshold. Default = 10.

	Output:
		data (pandas dataframe): the data obtained from the query
	"""
	#names of the filters
	names = ['Y - J', 'J - H']

	#open the database
	con = lite.connect(db_name)
	with con:

		query1 = """
				SELECT Y.Mag1 - J.Mag1, J.Mag1 - H.Mag1
				FROM (
					SELECT f.Mag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'Y' AND f.Flux1/f.dFlux1 > {0}) Y
				LEFT JOIN (
					SELECT f.Mag1, f.StarID, i.FieldID, f.Class
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'J' AND f.Flux1/f.dFlux1 > {0}
				) J On Y.StarID = J.StarID AND Y.FieldID = J.FieldID
				LEFT JOIN (
					SELECT f.Mag1, f.StarID, i.FieldID
					FROM FluxData f 
					JOIN FieldInfo i ON f.ID = i.ID
					WHERE i.Filter = 'H' AND f.Flux1/f.dFlux1 > {0}
				) H On Y.StarID = H.StarID AND Y.FieldID = H.FieldID AND J.Class = -1
				""".format(SN)

		#run the query
		data = pd.read_sql(query1, con)

		#rename the columns of the dataframe
		data.columns = names
		#remove all none or nan values
		data = data.dropna()

		print('Number of data points: {0}'.format(len(data[names[0]])))

		return data

def make2D_KDE(X, n_samp = 1e5, bandwidth = None, n_folds = 3, bw_train_size = 1000, bw_range_size = 20, doplot = True):
	"""
	Make a 2D Kernel Density Estimation and draw a n_samp number of samples from it
	best bandwidth obtained from previous runs
	bandwidth = 0.0546938775510204
	bandwidth = 0.05894736842105264

	Input:
		X (2D numpy array): the training data, consisting of the Y - J and J - H 
		colours.\n
		n_samp (int): the number of samples to draw from the KDE. Default = 100000.\n
		bandwidth (float): the bandwidth to use for the KDE from which the samples
		will be drawn. Set to None to let the script find the best bandwidth. 
		Default = None.\n
		n_folds (int): the number of folds to use when determining the bandwidth.\n
		bw_train_size (int): size of the training set that will be used to 
		determine the best bandwidth. Default = 1000.\n
		bw_range_size (int); the amount of bandwidths to try out in the interval
		0.04 to 0.1. Default = 20.\n
		doplot (boolean): whether to make a hex-bin plot of the drawn samples or
		not. Default = True.

	Output:
		samples (2D numpy array): the samples drawn from the KDE.
	"""
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.neighbors import KernelDensity
	from sklearn.model_selection import KFold
	from matplotlib import rcParams
	rcParams['font.family'] = 'Latin Modern Roman'
	from matplotlib.colors import LogNorm

	#shuffle the data
	np.random.shuffle(X)

	#determine the best bandwidth if it is not provided
	if bandwidth == None:
		#first we find the optimum bandwidth
		kf = KFold(n_splits = n_folds)

		#range of bandwidths to try
		bwrange = np.linspace(0.02, 0.08, bw_range_size)
		#the array which will store the likelyhood
		likelyhood = np.zeros(len(bwrange))
		
		print('Finding the best bandwidth...')
		for bw, i in zip(bwrange, np.arange(len(bwrange))):
			print('Iteration {0}, bandwidth {1}'.format(i, bw))
			lh = []
			#split the data into a train and test set using only the first 1000 samples
			for train_i, test_i in kf.split(X[:,:bw_train_size]):
				Xtrain, Xtest = X[train_i], X[test_i]
				kde = KernelDensity(bandwidth = bw, kernel = 'gaussian').fit(Xtrain)

				lhscore = kde.score(Xtest)
				
				lh = np.append(lh, lhscore)

				print('Bandwidth: {0}, score: {1}'.format(bw, lhscore))
				
			likelyhood[i] = np.mean(lh)

		plt.plot(bwrange, likelyhood)
		plt.xlabel('Bandwidth')
		plt.ylabel('Likelyhood')
		plt.title('KDE likelyhood for different bandwidths')
		plt.savefig('2D_KDE_likelyhood_run4.png', dpi = 300)
		plt.close()


		#find the bandwidth which gave the highest likelyhood
		bandwidth = bwrange[np.argmax(likelyhood)]

		print('Best bandwidth: {0}'.format(bandwidth))

	kde = KernelDensity(bandwidth = bandwidth, kernel = 'gaussian').fit(X)

	#pull samples from the kde
	samples = kde.sample(int(n_samp))
	
	#plot the samples in a hexbin plot
	if doplot:
		plt.hexbin(samples[:, 0], samples[:, 1], bins = 'log', cmap = 'Reds')
		plt.colorbar(label = 'Density of samples [logarithmic]')

		plt.xlabel('Y - J')
		plt.ylabel('J - H')
		plt.title('Distribution of samples in (Y-J, J-H) colour space')
		plt.savefig('Samples_distribution_hex.pdf', dpi = 300)
		plt.show()

	return samples
	

createDB()
fillDataBase()
R1()


#load the data as a pandas dataframe
# df = loadYJHdata()

#input the data as a numpy array and receive the 100000 samples
# samples = make2D_KDE(np.array([df['Y - J'], df['J - H']]).T, bandwidth = 0.0389474)
