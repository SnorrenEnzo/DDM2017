#for making a file list
import glob
import numpy as np
from astropy.table import Table
import sqlite3 as lite
import pandas

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
	posdata_keys = ['StarID','FieldID','Ra','Dec','PixelX','PixelY']
	fieldinfo_keys = ['FieldID','MJD','Airmass','Exptime']
	fluxdata_keys = ['StarID','FieldID','Flux1','dFlux1','Flux2','dFlux2','Flux3',
									'dFlux3','Mag1','dMag1','Mag2','dMag2','Mag3','dMag3','Filter','Class']
	
	#check if the database exists
	if len(glob.glob(db_name)) > 0:
		return posdata_keys, fieldinfo_keys, fluxdata_keys
		
	print('Creating new database with name "{0}"...'.format(db_name))
		
	con = lite.connect(db_name)
	cur = con.cursor()

	#create the table for the positional data
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT, 
	FieldID INT,
	Ra FLOAT, 
	Dec FLOAT, 
	PixelX FLOAT, 
	PixelY FLOAT)""".format('PosData')
	cur.execute(createtable) #run command
	
	
	#create the table for the field info like date, 
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(FieldID INT, 
	MJD DOUBLE,
	Airmass FLOAT,
	Exptime INT)""".format('FieldInfo')
	cur.execute(createtable) #run command
	
	#create the table for the fluxes and the magnitudes
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT,
	FieldID INT, 
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
	Filter varchar(5),
	Class INT
	)""".format('FluxData')
	cur.execute(createtable) #run command
	
	return posdata_keys, fieldinfo_keys, fluxdata_keys
	
def fillTable(db_name, data, keys, tablename):
	"""
	Fill a table in a database
	"""
	
	#open the database
	con = lite.connect(db_name)
	cur = con.cursor()
	
	#determine the number of keys
	nkeys = len(keys)

	#fill the table
	for i in np.arange(len(data[keys[0]])):
		insertcommand = 'INSERT INTO {0} VALUES('.format(tablename)
		#add all the values from the data array row
		for key, j in zip(keys, np.arange(nkeys)):
			insertcommand += str(data[key][i])
			
			if j < (nkeys - 1):
				insertcommand += ','
			
		#close the command with a bracket
		insertcommand += ')'

		cur.execute(insertcommand)
		
	#close the connection
	con.close()
	
posdata_keys, fieldinfo_keys, fluxdata_keys = createDB()

#load the csv
csv_data = Table.read('Tables/file_info_for_problem.csv')
#fill the 'FieldInfo' table with the data from the csv
fillTable(db_name, csv_data, fieldinfo_keys, 'FieldInfo')


#make a list of the filenames of the fits files
filelist = makeFileList('Tables/', '.fits')

#read a single fits file
data = Table.read('Tables/Field-1-Ks-E001.fits')

#print(data.keys())

#open the database
con = lite.connect(db_name)
cur = con.cursor()

#check if the data is inserted properly in the table
rows = con.execute('SELECT * FROM FieldInfo')
for row in rows:
	print(row)

