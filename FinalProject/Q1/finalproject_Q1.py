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
	#check if the database exists
	if len(glob.glob(db_name)) > 0:
		return
		
	print('Creating new database with name "{0}"...'.format(db_name))
		
	con = lite.connect(db_name)
	cur = con.cursor()

	#create the table for the positional data
	createmagtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT, 
	FieldID INT,
	Ra FLOAT, 
	Dec FLOAT, 
	PixelX FLOAT, 
	PixelY FLOAT)""".format('Positions')
	cur.execute(createmagtable) #run command
	
	#create the table for the field info like date, 
	createmagtable = """CREATE TABLE IF NOT EXISTS {0} 
	(FieldID INT, 
	MJD DOUBLE,
	Airmass FLOAT,
	Exptime INT)""".format('FieldInfo')
	cur.execute(createmagtable) #run command
	
	#create the table for the fluxes and the magnitudes
	createmagtable = """CREATE TABLE IF NOT EXISTS {0} 
	(StarID INT,
	FieldID INT, 
	Flux1 FLOAT,
	dFlux1 FLOAT,
	Flux2 FLOAT,
	dFlux2 FLOAT,
	Flux3 FLOAT,
	dFlux3 FLOAT,
	Filter varchar(5),
	Class INT
	)""".format('FluxData')
	cur.execute(createmagtable) #run command
	
createDB()
'''
#load the csv
csv_data = Table.read('Tables/file_info_for_problem.csv')
print(csv_data.keys())

#make a list of the filenames of the fits files
filelist = makeFileList('Tables/', '.fits')

#read a single fits file
data = Table.read('Tables/Field-1-Ks-E001.fits')

print(data.keys())
'''
