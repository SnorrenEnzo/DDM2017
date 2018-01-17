#for making a file list
import glob
import numpy as np
from astropy.table import Table
import sqlite3 as lite
import pandas
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
	fieldinfo_keys = np.array(['ID','FieldID','MJD','Airmass','Exptime','Filter'])
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
	
	
	#create the table for the field info like date
	createtable = """CREATE TABLE IF NOT EXISTS {0} 
	(ID INT,
	FieldID INT, 
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

def printrows(rows, maxprint = 30):
	"""
	Print the rows in a given sql query output. 

	Input:
		rows (str iterator): the rows obtained from an SQL query.\n
		maxprint (int): the maximum number of rows to print. Default = 30. Set to 
		-1 to print all the rows.
	"""
	i = 0
	for row in rows:
		if maxprint < 1:
			print(row)
		elif i < maxprint:
			print(row)
		i += 1

def R1():
	"""
	Test query R1
	"""
	#open the database
	con = lite.connect(db_name)
	with con:
		cur = con.cursor()
		
		
		query1 = """
				SELECT i.ID, COUNT(f.StarID)
				FROM FluxData f JOIN FieldInfo i 
				ON f.ID = i.ID
				WHERE f.Flux1/f.dFlux1 > 5
				GROUP BY i.ID HAVING MJD BETWEEN 56800 AND 57300
				"""

		#check if the data is inserted properly in the table
		rows = con.execute(query1)
		
		printrows(rows, maxprint = -1)

def R2():
	"""
	Test query R2
	"""
	#open the database
	con = lite.connect(db_name)
	with con:
		cur = con.cursor()

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
		
		#check if the data is inserted properly in the table
		rows = con.execute(query1)
		
		printrows(rows)

def R3():
	"""
	Test query R3
	"""
	#open the database
	con = lite.connect(db_name)
	with con:
		cur = con.cursor()

		query1 = """
				SELECT f.StarID
				FROM FluxData f 
				JOIN FieldInfo i ON f.ID = i.ID
				WHERE i.Filter = 'Ks' AND ABS(
					f.Flux1 - (
					SELECT AVG(f2.Flux1)
					FROM FluxData f2
					JOIN FieldInfo i2 ON f2.ID = i2.ID
					WHERE f.StarID = f2.StarID
					)) > 20 * f.dFlux1
				GROUP BY f.StarID
				"""

		
		#check if the data is inserted properly in the table
		rows = con.execute(query1)
		
		printrows(rows)

def R4():
	"""
	Test query R4. Catalogue = image
	"""
	#open the database
	con = lite.connect(db_name)
	with con:
		cur = con.cursor()

		query1 = """
				SELECT ID
				FROM FieldInfo
				WHERE FieldID = 1
				"""

		
		#check if the data is inserted properly in the table
		rows = con.execute(query1)
		
		printrows(rows)

# createDB()
# fillDataBase()
R4()

