import sqlite3 as lite
import numpy as np

con = lite.connect("stars_mags.db")
cur = con.cursor()

#create the two tables
createmagtable = """CREATE TABLE IF NOT EXISTS {0} (Name varchar(10), Ra varchar(10), Dec varchar(10), B DOUBLE, R DOUBLE)""".format('MagTable')
cur.execute(createmagtable) #run command

createphystable = """CREATE TABLE IF NOT EXISTS {0} (Name varchar(10), Teff DOUBLE, FeH DOUBLE)""".format('PhysTable')
cur.execute(createphystable) 

#the data to be put in the database
MagTable_name = ['VO-001', 'VO-002', 'VO-003', 'VO-004']
Ra = ['12:34:04.2', '12:15:00.0', '11:55:43.1', '11:32:42.1']
Dec = ['-00:00:23.4', '-14:23:15', '-02:34:17.2', '-00:01:17.3']
B = [15.4, 15.9, 17.2, 16.5]
R = [13.5, 13.6, 16.8, 14.3]

PhysTable_name = ['VO-001', 'VO-002', 'VO-003']
Teff = [4501, 5321, 6600] #in Kelvin
FeH = [0.13, -0.53, -0.32]

#put the data into MagTable
for i in np.arange(len(MagTable_name)):
	command = "INSERT INTO MagTable VALUES('{0}','{1}','{2}',{3},{4})".format(MagTable_name[i], Ra[i], Dec[i], B[i], R[i])
	#print(command)
	cur.execute(command)
	
#put the data into PhysTable
for i in np.arange(len(PhysTable_name)):
	command = "INSERT INTO PhysTable VALUES('{0}',{1},{2})".format(PhysTable_name[i], Teff[i], FeH[i])
	#print(command)
	cur.execute(command)


#run the query from (a)
print("------------ (a) ------------")
rows = con.execute('SELECT Ra, Dec FROM MagTable WHERE B > 16')
for row in rows:
	print(row)
	
#run the query from (b)
print("------------ (b) ------------")
rows = con.execute('SELECT M.Name, M.Ra, M.Dec, M.B, M.R, P.Teff, P.FeH FROM MagTable M INNER JOIN PhysTable P ON P.Name = M.Name')
for row in rows:
	print(row)

#run the query from (c)
print("------------ (c) ------------")
rows = con.execute('SELECT M.Name, M.Ra, M.Dec, M.B, M.R, P.Teff, P.FeH FROM MagTable M INNER JOIN PhysTable P ON P.Name = M.Name WHERE P.FeH > 0')
for row in rows:
	print(row)
	
#do assignment d
print("------------ (d) ------------")
#create the table
createphystable = """CREATE TABLE IF NOT EXISTS {0} (Name varchar(10), BminusR DOUBLE)""".format('ColourTable')
cur.execute(createphystable) 

#put the data into the table
for i in np.arange(len(MagTable_name)):
	command = "INSERT INTO ColourTable VALUES('{0}',{1})".format(MagTable_name[i], B[i] - R[i])
	#print(command)
	cur.execute(command)

rows = con.execute('SELECT * FROM ColourTable')
for row in rows:
	print(row)
