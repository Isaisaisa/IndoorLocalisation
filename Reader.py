import numpy as np
import csv
# Function to read CSV data at the 'input' path
# Returns data object (numpy array) with the following structure
# (dependend on the structure in the CSV file
#  return data:
# 0     1       2       3       4       5       6
# time  gFx     gFy     gFz     wx      wy      wz
# ...   ...     ...     ...     ...     ...     ...
def readCSVData(csvPath):
    data = np.zeros((1,7))
    with open(csvPath, "rt") as csvfile:
        dataReader = csv.reader(csvfile, delimiter=',')
        next(dataReader)
        for row in dataReader:
            rowData = np.zeros((1,7))
            rowData[0,0] = row[0]
            rowData[0,1] = row[1]
            rowData[0,2] = row[2]
            rowData[0,3] = row[3]
            rowData[0,4] = row[4]
            rowData[0,5] = row[5]
            rowData[0,6] = row[6]
            data = np.append(data, rowData, axis = 0)
            #data.append(rowData)
    return data
