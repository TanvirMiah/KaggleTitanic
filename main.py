# data analysis 
import numpy as np
import pandas as pd

'''
Import the data
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fullset = [train, test]

'''
Analyse the data by spotting basic trends 
'''
# first see what the column headers are
print(train.columns.values)

# print the head of the data to see what is categorical and what is numerical
# have a quick look at the data in the top 5 rows
train.head()

# look at the data types
train.info()
print('-', 40)
test.info()

# understand the data distribution 
train_dist = train.describe()