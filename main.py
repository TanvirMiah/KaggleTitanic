# data analysis 
import numpy as np
import pandas as pd

# data visualisation
from matplotlib import pyplot as plt
import seaborn as sns

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
# See what columns have null values
train_null_counts = train.isnull().sum()
train_null_counts[train_null_counts > 0].sort_values(ascending = False)

test_null_counts = test.isnull().sum()
test_null_counts[test_null_counts > 0].sort_values(ascending = False)

# look at the data types
train.info()
print('-', 40)
test.info()

# understand the data distribution for Numerical values
train_num_dist = train.describe()
test_num_dist = test.describe()

# understand the data distribution for Categorial values
train_cat_dist = test.describe(include=['O'])
test_cat_dist = test.describe(include=['O'])

'''
Assumptions from initial analysis
'''

'''
Correlating - Identify early on what features will contribute to this

Completing - What features do we need to fill the gaps? 
1. Age, as it is correlated to survival
2. Embarked as it might contribute to the survival

Correcting - What features can we drop? 
1. Ticketing as there are a lot of duplicates
2. Cabin as there are a lot of incomplete values
3. PassengerID as that doesn't tell us anything
4. Name as it's not standard and doesn't seem to contribute to anything

Creating - Do we want to create any new features? 
1. A new feature called family, baseed of Parch and SibSp
2. Extract the titles from the name 
3. Put Age into age bands
4. Fare range as that might be an indicator if someone survives

Classifying - In addition to the stated assumptions above, from the problem statement, what else can we say? 
1. Women were more likely to have survived
2. Children more likely to have survived
3. The upper-class more likely to have survived
'''

'''
Analyse data by pivoting tables
'''

# Passenger Class pivot
pclass_pivot = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Gender pivot
gender_pivot = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Siblings pivot
sib_pivot = train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Parents pivot
parch_pivot = train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

'''
Data Visualisation
'''

# plot histogram of age to identify age buckets
age_hist = sns.FacetGrid(train, col='Survived')
age_hist.map(plt.hist, 'Age', bins=20)

# talk about observations made

# plot Pclass and age against survival
Pclass_age_hist = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
Pclass_age_hist.map(plt.hist, 'Age', alpha=.8, bins=20)
Pclass_age_hist.add_legend()

# talk about observations made

# plot Pclass, gender and embarked against survival
age_gender_hist = sns.FacetGrid(train, col = 'Embarked', size = 2.2, aspect = 1.6)
age_gender_hist.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
age_gender_hist.add_legend()

# Observations made:
# Women who embarked in S or Q generally survived regardless of PClass
# Men who embarked on C had higher survival rate than women regardless of class
# Generally Pclass 3 had a lower survival rate
# Where a person embarked has a high influence of whether they survived