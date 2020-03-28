# data analysis 
import numpy as np
import pandas as pd

# data visualisation
from matplotlib import pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# machine learning and analysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

'''
**********DATA ANALYSIS****************
'''

'''
Import the data
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
fullset = [train, test]

'''
Remove the data points we don't need, and add custom data points
'''

# drop data
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)
fullset = [train, test]

# create new features

# check if title adds to the probability of survival
for dataset in fullset:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# replace uncommon endings with more common ones
for dataset in fullset:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# now that the titles are extracted and put into buckets, we can 
# 1. turn them into numerical ordinals
# 2. drop the name columns
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in fullset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# drop the name and passenger ID
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name', 'PassengerId'], axis=1)
fullset = [train, test]

'''
Convert strings into numerical values
'''

# convert the genders into 0 and 1
for dataset in fullset:
    dataset['Sex'] = dataset['Sex'].map( {'female' : 1, 'male': 0}).astype(int)
    

'''
#get onehotencoder
onehotencoder = OneHotEncoder()
#extract the column that needs to be onehotencoded
train_test = train['Sex'].map( {'female' : 1, 'male' : 0}).astype(int)
#turn into an array and reshape so that it can be onehotencoded
train_test = np.array(train_test).reshape(-1 , 1)
#fit and transform the array
train_test = onehotencoder.fit_transform(train_test).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = train_test)
#create a dummy train dataframe to keep the real test dataframe intact
train_onehotencode_test = train
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
train_onehotencode_test = train_onehotencode_test.merge(onehote_df, on = train_onehotencode_test.index)
#drop columns
train_onehotencode_test = train_onehotencode_test.drop(['key_0'], axis=1)
'''

# check everything is ok
train.head(10)

# write observations made

# using Pclass and Gender, we can guess ages
# create an empty array to guess ages
guess_ages = np.zeros((2,3))

# run a for loop that calculates the median age of the average male and female in 
# Pclase 1, 2 and 3, and then replace the missing ages
for dataset in fullset:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# check everything worked
train.head(10)

# put ages into discrete bins, using pandas cut feature
age_bands = train['AgeBand'] = pd.cut(train['Age'], 5)
age_bands = train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# now that we have the age bands, lets turn them into ordinals
for dataset in fullset:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

# check everything is ok
train.head(10)

# the AgeBand feature is not needed anymore
train = train.drop(['AgeBand'], axis=1)
fullset = [train, test]
train.head(10)

'''
Combine the Parch and Siblings data to a single datapoint
'''
for dataset in fullset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# add an additonal feature to see if being alone makes a difference
# removing the solo passengers from being included in the 'Larger than four' bin to remove the bias of low survival from being a solo passenger
for dataset in fullset:
    dataset['LargerThanFour'] = 0
    dataset.loc[ dataset['FamilySize'] == 1, 'LargerThanFour'] = 0
    dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'LargerThanFour'] = 1
    dataset.loc[ dataset['FamilySize'] > 4, 'LargerThanFour'] = 2
    dataset['LargerThanFour'] = dataset['LargerThanFour'].astype(int)
    

# parch and sibsp and family size can be dropped as they are not needed, as IsAlone covers it
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
fullset = [train, test]

# check everything is ok
train.head(10)

'''
Turn the embarked feature into ordinal numbers
'''

# two NA's in the embarked training data
# find the most common Embarked place and replace the two NA's with that 
freq_port = train.Embarked.dropna().mode()[0]

for dataset in fullset:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

embarked_freq = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# convert Embarked into ordinal numbers
for dataset in fullset:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
 
train.head(10)

'''
Convert the fares into buckets and fill NA values
'''
# this doesn't seem to work for whatever reason
median_fare = test.Fare.dropna().median()
'''
for dataset in test:
    dataset['Fare'] = dataset['Fare'].fillna(1)
    '''
test['Fare'] = test['Fare'].fillna(14.4542)

# check if everything is ok 
test.head(10)

# create FareBand using pandas
train['FareBand'] = pd.qcut(train['Fare'], 4)
fare_bands = train[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by='FareBand', ascending=True)

# turn bands into ordinals
for dataset in fullset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
# drop the FareBand column
train = train.drop(['FareBand'], axis = 1)
fullset = [train, test]

# check everything is ok 
train.head(10)
test.head(10)

'''
*********************OneHotEncoding*****************************
'''

#OneHotEncode Sex

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
train_test = train['Sex']
#turn into an array and reshape so that it can be onehotencoded
train_test = np.array(train_test).reshape(-1 , 1)
#fit and transform the array
train_test = onehotencoder.fit_transform(train_test).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = train_test)
#create a dummy train dataframe to keep the real test dataframe intact
train_dummy = train
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
train_dummy = train_dummy.merge(onehote_df, on = train_dummy.index)
#drop columns
train_dummy = train_dummy.drop(['Sex', 'key_0'], axis=1)
train = train_dummy

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
test_t = test['Sex']
#turn into an array and reshape so that it can be onehotencoded
test_t = np.array(test_t).reshape(-1 , 1)
#fit and transform the array
test_t = onehotencoder.fit_transform(test_t).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = test_t)
#create a dummy train dataframe to keep the real test dataframe intact
test_dummy = test
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
test_dummy = test_dummy.merge(onehote_df, on = test_dummy.index)
#drop columns
test_dummy = test_dummy.drop(['Sex', 'key_0'], axis=1)
test = test_dummy

#OneHotEncode Title

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
train_test = train['Title']
#turn into an array and reshape so that it can be onehotencoded
train_test = np.array(train_test).reshape(-1 , 1)
#fit and transform the array
train_test = onehotencoder.fit_transform(train_test).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = train_test)
#create a dummy train dataframe to keep the real test dataframe intact
train_dummy = train
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
train_dummy = train_dummy.merge(onehote_df, on = train_dummy.index)
#drop columns
train_dummy = train_dummy.drop(['Title', 'key_0'], axis=1)
train = train_dummy

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
test_t = test['Title']
#turn into an array and reshape so that it can be onehotencoded
test_t = np.array(test_t).reshape(-1 , 1)
#fit and transform the array
test_t = onehotencoder.fit_transform(test_t).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = test_t)
#create a dummy train dataframe to keep the real test dataframe intact
test_dummy = test
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
test_dummy = test_dummy.merge(onehote_df, on = test_dummy.index)
#drop columns
test_dummy = test_dummy.drop(['Title', 'key_0'], axis=1)
test = test_dummy

#OneHotEncode Embarked

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
train_test = train['Embarked']
#turn into an array and reshape so that it can be onehotencoded
train_test = np.array(train_test).reshape(-1 , 1)
#fit and transform the array
train_test = onehotencoder.fit_transform(train_test).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = train_test)
#create a dummy train dataframe to keep the real test dataframe intact
train_dummy = train
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
train_dummy = train_dummy.merge(onehote_df, on = train_dummy.index)
#drop columns
train_dummy = train_dummy.drop(['Embarked', 'key_0'], axis=1)
train = train_dummy

onehotencoder = OneHotEncoder(categories = 'auto')
#extract the column that needs to be onehotencoded
test_t = test['Embarked']
#turn into an array and reshape so that it can be onehotencoded
test_t = np.array(test_t).reshape(-1 , 1)
#fit and transform the array
test_t = onehotencoder.fit_transform(test_t).toarray()
#turn the array into a dataframe
onehote_df = pd.DataFrame(data = test_t)
#create a dummy train dataframe to keep the real test dataframe intact
test_dummy = test
#merge the dummy train dataframe with the new dataframe of the onehotencoded data
test_dummy = test_dummy.merge(onehote_df, on = test_dummy.index)
#drop columns
test_dummy = test_dummy.drop(['Embarked', 'key_0'], axis=1)
test = test_dummy

'''
********************Standard Scalar************************************
'''

sc_X = StandardScaler()
train_dummy = train
train_dummy = sc_X.fit_transform(train_dummy)
train = pd.DataFrame(data = train_dummy)

test_dummy = test
test_dummy = sc_X.fit_transform(test_dummy)
test = pd.DataFrame(data = test_dummy)

lab_enc = LabelEncoder()
train = lab_enc.fit_transform(train)

'''
****************MODELING AND PREDICITNG*************
'''

'''
Need to decide the most suitable model to predict.
The output of this model is whether someone will survive or not, therefore it is 
a clasification model. As a result, the classification algorithms I'll be trying
are as follows:
1. Logistic Regression
2. K-Nearest Neighbours
3. Support Vector Machine
4. Naive Bayes
5. Decision Tree Classification
6. Random Forest Classification
'''

# create dataframe that logs accuracy of the different models
model_accuracy = pd.DataFrame(columns=['Model Name', 'Accuracy (No K-Fold)', 'Accuracy (K-Fold)'])
# hardcoding value as I know there are 418 tests in the main test file
total_tests = 179

# Split the data up into train X and Y
X = train.drop(train.columns[0], axis = 1)
y = train[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# fill NA with a random bucket for now that is in the 14.4542 
#X_test = test.fillna(1)
#y_actual = sub_pred['Survived'].values

# logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_regpred = logreg.predict(X_test)
regpred_conmat = confusion_matrix(y_test, Y_regpred)
regpred_correct = regpred_conmat[0,0] + regpred_conmat[1,1]
regpred_acc = regpred_correct/total_tests
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = logreg, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'Logistic Regression', 'Accuracy (No K-Fold)': regpred_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)


# svc
svc_mod = SVC()
svc_mod.fit(X_train, y_train)
Y_svcpred = svc_mod.predict(X_test)
svc_conmat = confusion_matrix(y_test, Y_svcpred)
svc_correct = svc_conmat[0,0] + svc_conmat[1,1]
svc_mod_acc = (svc_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = svc_mod, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'SVC', 'Accuracy (No K-Fold)': svc_mod_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# random forest classifier
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
Y_RFC = RFC.predict(X_test)
RFC_conmat = confusion_matrix(y_test, Y_RFC)
RFC_correct = RFC_conmat[0,0] + RFC_conmat[1,1]
RFC_acc = (RFC_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = RFC, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'Random Forest', 'Accuracy (No K-Fold)': RFC_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# K Nearest Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_knn = knn.predict(X_test)
knn_conmat = confusion_matrix(y_test, Y_knn)
knn_correct = knn_conmat[0,0] + knn_conmat[1,1]
knn_acc = (knn_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'K Nearest Neighbors', 'Accuracy (No K-Fold)': knn_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
Y_gnb = gnb.predict(X_test)
gnb_conmat = confusion_matrix(y_test, Y_gnb)
gnb_correct = gnb_conmat[0,0] + gnb_conmat[1,1]
gnb_acc = (gnb_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = gnb, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'Naive Bayes', 'Accuracy (No K-Fold)': gnb_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# Decision tree
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, y_train)
Y_Dtree = Dtree.predict(X_test)
Dtree_conmat = confusion_matrix(y_test, Y_Dtree)
Dtree_correct = Dtree_conmat[0,0] + Dtree_conmat[1,1]
Dtree_acc = (Dtree_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = Dtree, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'Decision Tree Classifier', 'Accuracy (No K-Fold)': Dtree_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
Y_xgb = xgb.predict(X_test)
xgb_conmat = confusion_matrix(y_test, Y_xgb)
xgb_correct = xgb_conmat[0,0] + xgb_conmat[1,1]
xgb_acc = (xgb_correct/total_tests)
# apply k-fold cross validation and get mean accuracy
accuracies = cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
model_accuracy = model_accuracy.append({'Model Name': 'XG Boost', 'Accuracy (No K-Fold)': xgb_acc, 'Accuracy (K-Fold)':  accuracies.mean()}, ignore_index = True)

# prediction
test = test.fillna(1)
prediction = xgb.predict(test)
test_pass = pd.read_csv('test.csv')
pd.DataFrame(data={"PassengerId": test_pass['PassengerId'], "Survived": prediction.astype(int)}).to_csv("submission.csv", index=False)