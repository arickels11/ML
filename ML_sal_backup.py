# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:32:29 2022

@author: arick
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import f_classif 
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# import dataset
cols = ['age', 'work', 'weight', 'education', 'edu_num', 'married', 'occupation',
    'relationship', 'race', 'sex', 'cap_gain', 'cap_loss', 'hours_week', 'country', 'class']
sal = pd.read_csv('salary.csv', names=cols)

# plot class labels
sal['class'].value_counts().plot(kind='barh')

# encode label/class value (Y)
mapping = {' <=50K': '0', ' >50K': '1'}
sal_encoded = sal.replace({'class': mapping})
Y = sal_encoded['class']
print(Y.value_counts())
sal_X = sal.drop(columns='class')

# Encoding of categorical data
cat_cols = ['work', 'education', 'married', 'occupation',
    'relationship', 'race', 'sex', 'country']
cont_cols = ['age','weight','edu_num','cap_gain','cap_loss','hours_week']
X = pd.get_dummies(sal_X, columns=cat_cols)
#, drop_first=True - colinearity
X_cat = X.drop(columns=cont_cols)

# Scaling of continuous data
X_cont = pd.DataFrame(sal_X, columns=cont_cols)

#STANDARDIZED
sc = StandardScaler()
XX = sc.fit(X_cont)
XXX = sc.transform(X_cont)
X_cont_std = pd.DataFrame(XXX)

#NORMALIZED
sc = MinMaxScaler()
XX = sc.fit(X_cont)
XXX = sc.transform(X_cont)
X_cont_norm = pd.DataFrame(XXX)


#combine cont and cat columns again

#non standardized (better for DT, RF)
#X = pd.concat([X_cont, X_cat], axis=1)
#standardized (better for SVM, KNN)
X = pd.concat([X_cont_std, X_cat], axis=1)
#normalized 
#X = pd.concat([X_cont_norm, X_cat], axis=1)

# start: 32,560, 7840 and 24720 
#SMOTE for class imbalance - 49440 samples - 24720 of each
oversample = SMOTE()
X, Y = oversample.fit_resample(X, Y)
print(Y.size)
Y.value_counts().plot(kind='barh')

# Create test and training set
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)


# FEATURE SELECTION (only run using one at a time)

def feat_select(imp, num_feat):
    """returns list of top 10 features for the feature selection method"""
    impList = zip(X_train.columns, importance)
    feat_list = []
    i = 0
    for feature in sorted(impList, key = lambda t: t[1], reverse=True):
        #print(feature)
        if i < num_feat:
            feat_list.append(feature[0])
        i += 1
    return feat_list
    

# Chi-2 (need to normalize for chi because cant take neg values)
# cat input cat output
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train, y_train)
importance = fs.scores_
cols = feat_select(importance, 10)
X_train = X_train[cols]
X_test = X_test[cols]

# F-classif
# num input cat output
fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X_train, y_train)  
importance = fs.scores_
cols = feat_select(importance, 10)
X_train = X_train[cols]
X_test = X_test[cols]
    
# Mututal Info 
#categorical input cat output
selector = SelectKBest(mutual_info_regression, k='all')
X_train_new = selector.fit_transform(X_train, y_train) 
importance = selector.scores_
cols = feat_select(importance, 10)
X_train = X_train[cols]
X_test = X_test[cols]
  
# Decision Tree Regressor
#weights more heavily on continuous
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
cols = feat_select(importance, 10)
X_train = X_train[cols]
X_test = X_test[cols]

#DIMENSION REDUCTION - PCA ANALYSIS
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
exp_var = pca.explained_variance_ratio_*100
print('components to exp var:', len(exp_var))
plt.plot(np.cumsum(exp_var))
plt.xlabel('Num of Components')
plt.ylabel('Variance (%) Explained')

#Update training/test data
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

    
#EVAL FUNCTION
def ml_eval():
    """prints accuracy, recall, precision, and f-score for each model"""
    P, R, F, S = metrics.precision_recall_fscore_support(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print('CLASS: <= 50K (label 0)', '\nprecision:',
          P[0], '\nrecall: ', R[0], '\nf-score', F[0])
    print('CLASS: >50K (label 1)', '\nprecision:',
          P[1], '\nrecall: ', R[1], '\nf-score', F[1])


# MODELS

# DECISION TREE CLASSIFICATION 
# clf = DecisionTreeClassifier(criterion="gini", max_depth=12, splitter='best',class_weight=weights)
clf = DecisionTreeClassifier(criterion="gini", max_depth=12, splitter='best')
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('\n*****DECISION TREE EVALUATION*****')
ml_eval()

# plt.figure(figsize=(12,12))
# tree.plot_tree(clf)
# plt.show()

# SVM 
clf = svm.SVC()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n*****SVM EVALUATION*****')
ml_eval()

# KNN 
clf = KNeighborsClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n*****KNN EVALUATION*****')
ml_eval()

# RANDOM FOREST ENSEMBLE 
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('\n*****RANDOM FOREST EVALUATION*****')
ml_eval()





