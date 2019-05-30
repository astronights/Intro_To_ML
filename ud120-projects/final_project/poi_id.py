#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list = ['poi', 'salary', 'to_messages', 'loan_advances', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
dd = pd.DataFrame.from_dict(data_dict, orient='index')
dd = pd.DataFrame(dd, columns = features_list)
### Task 2: Remove outliers
dd = dd.drop('TOTAL')
for c in dd.columns:
    dd[c] = pd.to_numeric(dd[c], errors='coerce')
dd = dd.fillna(0)
dd
### Task 3: Create new feature(s)
#No new features to be added at the moment.
### Store to my_dataset for easy export below.
my_dataset = dd.T.to_dict()
my_dataset
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
parameters = {'criterion':('gini', 'entropy')}
DT = tree.DecisionTreeClassifier(random_state = 10)
clf = GridSearchCV(DT, parameters, scoring = 'f1')
clf= clf.fit(features_train, labels_train)
clf = clf.best_estimator_
estimators = [('scaler', MinMaxScaler()),
            ('reduce_dim', PCA()),
            ('clf', clf)]
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print( "Accuracy: ", accuracy)
target_names = ['non_poi', 'poi']
print( classification_report(y_true = labels_test, y_pred =pred, target_names = target_names))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
