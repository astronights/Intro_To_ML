#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

lols = [x for x, y in data_dict.items() if float(y['salary']) > 1000000 and y['bonus'] > 5000000]
print(lols)
