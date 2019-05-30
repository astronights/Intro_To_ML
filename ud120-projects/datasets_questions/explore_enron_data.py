#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle
import numpy as np
enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

count = 0
for key, value in enron_data.items():
    try:
        if(enron_data[key]['total_payments']== 'NaN'):
            count += 1
    except:
        pass
count

fd=open("../final_project/poi_names.txt", "r")
ed = fd.read()
ed = ed[ed.find('htm'):]
ed = ed[1:]
poi_names = ed.split('\n')
poi_names
len(poi_names)
