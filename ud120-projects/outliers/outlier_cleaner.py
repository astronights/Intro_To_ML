#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here
    import numpy as np
    errors = net_worths - predictions
    threshold = np.percentile(np.absolute(errors), 90)
    for i in range(len(predictions)):
        if(abs(errors[i]) <= threshold):
            cleaned_data.append((ages[i], net_worths[i], errors[i]))
    print(cleaned_data)
    return cleaned_data
