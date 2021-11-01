# Machine Learning Project 1 : Higgs Boson

This machine learning project makes predictions on a data set of Higgs Boson particules.
The data set can be found at this [address](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

## Pre processing of the data

Processing is key to reveal the information that lies within the  data. You can find all the process in the **pre_processing.py** file.

The steps are as follows :
- Load the data set and separate the ids and the classification feature from the others.
- Separate the data into 3 batches in function of the PRI_jet_num.
- Fill in the missing values and correlated features.
- Standardising the data.
- Reducing the skewness.
- Normalising to give a common scale for a better performance.


## Use of machine learning methods

We implemented 6 different methods of machine learning in the **implementation.py** file :
- Linear regression with gradient descent and stochastic gradient descent with learning rate gamma.
- Least squares.
- Ridge regression with hyper parameter lambda.
- Logistic and regularized logistic regression with hyper parameter lambda and learning rate gamma.

The best values for the parameters were determined by cross validation in the **project1.ipynb**. Of all the methods, we found that the ridge regression one was the most accurate.


## Run the script

You can run the **run.py** to compute the **result.csv** which is the best classification we can make of the test data set. 
You will need the train dataset in a directory named **resources** next to the script one. Run it from inside the script directory.

It uses the pre processing and the ridge regression method discussed above. The hyper parameter lambda and the degree used for the polynomial expansion were determined by cross validation. They are the best values we got, it gives an accuracy of 83.33 % on the test data set.