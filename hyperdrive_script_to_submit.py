from azureml.core import Run
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# create the run context
new_run = Run.get_context()

# get the workspace from the run
ws = new_run.experiment.workspace

# get parameters from the args
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int)
parser.add_argument("--min_samples_leaf", type=int)
parser.add_argument("--input_data", type=str)

args = parser.parse_args()

ne = args.n_estimators
msl = args.min_samples_leaf

#------------------------------------------------------------------------------
# Hyper parameters
#------------------------------------------------------------------------------

# load dataset
df = new_run.input_datasets["raw_data"].to_pandas_dataframe()
data_prep = df
data_prep = data_prep.dropna()
#get columns of dataframe
all_cols = data_prep.columns

# get missing values as vector of sum
dataNull = data_prep.isnull().sum()

# replace missing values of string variables (columns) with mode method
mode = data_prep.mode().iloc[0]
cols = data_prep.select_dtypes(include="object").columns
data_prep[cols] = data_prep[cols].fillna(mode)

# replace missing values of string variables (columns) with mean
mean = data_prep.mean()
data_prep = data_prep.fillna(mean)

# create dummy variables (1-hot encoding)
data_prep = pd.get_dummies(data_prep, drop_first=True)
# Create X and Y - Similar to "edit columns" in Train Module
Y = data_prep[['Loan_Status_Y']]
X = data_prep.drop(['Loan_Status_Y'], axis=1)
# Split test and train dataset
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)
# build Randomforest classifier
rfc = RandomForestClassifier(n_estimators=ne)
#rfc = RandomForestClassifier(n_estimators=ne, min_samples_leaf=msl)
# fit the random Forest with the data
rfc.fit(X_train, Y_train)
# predict the outcome using test data - Score Model
Y_predict = rfc.predict(X_test)
# get predict probabilities
Y_proba = rfc.predict_proba(X_test)[:, 1]
# get confusion matrix
cm = confusion_matrix(Y_test, Y_predict)
score =  rfc.score(X_test, Y_test)
# log the accuracy as metric
new_run.log("accuracy", score)
new_run.wait_for_completion(show_output=True)