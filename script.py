# import required libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import joblib

# loading the dataset to dataframe
df = pd.read_csv("diabetes.csv")

# separating data and labels
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()

X = scaler.fit_transform(X.values)
y = df['Outcome']

# Train test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state = 42)

classifier  = svm.SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

# accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy score of training data : ', training_data_accuracy)
print('Accuracy score of test data : ', test_data_accuracy)

joblib.dump(classifier, 'diabetes-prediction-model.pkl')



# def user_data(raw_input_data):
#     input_data = np.asarray(raw_input_data)
#     reshaped_input_data = input_data.reshape(1,-1)

#     std_input_data = scaler.transform(reshaped_input_data)

#     real_prediction = classifier.predict(std_input_data)

#     if real_prediction == 1:
#         print('person is diabetic')
#     else:
#         print('person is not diabetic')