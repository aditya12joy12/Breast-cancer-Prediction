import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)
# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
# print the first 5 rows of the dataframe
data_frame.head()
# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target
# print last 5 rows of the dataframe
data_frame.tail()
# number of rows and columns in the dataset
data_frame.shape
# getting some information about the data
data_frame.info()
# checking for missing values
data_frame.isnull().sum()
# statistical measures about the data
data_frame.describe()
# checking the distribution of Target Varibale
data_frame['label'].value_counts()
data_frame.groupby('label').mean()
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()
# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data = ', training_data_accuracy)

# input data
radius_mean = float(input("Enter radius mean :"))
texture_mean = float(input("Enter texture mean : "))
perimeter_mean = float(input("Enter premeter mean : "))
area_mean = float(input("Enter  area mean : "))
smoothness_mean = float(input("Enter smoothness mean : "))
compactness_mean = float(input("Enter compactness mean : "))
compactness_mean = float(input("Enter compactness mean : "))
concavity_mean = float(input("Enter concavity mean : "))
concave_points_mean = float(input("Enter concave points mean : "))
symmetry_mean = float(input("Enter symmetry mean :"))
fractal_dimension_mean = float(input("Enter fractal dimension mean : "))
radius_se = float(input("Enter radius se : "))
texture_se = float(input("Enter texture se : "))
perimeter_se = float(input("Enter perimeter se : "))
area_se = float(input("Enter area se : "))
smoothness_se = float(input("Enter smoothness se : "))
compactness_se = float(input("Enter compactness se : "))
concavity_se = float(input("Enter concavity se : "))
concave_points_se = float(input("Enter concave points se : "))
symmetry_se = float(input("Enter symmetry se : "))
fractal_dimension_se = float(input("Enter fractal dimension se : "))
radius_worst = float(input("Enter radius worst : "))
texture_worst = float(input("Enter texture worst : "))
perimeter_worst = float(input("Enter perimeter worst : "))
area_worst = float(input("Enter area worst : "))
smoothness_worst = float(input("Enter smoothness worst : "))
compactness_worst = float(input("Enter compactness worst : "))
concavity_worst = float(input("Enter concavity worst : "))
concave_points_worst = float(input("Enter concave points worst : "))
symmetry_worst = float(input("Enter symmetry worst : "))
fractal_dimension_worst = float(input("Enter fractal dimension worst : "))

input_data = (radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst)
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data = ', test_data_accuracy)
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

