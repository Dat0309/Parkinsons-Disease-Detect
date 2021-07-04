import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('parkinsons.csv')
# print(parkinsons_data.head())

# # number of rows and col in the dataframe
# print(parkinsons_data.shape)
# # Getting more infomation of the data
# print(parkinsons_data.info())
# # Getting some statistical meansures about the data
# print(parkinsons_data.describe())

'''
Distribution of target variable:
1 -> Parkinson's Positive
0 -> Healthy 
''' 
print(parkinsons_data['status'].value_counts())

# grouping the data based on the target vatiable
print(parkinsons_data.groupby('status').mean())

x = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']

'''
Splitting the data to training data & test data
'''
trainX, testX, trainY, testY = train_test_split(x,y, test_size=0.2, random_state=2)
print(x.shape, trainX.shape, testX.shape)

scl = StandardScaler()
scl.fit(trainX)
trainX = scl.transform(trainX)
testX = scl.transform(testX)

#training the SVM model with training data
model = svm.SVC(kernel='linear')
model.fit(trainX, trainY)

'''
Model evalution
'''
# Accuracy score on training data
trainX_pred = model.predict(trainX)
training_data_acc = accuracy_score(trainX_pred, trainY)
#print('Accuracy train : ', training_data_acc)

testX_pred = model.predict(testX)
test_data_acc = accuracy_score(testX_pred, testY)
#print('Accuracy test : ', test_data_acc)

'''
Build 
'''
input_data = (252.45500,261.48700,182.78600,0.00185,0.000007,0.00092,0.00113,0.00276,0.01152,0.10300,0.00614,0.00730,0.00860,0.01841,0.00432,26.80500,0.610367,0.635204,-7.319510,0.200873,2.028612,0.086398)
input_data_as_arr = np.array(input_data)
input_data_reshaped = input_data_as_arr.reshape(1,-1)
std_data = scl.transform(input_data_reshaped)

prediction = model.predict(std_data)
#print(prediction)

if(prediction[0] == 0):
    print('The person is Healthy!!')
else:
    print('The person have Parkinsons Disease')