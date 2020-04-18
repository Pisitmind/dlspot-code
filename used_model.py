import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
import seaborn as sb
from openpyxl import Workbook
from datetime import datetime
import seaborn as sns
from matplotlib.pyplot import figure

from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection
import pickle
from sklearn.metrics import confusion_matrix



##########################################################################
# # Fit the model on training set
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
data_test =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model.xlsx' ,sheet_name='testset1')
data_test_2 =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model.xlsx' ,sheet_name='testset2')
data_test_3 =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model.xlsx' ,sheet_name='testset3')

df = pd.DataFrame(data_test, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_2 = pd.DataFrame(data_test_2, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_3 = pd.DataFrame(data_test_3, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])

df =df.fillna(0)
df_2 =df.fillna(0)
df_3 =df.fillna(0)

answer_r = df.isGood
answer_r2 = df_2.isGood
answer_r3 = df_3.isGood
# Test_X=df.drop('isGood', axis='columns')  #old data
X_test = df.drop('isGood', axis='columns')
X_test_2 = df_2.drop('isGood', axis='columns')
X_test_3 = df_3.drop('isGood', axis='columns')
# # some time later...
# X_train_o, X_test_o, Y_train_o, Y_test_o =train_test_split(inputs,target,test_size=0.2,random_state=0) #all old
# # load the model from disk
loaded_model_gau_ori = pickle.load(open('./model_gaussianNB_ori.sav', 'rb'))
loaded_model_Knn_ori = pickle.load(open('./model_KNN_ori.sav', 'rb'))
loaded_model_logistic_ori = pickle.load(open('./model_logistic_ori.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)

predict_out_gau_ori = loaded_model_gau_ori.predict(X_test)
predict_out_Knn_ori = loaded_model_Knn_ori.predict(X_test)
predict_out_logistic_ori = loaded_model_logistic_ori.predict(X_test)

predict_out_gau_ori_2 = loaded_model_gau_ori.predict(X_test_2)
predict_out_Knn_ori_2 = loaded_model_Knn_ori.predict(X_test_2)
predict_out_logistic_ori_2 = loaded_model_logistic_ori.predict(X_test_2)

predict_out_gau_ori_3 = loaded_model_gau_ori.predict(X_test_3)
predict_out_Knn_ori_3 = loaded_model_Knn_ori.predict(X_test_3)
predict_out_logistic_ori_3 = loaded_model_logistic_ori.predict(X_test_3)


acc_gau_rate = confusion_matrix(answer_r, predict_out_gau_ori)
acc_Knn_rate = confusion_matrix(answer_r, predict_out_Knn_ori)
acc_logis_rate = confusion_matrix(answer_r, predict_out_logistic_ori)

acc_gau_rate_2 = confusion_matrix(answer_r2, predict_out_gau_ori_2)
acc_Knn_rate_2 = confusion_matrix(answer_r2, predict_out_Knn_ori_2)
acc_logis_rate_2 = confusion_matrix(answer_r2, predict_out_logistic_ori_2)

acc_gau_rate_3 = confusion_matrix(answer_r3, predict_out_gau_ori_3)
acc_Knn_rate_3 = confusion_matrix(answer_r3, predict_out_Knn_ori_3)
acc_logis_rate_3 = confusion_matrix(answer_r3, predict_out_logistic_ori_3)

acc_test_gau_1 = accuracy_score(answer_r,predict_out_gau_ori)
acc_test_Knn_1 = accuracy_score(answer_r,predict_out_Knn_ori)
acc_test_logis_1 = accuracy_score(answer_r,predict_out_logistic_ori)

acc_test_gau_2 = accuracy_score(answer_r2,predict_out_gau_ori_2)
acc_test_Knn_2 = accuracy_score(answer_r2,predict_out_Knn_ori_2)
acc_test_logis_2 = accuracy_score(answer_r2,predict_out_logistic_ori_2)

acc_test_gau_3 = accuracy_score(answer_r3,predict_out_gau_ori_3)
acc_test_Knn_3 = accuracy_score(answer_r3,predict_out_Knn_ori_3)
acc_test_logis_3 = accuracy_score(answer_r3,predict_out_logistic_ori_3)

print('#####################################-----result Set 1----#####################################')
print(predict_out_gau_ori)

print('Accuracy : ',acc_gau_rate)
print('')
print('Accuracy predict GaussianNB : ',acc_test_gau_1)
print('')

print(predict_out_Knn_ori)
print('')
print('Accuracy : ',acc_Knn_rate)
print('')
print('Accuracy predict Knn : ',acc_test_Knn_1)
print('')

print(predict_out_logistic_ori)
print('')
print('Accuracy : ',acc_logis_rate)
print('')
print('Accuracy predict Logistic : ',acc_test_logis_1)
print('')



print('#####################################-----result Set 2----#####################################')
print(predict_out_gau_ori_2)
print('')
print('Accuracy : ',acc_gau_rate_2)
print('')
print('Accuracy predict GaussianNB : ',acc_test_gau_2)
print('')

print(predict_out_Knn_ori_2)
print('')
print('Accuracy : ',acc_Knn_rate_2)
print('')
print('Accuracy predict Knn : ',acc_test_Knn_2)
print('')

print(predict_out_logistic_ori_2)
print('')
print('Accuracy : ',acc_logis_rate_2)
print('')
print('Accuracy predict Logistic : ',acc_test_logis_2)
print('')


print('#####################################-----result Set 3----#####################################')
print(predict_out_gau_ori_3)
print('')
print('Accuracy : ',acc_gau_rate_3)
print('')
print('Accuracy predict GaussianNB : ',acc_test_gau_3)
print('')

print(predict_out_Knn_ori_3)
print('')
print('Accuracy : ',acc_Knn_rate_3)
print('')
print('Accuracy predict Knn : ',acc_test_Knn_3)
print('')

print(predict_out_logistic_ori_3)
print('')
print('Accuracy : ',acc_logis_rate_3)
print('')
print('Accuracy predict Logistic : ',acc_test_logis_3)
print('')

##########################################################################