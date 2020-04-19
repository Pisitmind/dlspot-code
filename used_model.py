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
data_test =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model1.xlsx' ,sheet_name='Set1')###good day
data_test_2 =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model1.xlsx' ,sheet_name='Set2')###good night
data_test_3 =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model1.xlsx' ,sheet_name='Set3')###bad day
data_test_4 =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\test_model1.xlsx' ,sheet_name='Set4')###bad night


# print(data_test)
# print("----")
# print(data_test_2)

df = pd.DataFrame(data_test, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_2 = pd.DataFrame(data_test_2, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_3 = pd.DataFrame(data_test_3, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_4 = pd.DataFrame(data_test_4, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ,'isGood' ])
df_1hr = pd.DataFrame(data_test, columns= ['Humidity','Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_1hr'	,'XbarVis_1hr',	'XbarUVindex_1hr','XbarObject(*C)_1hr',
                        'XbarCO2Analog(ppm)_1hr','SD_Humidity_1hr'	,'SD_Vis_1hr'	,'SD_UVindex_1hr', 'SD_Object(*C)_1hr',	'SD_co2analog_1hr'])

df_1hr_2 = pd.DataFrame(data_test_2, columns= ['Humidity','Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_1hr'	,'XbarVis_1hr',	'XbarUVindex_1hr','XbarObject(*C)_1hr',
                        'XbarCO2Analog(ppm)_1hr','SD_Humidity_1hr'	,'SD_Vis_1hr'	,'SD_UVindex_1hr', 'SD_Object(*C)_1hr',	'SD_co2analog_1hr'])

df_1hr_3 = pd.DataFrame(data_test_3, columns= ['Humidity','Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_1hr'	,'XbarVis_1hr',	'XbarUVindex_1hr','XbarObject(*C)_1hr',
                        'XbarCO2Analog(ppm)_1hr','SD_Humidity_1hr'	,'SD_Vis_1hr'	,'SD_UVindex_1hr', 'SD_Object(*C)_1hr',	'SD_co2analog_1hr'])

df_1hr_4 = pd.DataFrame(data_test_4, columns= ['Humidity','Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_1hr'	,'XbarVis_1hr',	'XbarUVindex_1hr','XbarObject(*C)_1hr',
                        'XbarCO2Analog(ppm)_1hr','SD_Humidity_1hr'	,'SD_Vis_1hr'	,'SD_UVindex_1hr', 'SD_Object(*C)_1hr',	'SD_co2analog_1hr'])


# df =df.fillna(0)
# df_2 =df_2.fillna(0)
df_1hr =df_1hr.fillna(0)
df_1hr_2 =df_1hr_2.fillna(0)
df_1hr_3 =df_1hr_3.fillna(0)
df_1hr_4 =df_1hr_4.fillna(0)

# print(df.head())
# print(df_2.head())
# print(df_3.head())
print('------------')
answer_r = df.isGood
answer_r2 = df_2.isGood
answer_r3 = df_3.isGood
answer_r4 = df_4.isGood


answer_1hr =df_1hr.isGood
answer_1hr_2 =df_1hr_2.isGood
answer_1hr_3 =df_1hr_3.isGood
answer_1hr_4 =df_1hr_4.isGood
X_test_1hr = df_1hr.drop('isGood', axis='columns')
X_test_1hr_2 = df_1hr_2.drop('isGood', axis='columns')
X_test_1hr_3 = df_1hr_3.drop('isGood', axis='columns')
X_test_1hr_4 = df_1hr_4.drop('isGood', axis='columns')


# Test_X=df.drop('isGood', axis='columns')  #old data
X_test = df.drop('isGood', axis='columns')
X_test_2 = df_2.drop('isGood', axis='columns')
X_test_3 = df_3.drop('isGood', axis='columns')
X_test_4 = df_4.drop('isGood', axis='columns')

# print(X_test.head())
# print(X_test_2.head())
# print(X_test_3.head())
# print(X_test_1hr.head())



# # some time later...
# X_train_o, X_test_o, Y_train_o, Y_test_o =train_test_split(inputs,target,test_size=0.2,random_state=0) #all old
# # load the model from disk
loaded_model_gau_ori = pickle.load(open('./model_gaussianNB_ori.sav', 'rb'))
loaded_model_Knn_ori = pickle.load(open('./model_KNN_ori.sav', 'rb'))
loaded_model_logistic_ori = pickle.load(open('./model_logistic_ori.sav', 'rb'))
# # load the model from 1hr
loaded_model_gau_1hr = pickle.load(open('./model_gaussianNB_1.sav', 'rb'))
loaded_model_Knn_1hr = pickle.load(open('./model_KNN_1h.sav', 'rb'))
loaded_model_logistic_1hr = pickle.load(open('./model_logistic_1h.sav', 'rb'))
# # load the model from 3hr
loaded_model_gau_3hr = pickle.load(open('./model_gaussianNB_3.sav', 'rb'))
loaded_model_Knn_3hr = pickle.load(open('./model_KNN_3h.sav', 'rb'))
loaded_model_logistic_3hr = pickle.load(open('./model_logistic_3h.sav', 'rb'))
# # load the model from 6hr
loaded_model_gau_6hr = pickle.load(open('./model_gaussianNB_6.sav', 'rb'))
loaded_model_Knn_6hr = pickle.load(open('./model_KNN_6h.sav', 'rb'))
loaded_model_logistic_6hr = pickle.load(open('./model_logistic_6h.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)




################ use model to predict pure data ######################
predict_out_gau_ori = loaded_model_gau_ori.predict(X_test)
predict_out_Knn_ori = loaded_model_Knn_ori.predict(X_test)
predict_out_logistic_ori = loaded_model_logistic_ori.predict(X_test)

predict_out_gau_ori_2 = loaded_model_gau_ori.predict(X_test_2)
predict_out_Knn_ori_2 = loaded_model_Knn_ori.predict(X_test_2)
predict_out_logistic_ori_2 = loaded_model_logistic_ori.predict(X_test_2)

predict_out_gau_ori_3 = loaded_model_gau_ori.predict(X_test_3)
predict_out_Knn_ori_3 = loaded_model_Knn_ori.predict(X_test_3)
predict_out_logistic_ori_3 = loaded_model_logistic_ori.predict(X_test_3)

predict_out_gau_ori_4 = loaded_model_gau_ori.predict(X_test_4)
predict_out_Knn_ori_4 = loaded_model_Knn_ori.predict(X_test_4)
predict_out_logistic_ori_4 = loaded_model_logistic_ori.predict(X_test_4)


############use model to predict 1hr data ########################

predict_out_gau_1hr = loaded_model_gau_1hr.predict(X_test_1hr)
acc_gau_rate_1hr = confusion_matrix(answer_1hr, predict_out_gau_1hr)
acc_test_gau_1hr = accuracy_score(answer_1hr,predict_out_gau_1hr)

predict_out_Knn_1hr = loaded_model_Knn_1hr.predict(X_test_1hr)
acc_Knn_rate_1hr = confusion_matrix(answer_1hr, predict_out_Knn_1hr)
acc_test_Knn_1hr = accuracy_score(answer_1hr,predict_out_Knn_1hr)

predict_out_logis_1hr = loaded_model_logistic_1hr.predict(X_test_1hr)
acc_logis_rate_1hr = confusion_matrix(answer_1hr, predict_out_logis_1hr)
acc_test_logis_1hr = accuracy_score(answer_1hr,predict_out_logis_1hr)

# ----------------------- set 2-------------------------#

predict_out_gau_1hr_2 = loaded_model_gau_1hr.predict(X_test_1hr_2)
acc_gau_rate_1hr_2 = confusion_matrix(answer_1hr_2, predict_out_gau_1hr_2)
acc_test_gau_1hr_2 = accuracy_score(answer_1hr_2,predict_out_gau_1hr_2)

predict_out_Knn_1hr_2 = loaded_model_Knn_1hr.predict(X_test_1hr_2)
acc_Knn_rate_1hr_2 = confusion_matrix(answer_1hr_2, predict_out_Knn_1hr_2)
acc_test_Knn_1hr_2 = accuracy_score(answer_1hr_2,predict_out_Knn_1hr_2)

predict_out_logis_1hr_2 = loaded_model_logistic_1hr.predict(X_test_1hr_2)
acc_logis_rate_1hr_2 = confusion_matrix(answer_1hr_2, predict_out_logis_1hr_2)
acc_test_logis_1hr_2 = accuracy_score(answer_1hr_2,predict_out_logis_1hr_2)

# -------------------- set 3 -----------------------------#

predict_out_gau_1hr_3 = loaded_model_gau_1hr.predict(X_test_1hr_3)
acc_gau_rate_1hr_3 = confusion_matrix(answer_1hr_3, predict_out_gau_1hr_3)
acc_test_gau_1hr_3 = accuracy_score(answer_1hr_3,predict_out_gau_1hr_3)

predict_out_Knn_1hr_3 = loaded_model_Knn_1hr.predict(X_test_1hr_3)
acc_Knn_rate_1hr_3 = confusion_matrix(answer_1hr_3, predict_out_Knn_1hr_3)
acc_test_Knn_1hr_3 = accuracy_score(answer_1hr_3,predict_out_Knn_1hr_3)

predict_out_logis_1hr_3 = loaded_model_logistic_1hr.predict(X_test_1hr_3)
acc_logis_rate_1hr_3 = confusion_matrix(answer_1hr_3, predict_out_logis_1hr_3)
acc_test_logis_1hr_3 = accuracy_score(answer_1hr_3,predict_out_logis_1hr_3)

# -------------------- set 4 -----------------------------#

predict_out_gau_1hr_4 = loaded_model_gau_1hr.predict(X_test_1hr_3)
acc_gau_rate_1hr_4 = confusion_matrix(answer_1hr_3, predict_out_gau_1hr_3)
acc_test_gau_1hr_4 = accuracy_score(answer_1hr_3,predict_out_gau_1hr_3)

predict_out_Knn_1hr_4 = loaded_model_Knn_1hr.predict(X_test_1hr_3)
acc_Knn_rate_1hr_4 = confusion_matrix(answer_1hr_3, predict_out_Knn_1hr_3)
acc_test_Knn_1hr_4 = accuracy_score(answer_1hr_3,predict_out_Knn_1hr_3)

predict_out_logis_1hr_4 = loaded_model_logistic_1hr.predict(X_test_1hr_4)
acc_logis_rate_1hr_4 = confusion_matrix(answer_1hr_4, predict_out_logis_1hr_4)
acc_test_logis_1hr_4 = accuracy_score(answer_1hr_4,predict_out_logis_1hr_4)





############# matrix pure data ###########

acc_gau_rate = confusion_matrix(answer_r, predict_out_gau_ori)
acc_Knn_rate = confusion_matrix(answer_r, predict_out_Knn_ori)
acc_logis_rate = confusion_matrix(answer_r, predict_out_logistic_ori)

acc_gau_rate_2 = confusion_matrix(answer_r2, predict_out_gau_ori_2)
acc_Knn_rate_2 = confusion_matrix(answer_r2, predict_out_Knn_ori_2)
acc_logis_rate_2 = confusion_matrix(answer_r2, predict_out_logistic_ori_2)

acc_gau_rate_3 = confusion_matrix(answer_r3, predict_out_gau_ori_3)
acc_Knn_rate_3 = confusion_matrix(answer_r3, predict_out_Knn_ori_3)
acc_logis_rate_3 = confusion_matrix(answer_r3, predict_out_logistic_ori_3)

acc_gau_rate_4 = confusion_matrix(answer_r4, predict_out_gau_ori_4)
acc_Knn_rate_4 = confusion_matrix(answer_r4, predict_out_Knn_ori_4)
acc_logis_rate_4 = confusion_matrix(answer_r4, predict_out_logistic_ori_4)




####### acc  pure data####
acc_test_gau_1 = accuracy_score(answer_r,predict_out_gau_ori)
acc_test_Knn_1 = accuracy_score(answer_r,predict_out_Knn_ori)
acc_test_logis_1 = accuracy_score(answer_r,predict_out_logistic_ori)

acc_test_gau_2 = accuracy_score(answer_r2,predict_out_gau_ori_2)
acc_test_Knn_2 = accuracy_score(answer_r2,predict_out_Knn_ori_2)
acc_test_logis_2 = accuracy_score(answer_r2,predict_out_logistic_ori_2)

acc_test_gau_3 = accuracy_score(answer_r3,predict_out_gau_ori_3)
acc_test_Knn_3 = accuracy_score(answer_r3,predict_out_Knn_ori_3)
acc_test_logis_3 = accuracy_score(answer_r3,predict_out_logistic_ori_3)

acc_test_gau_4 = accuracy_score(answer_r4,predict_out_gau_ori_4)
acc_test_Knn_4 = accuracy_score(answer_r4,predict_out_Knn_ori_4)
acc_test_logis_4 = accuracy_score(answer_r4,predict_out_logistic_ori_4)

print('#####################################-----result Set 1(Good day)----#####################################')


print(predict_out_gau_ori)
print('Confusion Matrix : ',"\n",acc_gau_rate)
print('ค่าความถูกต้องของ model GaussianNB : ',acc_test_gau_1)
print('------------------------------')
print('')

print(predict_out_Knn_ori)
print('Confusion Matrix : ',"\n",acc_Knn_rate)
print('ค่าความถูกต้องของ model  Knn : ',acc_test_Knn_1)
print('------------------------------')
print('')


print(predict_out_logistic_ori)
print('Confusion Matrix : ',"\n",acc_logis_rate)
print('ค่าความถูกต้องของ model  Logistic : ',acc_test_logis_1)
print('------------------------------')
print('')




print('#####################################-----result Set 2(Good night)-----#####################################')
print(predict_out_gau_ori_2)
print('Confusion Matrix : ',"\n",acc_gau_rate_2)
print('ค่าความถูกต้องของ model  GaussianNB : ',acc_test_gau_2)
print('------------------------------')
print('')


print(predict_out_Knn_ori_2)
print('Confusion Matrix : ',"\n",acc_Knn_rate_2)
print('ค่าความถูกต้องของ model  Knn : ',acc_test_Knn_2)
print('------------------------------')
print('')


print(predict_out_logistic_ori_2)
print('Confusion Matrix : ',"\n",acc_logis_rate_2)
print('ค่าความถูกต้องของ model  Logistic : ',acc_test_logis_2)
print('------------------------------')
print('')


print('#####################################-----result Set 3(Bad day)----#####################################')
print(predict_out_gau_ori_3)
print('Confusion Matrix : ',"\n",acc_gau_rate_3)
print('ค่าความถูกต้องของ model GaussianNB : ',acc_test_gau_3)
print('------------------------------')
print('')

print(predict_out_Knn_ori_3)
print('Confusion Matrix : ',"\n",acc_Knn_rate_3)
print('ค่าความถูกต้องของ model Knn : ',acc_test_Knn_3)
print('------------------------------')
print('')

print(predict_out_logistic_ori_3)
print('Confusion Matrix : ',"\n",acc_logis_rate_3)
print('ค่าความถูกต้องของ model Logistic : ',acc_test_logis_3)
print('------------------------------')
print('')

print('#####################################-----result Set 4(Bad night)----#####################################')

print(predict_out_gau_ori_4)
print('Confusion Matrix : ',"\n", acc_gau_rate_4)
print('ค่าความถูกต้องของ model  GaussianNB : ',acc_test_gau_4)
print('------------------------------')
print('')

print(predict_out_Knn_ori_4)
print('Confusion Matrix : ', "\n", acc_Knn_rate_4)
print('ค่าความถูกต้องของ model  Knn : ',acc_test_Knn_4)
print('------------------------------')
print('')

print(predict_out_logistic_ori_4)
print('Confusion Matrix : ',"\n", acc_logis_rate_4)
print('ค่าความถูกต้องของ model  Logistic : ',acc_test_logis_4)
print('------------------------------')
print('')

##########################################################################

print('#####################################-----result 1 hr set 1----#####################################')

###### 1hr set 1###############
print(predict_out_gau_1hr)
print('Confusion Matrix : ',"\n", acc_gau_rate_1hr)
print('ค่าความถูกต้องของ model  GaussianNB 1hr : ',acc_test_gau_1hr)
print('------------------------------')
print('')
print(predict_out_Knn_1hr)
print('Confusion Matrix : ',"\n", acc_Knn_rate_1hr)
print('ค่าความถูกต้องของ model  Knn 1hr : ',acc_test_Knn_1hr)
print('------------------------------')
print('')
print(predict_out_logis_1hr)
print('Confusion Matrix : ',"\n", acc_logis_rate_1hr)
print('ค่าความถูกต้องของ model  logistic 1hr : ',acc_test_logis_1hr)
print('------------------------------')
print('')

print('#####################################-----result 1 hr set 2----#####################################')
###### 1hr set 2###############
print(predict_out_gau_1hr_2)
print('Confusion Matrix : ',"\n", acc_gau_rate_1hr_2)
print('ค่าความถูกต้องของ model  GaussianNB 1hr : ',acc_test_gau_1hr_2)
print('------------------------------')
print('')
print(predict_out_Knn_1hr_2)
print('Confusion Matrix : ',"\n", acc_Knn_rate_1hr_2)
print('ค่าความถูกต้องของ model  Knn 1hr : ',acc_test_Knn_1hr_2)
print('------------------------------')
print('')
print(predict_out_logis_1hr_2)
print('Confusion Matrix : ',"\n", acc_logis_rate_1hr_2)
print('ค่าความถูกต้องของ model  logistic 1hr : ',acc_test_logis_1hr_2)
print('------------------------------')
print('')

print('#####################################-----result 1 hr set 3----#####################################')
###### 1hr set 2###############
print(predict_out_gau_1hr_3)
print('Confusion Matrix : ',"\n", acc_gau_rate_1hr_3)
print('ค่าความถูกต้องของ model  GaussianNB 1hr : ',acc_test_gau_1hr_3)
print('------------------------------')
print('')
print(predict_out_Knn_1hr_3)
print('Confusion Matrix : ',"\n", acc_Knn_rate_1hr_3)
print('ค่าความถูกต้องของ model  Knn 1hr : ',acc_test_Knn_1hr_3)
print('------------------------------')
print('')
print(predict_out_logis_1hr_3)
print('Confusion Matrix : ',"\n", acc_logis_rate_1hr_3)
print('ค่าความถูกต้องของ model  logistic 1hr : ',acc_test_logis_1hr_3)
print('------------------------------')
print('')

print('#####################################-----result 1 hr set 4----#####################################')
###### 1hr set 2###############
print(predict_out_gau_1hr_4)
print('Confusion Matrix : ',"\n", acc_gau_rate_1hr_4)
print('ค่าความถูกต้องของ model  GaussianNB 1hr : ',acc_test_gau_1hr_4)
print('------------------------------')
print('')
print(predict_out_Knn_1hr_4)
print('Confusion Matrix : ',"\n", acc_Knn_rate_1hr_4)
print('ค่าความถูกต้องของ model  Knn 1hr : ',acc_test_Knn_1hr_4)
print('------------------------------')
print('')
print(predict_out_logis_1hr_4)
print('Confusion Matrix : ',"\n", acc_logis_rate_1hr_4)
print('ค่าความถูกต้องของ model  logistic 1hr : ',acc_test_logis_1hr_4)
print('------------------------------')
print('')

#####################