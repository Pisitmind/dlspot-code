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
##ML Zone ##
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.model_selection import KFold,cross_val_score

data =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx')
data_1hr =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx', sheet_name='all1hr')
data_3hr =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx', sheet_name='all3hr')
data_6hr =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx', sheet_name='all6hr')
olddata =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx', sheet_name='old')
olddata_wodate =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\alldata_update.xlsx', sheet_name='old_wodate')



# data = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\output_cur.xlsx', sheet_name='feature_new') 
# data_old = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdate.xlsx', sheet_name='Alldata') 
data.to_csv (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\updatedata_April.csv', index = None, header=True)

df_filter = pd.DataFrame(data_1hr, columns= ['Humidity','Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_1hr'	,'XbarVis_1hr',	'XbarUVindex_1hr','XbarObject(*C)_1hr',
                        'XbarCO2Analog(ppm)_1hr','SD_Humidity_1hr'	,'SD_Vis_1hr'	,'SD_UVindex_1hr', 'SD_Object(*C)_1hr',	'SD_co2analog_1hr'])
df_filter = df_filter.fillna(0) #ดัก missing value ถ้าเจอ เติม 0

df_filter_3hr = pd.DataFrame(data_3hr, columns= ['Humidity',	'Vis',	'UVindex','Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood','XbarHumidity_3hr',	'XbarVis_3hr',	'XbarUVindex_3hr',	'XbarObject(*C)_3hr',
                        'Xbar_co2analog_3hr', 'SD_Humidity_3hr',	'SD_Vis_3hr',	'SD_UVindex_3hr', 'SD_Object(*C)_3hr'	,'SD_CO2Analog(ppm)_3hr'])
df_filter_3hr = df_filter_3hr.fillna(0) #ดัก missing value ถ้าเจอ เติม 0

df_filter_6hr = pd.DataFrame(data_6hr, columns= ['Humidity'	,'Object(*C)','CO2Analog(ppm)','CO2PWM(ppm)','isGood',
                        'Xbar_Humidity_6hr'	,'Xbar_Vis_6hr'	,'Xbar_UVindex_6hr','Xbar_Object(*C)_6hr',	'Xbar_co2analog_6hr',
                        'SD_Humidity_6hr',	'SD_Vis_6hr',	'SD_UVindex_6hr', 'SD_Object(*C)_6hr','SD_CO2Analog(ppm)_6hr'])
df_filter_6hr = df_filter_6hr.fillna(0) #ดัก missing value ถ้าเจอ เติม 0

df = pd.DataFrame(olddata, columns= ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)','isGood' ])

df = df.fillna(0) #ดัก missing value ถ้าเจอ เติม 0

# df.loc[df['isGood'] < 1, 'labeled'] = '0' 
# df.loc[df['isGood'] >= 1, 'labeled'] = '1' 

# df = df.dropna(axis=0)
# ############################################

# df_filter.loc[df_filter['isGood'] < 1, 'labeled'] = '0' 
# df_filter.loc[df_filter['isGood'] >= 1, 'labeled'] = '1' 

# df_filter = df_filter.dropna(axis=0)  #case missing value drop ทิ้งเลย

target=df.isGood
# print(target)


# df_filter = df_filter.dropna(axis=0)  #case missing value drop ทิ้งเลย

target_filter_1hr =df_filter.isGood  #targetdata_1hr
target_filter_3hr =df_filter_3hr.isGood #targetdata_3hr
target_filter_6hr =df_filter_6hr.isGood #targetdata_3hr


inputs=df.drop('isGood', axis='columns')  #old data
input_1hr = df_filter.drop('isGood', axis='columns') #1hrdata
input_3hr = df_filter_3hr.drop('isGood', axis='columns') #3hrdata
input_6hr = df_filter_6hr.drop('isGood', axis='columns') #6hrdata

# print('+++++++++++++++++++++++++')
# print(inputs)
# print('+++++++++++++++++++++++++')
# print(target)
# print('+++++++++++++++++++++++++')
# print(target_filter_1hr)
# print('+++++++++++++++++++++++++')
# print(input_1hr)
# print('+++++++++++++++++++++++++')
# print(target_filter_3hr)
# print('+++++++++++++++++++++++++')
# print(input_3hr)
# print('+++++++++++++++++++++++++')
# print(target_filter_6hr)
# print('+++++++++++++++++++++++++')
# print(input_6hr)

# # print(df_filter)
# # print(df_filter.shape)
# # test = df.head()
# df['some_column'].plot(figsize=(10, 5))

# print(df.class_day)






# def normalize(dataset):
#     dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*100    #อิงค่า dataset[?]ให้อยู่ในช่วง 0-100 ค่าอื่นก็ลดหย่อนเป็นสัดส่วน
#     dataNorm["CO2PWM(ppm)"]=dataset["CO2PWM(ppm)"]
#     # dataNorm["CO2Analog(ppm)"]=dataset["CO2Analog(ppm)"]
#     return dataNorm


# data_norm = normalize(olddata_wodate)  #### edit here to change data to norm

# df_n =pd.DataFrame(data_norm, columns=   ['Humidity',	'Vis', 'UVindex',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)','isGood' ])
# # df_n =pd.DataFrame(data_norm, columns=   ['Temperature'	,'Humidity'	,'Object(*C)','CO2Analog(ppm)',
# #                         'CO2PWM(ppm)',		'Xbar_Humidity', 'Xbar_co2analog',	'SD_Humidity',
# #                         'SD_co2analog',		'isGood'])
# df_n.loc[df_n['isGood'] < 1, 'labeled'] = '0' 
# df_n.loc[df_n['isGood'] >= 1, 'labeled'] = '1' 
# target_norm =df_n.labeled
# # df_n = df_n.dropna(axis=0)
# input_norm = df_n.drop('isGood', axis='columns')
# input_norm.drop('labeled', axis='columns',inplace=True) 
# input_norm.drop('CO2PWM(ppm)', axis='columns',inplace=True) 
# # # print('Normmmm na --------------')
# # # print(target_norm)
# # # print(input_norm)



####################phase แสดงผล ค่าต่างๆในตารางก่อนนำไปเทรน และ เทส#####################################



# # print("---------------------1---------------------")

# print(inputs)





# # # ############ close tag after set file #############

# writer = pd.ExcelWriter('outputupdate_136hr_new.xlsx')
# target.to_excel(writer,'labeled_old')
# target_filter_1hr.to_excel(writer,'labeled_1hr')  
# target_filter_3hr.to_excel(writer,'labeled_3hr')
# target_filter_6hr.to_excel(writer,'labeled_6hr')
# # target_norm.to_excel(writer,'labeled_norm')
# inputs.to_excel(writer,'feature_old')
# input_1hr.to_excel(writer,'feature_1hr')
# input_3hr.to_excel(writer,'feature_3hr')
# input_6hr.to_excel(writer,'feature_6hr')
# # input_norm.to_excel(writer,'feature_norm')
# writer.save()


# # # standardized_X = preprocessing.scale(inputs)

# # # ###################################### Prediction Zone #######################################


# X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(input_norm,target_norm,test_size=0.2,random_state=0) #all  norm and have tmp
X_train_o, X_test_o, Y_train_o, Y_test_o =train_test_split(inputs,target,test_size=0.2,random_state=0) #all old
X_train_1, X_test_1, Y_train_1, Y_test_1 =train_test_split(input_1hr,target_filter_1hr,test_size=0.2,random_state=0) #all with  xbar sd  1hr
X_train_3, X_test_3, Y_train_3, Y_test_3 =train_test_split(input_3hr,target_filter_3hr,test_size=0.2,random_state=0) #all with  xbar sd  3hr
X_train_6, X_test_6, Y_train_6, Y_test_6 =train_test_split(input_6hr,target_filter_6hr,test_size=0.2,random_state=0) #all with  xbar sd  6hr





no_neighbors = np.arange(2, 10)
test_accuracy_ori = np.empty(len(no_neighbors))
test_accuracy_1hr = np.empty(len(no_neighbors))
test_accuracy_3hr = np.empty(len(no_neighbors))
test_accuracy_6hr = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn_ori = KNeighborsClassifier(n_neighbors=k)
    knn_1hr = KNeighborsClassifier(n_neighbors=k)
    knn_3hr = KNeighborsClassifier(n_neighbors=k)
    knn_6hr = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn_ori.fit(X_train_o,Y_train_o)
    knn_1hr.fit(X_train_1,Y_train_1)
    knn_3hr.fit(X_train_3,Y_train_3)
    knn_6hr.fit(X_train_6,Y_train_6)
    
    # Compute accuracy on the training set

    # Compute accuracy on the testing set
    test_accuracy_ori[i] = knn_ori.score(X_test_o, Y_test_o)
    test_accuracy_1hr[i] = knn_1hr.score(X_test_1, Y_test_1)
    test_accuracy_3hr[i] = knn_3hr.score(X_test_3, Y_test_3)
    test_accuracy_6hr[i] = knn_6hr.score(X_test_6, Y_test_6)

# Visualization of k values vs accuracy

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy_1hr, label = '1-hr Average data')
plt.plot(no_neighbors, test_accuracy_3hr, label = '3-hr Average data')
plt.plot(no_neighbors, test_accuracy_6hr, label = '6-hr Average data')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.savefig('./graph_average.png')
plt.show()


##################################################################################





# #############Gau Ori###########
model_ori= GaussianNB()
model_ori.fit(X_train_o,Y_train_o)
y_model_o = model_ori.predict(X_test_o)

# #############Gau norm#############
# model_norm= GaussianNB()
# model_norm.fit(X_train_norm,Y_train_norm)
# y_model_norm = model_norm.predict(X_test_norm)

# # #############Gau new###########
gau_model_1= GaussianNB()
gau_model_1.fit(X_train_1 ,Y_train_1)
y_model_1 = gau_model_1.predict(X_test_1)

#############Gau 3hr###########
gau_model_3= GaussianNB()
gau_model_3.fit(X_train_3 ,Y_train_3)
y_model_3 = gau_model_3.predict(X_test_3)

# #############Gau 6hr###########
gau_model_6 = GaussianNB()
gau_model_6.fit(X_train_6 ,Y_train_6)
y_model_6 = gau_model_6.predict(X_test_6)



# ###########Knn ########## ลองเพิ่มม-ลด paramiter ดูผลกระทบ
modelknn_n2 = KNeighborsClassifier(n_neighbors=2)
modelknn_n3 = KNeighborsClassifier(n_neighbors=3)
modelknn_n4 = KNeighborsClassifier(n_neighbors=4)
modelknn_n5 = KNeighborsClassifier(n_neighbors=5)
modelknn_n6 = KNeighborsClassifier(n_neighbors=6)
modelknn_n7 = KNeighborsClassifier(n_neighbors=7)
modelknn_n8 = KNeighborsClassifier(n_neighbors=8)
modelknn_n9 = KNeighborsClassifier(n_neighbors=9)
modelknn_n10 = KNeighborsClassifier(n_neighbors=10)
##########################################################


modelknn = KNeighborsClassifier(n_neighbors=3)
# modelknn.fit(X_train_o,Y_train_o)
# knn_score = modelknn.score(X_train_o,Y_train_o)
# answer = modelknn.predict(X_test_o)
# ###########Knn norm##########
# modelknn_norm = KNeighborsClassifier(n_neighbors=5)
# modelknn_norm.fit(X_train_norm,Y_train_norm)
# knn_score_norm = modelknn_norm.score(X_train_norm,Y_train_norm)
# answer_norm = modelknn_norm.predict(X_test_norm)
# # print("Knn score : ", knn_score)
##########Knn 1hr ##########
modelknn_1 = KNeighborsClassifier(n_neighbors=3)
modelknn_1.fit(X_train_1,Y_train_1)
knn_score_1 = modelknn_1.score(X_train_1,Y_train_1)
answer_1 = modelknn_1.predict(X_test_1)
# ##########Knn 3hr ##########
modelknn_3 = KNeighborsClassifier(n_neighbors=3)
modelknn_3.fit(X_train_3,Y_train_3)
knn_score_3 = modelknn_3.score(X_train_3,Y_train_3)
answer_3 = modelknn_3.predict(X_test_3)
##########Knn 6hr ##########
modelknn_6 = KNeighborsClassifier(n_neighbors=3)
modelknn_6.fit(X_train_6,Y_train_6)
knn_score_6 = modelknn_6.score(X_train_6,Y_train_6)
answer_6 = modelknn_6.predict(X_test_6)

# print("Knn score : ", knn_score)

# #############Logistic Regress ############# #เป็นวิธีการที่นิยมใช้เพื่อจำแนกประเภทข้อมูลหรือสิ่งของบางอย่างออกเป็นสองกลุ่ม
Lo_test = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=10000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)

logistic_regression= LogisticRegression(max_iter=10000, solver='saga')
logistic_regression_saga= LogisticRegression(max_iter=10000, solver='saga')
logistic_regression_sag= LogisticRegression(max_iter=10000, solver='sag')
logistic_regression_lbfgs= LogisticRegression(max_iter=10000, solver='lbfgs')
logistic_regression_newton= LogisticRegression(max_iter=10000, solver='newton-cg')
# logistic_regression.fit(X_train_o,Y_train_o)
# y_pred=logistic_regression.predict(X_test_o)

# # #############Logistic Regress with norm############# #เป็นวิธีการที่นิยมใช้เพื่อจำแนกประเภทข้อมูลหรือสิ่งของบางอย่างออกเป็นสองกลุ่ม แบบnorm
# logistic_regression_norm= LogisticRegression()
# logistic_regression_norm.fit(X_train_norm,Y_train_norm)
# y_pred=logistic_regression_norm.predict(X_test_norm)

#############Logistic Regress with new############ #เป็นวิธีการที่นิยมใช้เพื่อจำแนกประเภทข้อมูลหรือสิ่งของบางอย่างออกเป็นสองกลุ่ม 
logistic_regression_1_allset= Lo_test
logistic_regression_1_saga= logistic_regression_saga
logistic_regression_1_sag= logistic_regression_sag
logistic_regression_1_lbfgs= logistic_regression_lbfgs
logistic_regression_1_newton= logistic_regression_newton

logistic_regression_3_saga = logistic_regression_saga
logistic_regression_3_sag  = logistic_regression_sag
logistic_regression_3_lbfgs= logistic_regression_lbfgs
logistic_regression_3_newton= logistic_regression_newton
# logistic_regression_3.fit(X_train_3,Y_train_3)
# y_pred_3=logistic_regression_3.predict(X_test_3)

logistic_regression_6_saga = logistic_regression_saga
logistic_regression_6_sag = logistic_regression_sag
logistic_regression_6_lbfgs = logistic_regression_lbfgs
logistic_regression_6_newton = logistic_regression_newton
# logistic_regression_6.fit(X_train_6,Y_train_6)
# y_pred_6=logistic_regression_6.predict(X_test_6)





def get_score(model,xtrain, xtest, ytrain, ytest):  #function แสดงค่า acc ของ mddel 
    model.fit(xtrain,ytrain)
    return model.score(xtest, ytest)


# # # ############################## use def get score#############################

# # acc_scale = get_score(model_scale, X_train_std, X_test_std, Y_train_std, Y_test_std ) #Gau
# # acc_minmax = get_score(model_minmax,X_train_mm,X_test_mm,Y_train_mm,Y_test_mm) #Gau minmax

acc_ori = get_score(model_ori, X_train_o, X_test_o, Y_train_o, Y_test_o ) #Gau ori
# acc_norm = get_score(model_norm, X_train_norm, X_test_norm, Y_train_norm, Y_test_norm ) #Gau ori
acc_gau_1 = get_score(gau_model_1, X_train_1, X_test_1, Y_train_1, Y_test_1 ) #Gau 1hr
acc_gau_3 = get_score(gau_model_3, X_train_3, X_test_3, Y_train_3, Y_test_3 ) #Gau 3hr
acc_gau_6= get_score(gau_model_6, X_train_6, X_test_6, Y_train_6, Y_test_6 ) #Gau 6hr



# acc_svm = get_score(clf_o,X_train_o,X_test_o,Y_train_o,Y_test_o) #use SVM old dataset
# # acc_svm_norm = get_score(clf,X_train_norm,X_test_norm,Y_train_norm,Y_test_norm) #use SVM with norm
# acc_svm_1 = get_score(clf_1,X_train_1,X_test_1,Y_train_1,Y_test_1) #use SVM 1hr dataset
# acc_svm_3 = get_score(clf_3,X_train_3,X_test_3,Y_train_3,Y_test_3) #use SVM 3hr dataset
# acc_svm_6 = get_score(clf_6,X_train_6,X_test_6,Y_train_6,Y_test_6) #use SVM 6hr dataset

acc_knn_n2 = get_score(modelknn_n2,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n3 = get_score(modelknn_n3,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n4 = get_score(modelknn_n4,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n5 = get_score(modelknn_n5,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n6 = get_score(modelknn_n6,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n7 = get_score(modelknn_n7,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n8 = get_score(modelknn_n8,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n9 = get_score(modelknn_n9,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN
acc_knn_n10 = get_score(modelknn_n10,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN

acc_knn = get_score(modelknn,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN

# acc_knn_norm = get_score(modelknn_norm,X_train_norm,X_test_norm,Y_train_norm,Y_test_norm) #use KNN

acc_knn_1 = get_score(modelknn_1,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr
acc_knn_1_n2 = get_score(modelknn_n2,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n2
acc_knn_1_n3 = get_score(modelknn_n3,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n3
acc_knn_1_n4 = get_score(modelknn_n4,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n4
acc_knn_1_n5 = get_score(modelknn_n5,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n5
acc_knn_1_n6 = get_score(modelknn_n6,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n6
acc_knn_1_n7 = get_score(modelknn_n7,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n7
acc_knn_1_n8 = get_score(modelknn_n8,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n8
acc_knn_1_n9 = get_score(modelknn_n9,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n9
acc_knn_1_n10 = get_score(modelknn_n10,X_train_1,X_test_1,Y_train_1,Y_test_1) #use KNN 1hr n10


acc_knn_3 = get_score(modelknn_3,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n2 = get_score(modelknn_n2,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n3 = get_score(modelknn_n3,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n4 = get_score(modelknn_n4,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n5 = get_score(modelknn_n5,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n6 = get_score(modelknn_n6,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n7 = get_score(modelknn_n7,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n8 = get_score(modelknn_n8,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n9 = get_score(modelknn_n9,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr
acc_knn_3_n10 = get_score(modelknn_n10,X_train_3,X_test_3,Y_train_3,Y_test_3) #use KNN 3hr

acc_knn_6 = get_score(modelknn_6,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n2 = get_score(modelknn_n2,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n3 = get_score(modelknn_n3,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n4 = get_score(modelknn_n4,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n5 = get_score(modelknn_n5,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n6 = get_score(modelknn_n6,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n7 = get_score(modelknn_n7,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n8 = get_score(modelknn_n8,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n9 = get_score(modelknn_n9,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr
acc_knn_6_n10 = get_score(modelknn_n10,X_train_6,X_test_6,Y_train_6,Y_test_6) #use KNN 6hr


acc_logistic = get_score(logistic_regression,X_train_o,X_test_o,Y_train_o,Y_test_o)


# acc_logistic_norm = get_score(logistic_regression_norm,X_train_norm,X_test_norm,Y_train_norm,Y_test_norm)
acc_logistic_1_allset = get_score(logistic_regression_1_allset,X_train_1,X_test_1,Y_train_1,Y_test_1) #use logistic regress 1hr
acc_logistic_1_saga = get_score(logistic_regression_1_saga,X_train_1,X_test_1,Y_train_1,Y_test_1) #use logistic regress 1hr
acc_logistic_1_sag = get_score(logistic_regression_1_sag,X_train_1,X_test_1,Y_train_1,Y_test_1) #use logistic regress 1hr
acc_logistic_1_lbfgs = get_score(logistic_regression_1_lbfgs,X_train_1,X_test_1,Y_train_1,Y_test_1) #use logistic regress 1hr
acc_logistic_1_newton = get_score(logistic_regression_1_newton,X_train_1,X_test_1,Y_train_1,Y_test_1) #use logistic regress 1hr


acc_logistic_3_saga = get_score(logistic_regression_3_saga,X_train_3,X_test_3,Y_train_3,Y_test_3) #use logistic regress 3hr
acc_logistic_3_sag = get_score(logistic_regression_3_sag,X_train_3,X_test_3,Y_train_3,Y_test_3) #use logistic regress 3hr
acc_logistic_3_lbfgs = get_score(logistic_regression_3_lbfgs,X_train_3,X_test_3,Y_train_3,Y_test_3) #use logistic regress 3hr
acc_logistic_3_newton = get_score(logistic_regression_3_newton,X_train_3,X_test_3,Y_train_3,Y_test_3) #use logistic regress 3hr


acc_logistic_6_saga = get_score(logistic_regression_6_saga,X_train_6,X_test_6,Y_train_6,Y_test_6) #use logistic regress 6hr
acc_logistic_6_sag = get_score(logistic_regression_6_sag,X_train_6,X_test_6,Y_train_6,Y_test_6) #use logistic regress 6hr
acc_logistic_6_lbfgs = get_score(logistic_regression_6_lbfgs,X_train_6,X_test_6,Y_train_6,Y_test_6) #use logistic regress 6hr
acc_logistic_6_newton = get_score(logistic_regression_6_newton,X_train_6,X_test_6,Y_train_6,Y_test_6) #use logistic regress 6hr



# # # print("inputs is :", inputs)
# # # print("Norm here :", normalized_X)

# # # # kf = KFold(n_splits = 10, shuffle = True, random_state=200)


# # # # # rf_reg1 = RandomForestRegressor()
# # # # # scores = []
# # # # # for i in range(5):
# # # # #     result = next(kf.split(inputs), None)
# # # # #     X_train_s = inputs.iloc[result[0]]
# # # # #     X_test_s =  inputs.iloc[result[1]]
# # # # #     Y_train_s = target.iloc[result[0]]
# # # # #     Y_test_s = target.iloc[result[1]]
# # # # #     model_s = rf_reg1.fit(X_train_s,Y_train_s)
# # # # #     predictions = rf_reg1.predict(X_test_s)
# # # # #     scores.append(model_s.score(X_test_s,Y_test_s))
# # # # # print("Score for each ", scores)
# # # # # print("average score : ", np.mean(scores))
# # # # y_model = model.predict(X_test)
# # # # print(y_model)
# # # # print('predict and x_test')

# # # ############################# Print ACC ############################

print('---------------Accuracy Report------------')
print('')

print('--------------- old data--------------')
print('')

print('ค่า Accuracy with GaussianNB old :',acc_ori)
print('ค่า Accuracy with KNN old :',acc_knn)
print('ค่า Accuracy with KNN old n=2 :',acc_knn_n2)
print('ค่า Accuracy with KNN old n=3 :',acc_knn_n3)
print('ค่า Accuracy with KNN old n=4 :',acc_knn_n4)
print('ค่า Accuracy with KNN old n=5 :',acc_knn_n5)
print('ค่า Accuracy with KNN old n=6 :',acc_knn_n6)
print('ค่า Accuracy with KNN old n=7 :',acc_knn_n7)
print('ค่า Accuracy with KNN old n=8 :',acc_knn_n8)
print('ค่า Accuracy with KNN old n=9 :',acc_knn_n9)
print('ค่า Accuracy with KNN old n=10 :',acc_knn_n10)

print('ค่า Accuracy with logistic regress :',acc_logistic)

# print('')
# print('--------------- norm data--------------')
# print('')


# print('ค่า Accuracy with GaussianNB(& norm) :',acc_norm)

# print('ค่า Accuracy with KNN (& norm) :',acc_knn_norm)
# print('ค่า Accuracy with logistic regress(with norm) :',acc_logistic_norm)


print('')
print('--------------- 1hr data--------------')
print('')

print('ค่า Accuracy with GaussianNB 1hr data :',acc_gau_1)
# print('ค่า Accuracy with SVM 1hr data :',acc_svm_1)
print('ค่า Accuracy with KNN 1hr data :',acc_knn_1)
print('ค่า Accuracy with KNN 1hr data n=2 :',acc_knn_1_n2)
print('ค่า Accuracy with KNN 1hr data n=3 :',acc_knn_1_n3)
print('ค่า Accuracy with KNN 1hr data n=4 :',acc_knn_1_n4)
print('ค่า Accuracy with KNN 1hr data n=5 :',acc_knn_1_n5)
print('ค่า Accuracy with KNN 1hr data n=6 :',acc_knn_1_n6)
print('ค่า Accuracy with KNN 1hr data n=7 :',acc_knn_1_n7)
print('ค่า Accuracy with KNN 1hr data n=8 :',acc_knn_1_n8)
print('ค่า Accuracy with KNN 1hr data n=9 :',acc_knn_1_n9)
print('ค่า Accuracy with KNN 1hr data n=10 :',acc_knn_1_n10)

print('ค่า Accuracy with logistic regress 1hr allset :',acc_logistic_1_allset)
print('ค่า Accuracy with logistic regress 1hr saga :',acc_logistic_1_saga)
print('ค่า Accuracy with logistic regress 1hr sag :',acc_logistic_1_sag)
print('ค่า Accuracy with logistic regress 1hr lbfgs :',acc_logistic_1_lbfgs)
print('ค่า Accuracy with logistic regress 1hr newton :',acc_logistic_1_newton)


print('')
print('--------------- 3hr data--------------')
print('')

print('ค่า Accuracy with GaussianNB 3hr data :',acc_gau_3)

print('ค่า Accuracy with KNN 3hr data :',acc_knn_3)
print('ค่า Accuracy with KNN 3hr data n=2 :',acc_knn_3_n2)
print('ค่า Accuracy with KNN 3hr data n=3 :',acc_knn_3_n3)
print('ค่า Accuracy with KNN 3hr data n=4 :',acc_knn_3_n4)
print('ค่า Accuracy with KNN 3hr data n=5 :',acc_knn_3_n5)
print('ค่า Accuracy with KNN 3hr data n=6 :',acc_knn_3_n6)
print('ค่า Accuracy with KNN 3hr data n=7 :',acc_knn_3_n7)
print('ค่า Accuracy with KNN 3hr data n=8 :',acc_knn_3_n8)
print('ค่า Accuracy with KNN 3hr data n=9 :',acc_knn_3_n9)
print('ค่า Accuracy with KNN 3hr data n=10:',acc_knn_3_n10)
print('ค่า Accuracy with logistic regress 3hr saga :',acc_logistic_3_saga)
print('ค่า Accuracy with logistic regress 3hr sag :',acc_logistic_3_sag)
print('ค่า Accuracy with logistic regress 3hr lbfgs :',acc_logistic_3_lbfgs)
print('ค่า Accuracy with logistic regress 3hr newton :',acc_logistic_3_newton)


print('')
print('--------------- 6hr data--------------')
print('')

print('ค่า Accuracy with GaussianNB 6hr data :',acc_gau_6)
print('ค่า Accuracy with KNN 6hr data :',acc_knn_6)
print('ค่า Accuracy with KNN 6hr data n=2 :',acc_knn_6_n2)
print('ค่า Accuracy with KNN 6hr data n=3 :',acc_knn_6_n3)
print('ค่า Accuracy with KNN 6hr data n=4 :',acc_knn_6_n4)
print('ค่า Accuracy with KNN 6hr data n=5 :',acc_knn_6_n5)
print('ค่า Accuracy with KNN 6hr data n=6 :',acc_knn_6_n6)
print('ค่า Accuracy with KNN 6hr data n=7 :',acc_knn_6_n7)
print('ค่า Accuracy with KNN 6hr data n=8 :',acc_knn_6_n8)
print('ค่า Accuracy with KNN 6hr data n=9 :',acc_knn_6_n9)
print('ค่า Accuracy with KNN 6hr data n=10 :',acc_knn_6_n10)

print('ค่า Accuracy with logistic regress 6hr saga :',acc_logistic_6_saga)
print('ค่า Accuracy with logistic regress 6hr sag :',acc_logistic_6_sag)
print('ค่า Accuracy with logistic regress 6hr lbfgs :',acc_logistic_6_lbfgs)
print('ค่า Accuracy with logistic regress 6hr newton :',acc_logistic_6_newton)




# # # #######################################################################
# # # # print(model.fit(X_train,Y_train))

# # # # print(model1.fit(X_train1,Y_train1))

# # # # # print("------------------ y model ---------------")
# # # # print(y_model)

# # # # #  #######################################################


# # # # # model.score(X_test,Y_test)


# # # # # plt.scatter(df['Round'],df['Temperature'])  ##test plot Round & Temp  ### plot with dot graph

# # # # # ##############################################

# # # # #--------------show len data---------------------#
# # # # # print(len(X_train))
# # # # # print(len(X_test))

# # # # # # # len(inputs)
# # # # # # # print(len(inputs))

# # # # # print("---------------------------phase Data Type---------------------------")
# # # # # print(inputs.dtypes)

# # # # # is_time1 = inputs['time_only']
# # # # # is_temp = inputs['Temperature']

# # # # ######
# # # # ### โพยย #####
# # # # # start_r Round #xlabel function for  plt
# # # # # tmp_all  Temperature
# # # # # hu_all Humidity
# # # # # uv_all UVindex
# # # # # tmp_obj Object(*C)
# # # # # co_analog_all CO2Analog(ppm)
# # # # # co_pwm_all CO2PWM(ppm)

# # # # # plt.plot(start_r, hu_all )
# # # # # plt.plot(start_r, uv_all )
# # # # # plt.plot(start_r, tmp_obj )
# # # # # plt.plot(start_r, co_analog_all )
# # # # # plt.plot(start_r, co_pwm_all )


# # # # # ##-----------------------Graph plot zone -------------------------------##

# # # # # plt.figure(1)  #nice but not fig
# # # # # plt.plot(start_r, tmp_all )
# # # # # plt.title('กราฟแสดงอุณหภูมิ (°C)  ',fontname='Tahoma',fontsize='13') 
# # # # # plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
# # # # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # # # plt.savefig('./save/graph_tmp.png')


# # # # # plt.figure(2) #test
# # # # # plt.plot(start_r, uv_all )
# # # # # plt.title('กราฟแสดง UV  ',fontname='Tahoma',fontsize='13') 
# # # # # plt.ylabel('UV',fontname='Tahoma',fontsize='12')
# # # # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # # # plt.savefig('./save/graph_uv.png')


# # # # # plt.figure(3) #test3
# # # # # plt.plot(start_r, tmp_obj )
# # # # # plt.title('กราฟแสดง อุณหภูมิของใบ  ',fontname='Tahoma',fontsize='13') 
# # # # # plt.ylabel('temperature',fontname='Tahoma',fontsize='12')
# # # # # plt.xlabel('Queue ใน 4วัน ',fontname='Tahoma',fontsize='12')
# # # # # plt.savefig('./save/tem_obj.png')


# # # # # plt.figure(4) #test4
# # # # # plt.plot(start_r, co_analog_all )
# # # # # plt.title('กราฟแสดง ค่า CO2 แบบ Analog  ',fontname='Tahoma',fontsize='13') 
# # # # # plt.ylabel('carbon (ppm)',fontname='Tahoma',fontsize='12')
# # # # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # # # plt.savefig('./save/co_analog.png')

# # # # # plt.figure(5) #test5
# # # # # plt.plot(start_r, co_pwm_all )
# # # # # plt.title('กราฟแสดง ค่า CO2 แบบ pwm  ',fontname='Tahoma',fontsize='13') 
# # # # # plt.ylabel('carbon (ppm)',fontname='Tahoma',fontsize='12')
# # # # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # # # plt.savefig('./save/co_pwm.png')


# # # # # # plt.show()


# # # # # avg_tmp = tmp_all.mean(skipna = True)
# # # # # avg_hu = hu_all.mean(skipna = True)
# # # # # avg_uv = uv_all.mean(skipna = True)
# # # # avg_tmp_obj = tmp_obj.mean(skipna = True)
# # # # avg_co2_ppm = co_analog_all.mean(skipna = True)
# # # # avg_co2_pwm = co_pwm_all.mean(skipna = True)


# # # # print("----------------------- Summary Data 9-12 Feb 2020 ------------------------- \n" )

# # # # # print( "Temperature average is :" , round(float(avg_tmp),2) , "°C")
# # # # # print( "Humidity average is :" , round(float(avg_hu),2) , "%")
# # # # # print( "UV index average is :" , round(float(avg_uv),2) , " ")
# # # # print( "Object Temp average is :" , round(float(avg_tmp_obj),2) , "°C")
# # # # print( "Carbon dioxide(PPM) average is :" , round(float(avg_co2_ppm),2) , "ppm")
# # # # print( "Carbon dioxide(PWM) average is :" , round(float(avg_co2_pwm),2) , "ppm")

# # # # print()
# # # # print()

# # # # # print("---------------------------------------------------------")
# # # # print("------------------ Accuracy ---------------")

# # # # print("Accuracy Gauis :", acc_UV )

# # # # print()
# # # # print()

# # # # print("------------------ Accuracy Without UVindex  ---------------")

# # # # print("Accuracy is :", acc_wotemp )

# # # # print()
# # # # print()


# # # # print("------")
# # # # print("acc_Kneighbor : ",acc_nb)
# # # # print("------")




# # # print("\n"," -*****************End Demo **** ")

# # # # print(df.info())