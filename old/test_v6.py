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



data = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdate.xlsx', sheet_name='Alldata') 
# data = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\clean\march\testdata_badgood.xlsx') 
# data_9 = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\clean\data_allday.xlsx','9-02-2020') 
data.to_csv (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdate_new.csv', index = None, header=True)


# df_filter =pd.DataFrame(data, columns= ['Round'	,'Date:time',	'Temperature'	,'Pressure'	,'Humidity'	,'Altitude',	'Vis',	'IR',	'UVindex',	'Ambient(*C)',	'Object(*C)',	'Ambient(*F)',	'Object(*F)',	'CO2Serial(ppm)',	'CO2Analog(ppm)',	'CO2PWM(ppm)',	'isGood',	'Xbar_Humidity',
# 	'Xbar_Altitude',	'Xbar_Vis',	'Xbar_IR',	'Xbar_UVindex',	'Xbar_Ambient(*C)',	'Xbar_Object(*C)',	
#     'Xbar_Ambient(*F)',	'Xbar_Object(*F)',	'Xbar_co2analog',	'SD_Humidity',	'SD_Altitude',	'SD_Vis',	
#     'SD_IR',	'SD_UVindex',	'SD_Ambient(*C)',	'SD_Object(*C)',	'SD_Ambient(*F)',	'SD_Object(*F)',	
#     'SD_co2analog',	'Mode_Humidity'	,'Mode_Vis'	,'Mode_IR'	,'Mode_Ambient(*C)',	'Mode_Object(*C)',	
#     'Mode_Ambient(*F)',	'Mode_Object(*F)',	'Mode_co2analog'])

df_filter = pd.DataFrame(data, columns= ['Temperature'	,'Humidity'	,'Object(*C)','CO2Analog(ppm)',
                        'CO2PWM(ppm)',	'isGood',	'Xbar_Humidity', 'Xbar_co2analog',	'SD_Humidity',
                        'SD_co2analog',	'Mode_Humidity'	,	'Mode_co2analog'])

df = pd.DataFrame(data, columns= ['Humidity',	'UVindex', 'Temperature',	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)','isGood' ])
df = df.fillna(0) #ดัก missing value ถ้าเจอ เติม 0
# df = df.dropna(axis=0)  #case missing value drop ทิ้งเลย
df.loc[df['isGood'] < 1, 'labeled'] = '0' 
df.loc[df['isGood'] >= 1, 'labeled'] = '1' 

df = df.dropna(axis=0)
############################################

df_filter.loc[df_filter['isGood'] < 1, 'labeled'] = '0' 
df_filter.loc[df_filter['isGood'] >= 1, 'labeled'] = '1' 


target=df.labeled


df_filter = df_filter.dropna(axis=0)  #case missing value drop ทิ้งเลย

target_filter =df_filter.labeled  #targetdata_new



inputs=df.drop('isGood', axis='columns')
input_all = df_filter.drop('isGood', axis='columns')  #featuredata_new

# print(input_all)

# print(df_filter)
# print(df_filter.shape)
# # test = df.head()
# df['some_column'].plot(figsize=(10, 5))

# print(df.class_day)

# print(data_norm)
tmp_obj = df.get('Object(*C)')
co_analog_all = df.get('CO2Analog(ppm)')
co_pwm_all = df.get('CO2PWM(ppm)')

xbar_hu = df_filter.get('Xbar_Humidity')

# print(xbar_hu)

# print(xbar_hu.shape)



def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*100    #อิงค่า dataset[?]ให้อยู่ในช่วง 0-100 ค่าอื่นก็ลดหย่อนเป็นสัดส่วน
    dataNorm["CO2PWM(ppm)"]=dataset["CO2PWM(ppm)"]
    # dataNorm["CO2Analog(ppm)"]=dataset["CO2Analog(ppm)"]
    return dataNorm


data_norm = normalize(data)

df_n =pd.DataFrame(data_norm, columns= [  'Humidity', 	'Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)' ])

df_n.loc[df_n['Humidity'] < 1, 'spe'] = '0' 
input_norm = df_n.drop('spe', axis='columns')

# print(input_norm)


input_all.drop('labeled', axis='columns',inplace=True)   #drop ของดาต้าใหม่

# print(input_all)

# print(target_filter)



inputs.drop('labeled', axis='columns',inplace=True)   #drop ตัวที่ไม่ใช้ออกไปอีก
inputs.drop('UVindex', axis='columns',inplace=True)   #drop ตัวที่ไม่ใช้ออกไปอีก

# inputs = ฟีเจอร์ที่จะเอาเข้าไปเทรน



# print(df)
# print(inputs)

inputs2=df.drop('Object(*C)', axis='columns',inplace=True)          #get inputs2 to make date sheet without UVindex
inputs2=df.drop('UVindex', axis='columns')          #get inputs2 to make date sheet without UVindex
inputs2.drop('isGood', axis='columns',inplace=True)   #drop ตัวที่ไม่ใช้ออกไปอีก
inputs2.drop('labeled', axis='columns',inplace=True)   #drop ตัวที่ไม่ใช้ออกไปอีก

# inputs2 = ฟีเจอร์ที่จะเอาเข้าไปเทรนโดยไม่มีค่า tmp

####################phase แสดงผล ค่าต่างๆในตารางก่อนนำไปเทรน และ เทส#####################################



# print("-----------------------------------after del class_Day& labeled--------------------------")
# print(inputs)   #use inputs to x train
# print("-----------------------------------table w_out UVindex--------------------------")
# print(inputs2)   #use inputs to x train



# # print("---------------------1---------------------")

# print(inputs)





# # ############ close tag after set file #############

# writer = pd.ExcelWriter('output_goodbadv2.xlsx')
# target.to_excel(writer,'1isgod0isbad')
# inputs.to_excel(writer,'feature_data')
# inputs2.to_excel(writer,'featuredata_without_tmp')
# input_norm.to_excel(writer,'feature_withnorm')
# writer.save()


# writer = pd.ExcelWriter('output_alldata.xlsx')
# target_filter.to_excel(writer,'1isgod0isbad')
# input_all.to_excel(writer,'feature')
# inputs2.to_excel(writer,'featuredata_without_tmp')
# # input_norm.to_excel(writer,'feature_withnorm')
# writer.save()


# standard
standardized_X = preprocessing.scale(inputs)

# # ###################################### Prediction Zone #######################################



# # X_train_scale, X_test_scale, Y_train_scale, Y_test_scale = train_test_split(x_scaled,target,test_size=0.2,random_state=0) #all  scale and have tmp
X_train_std, X_test_std, Y_train_std, Y_test_std = train_test_split(standardized_X,target,test_size=0.2,random_state=0) #all  standardized and have tmp
# X_train_mm, X_test_mm, Y_train_mm, Y_test_mm =train_test_split(X_train_minmax,target,test_size=0.2,random_state=0) #all  minmax and have tmp


X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(input_all,target_filter,test_size=0.2,random_state=0) #all  norm and have tmp
X_train_o, X_test_o, Y_train_o, Y_test_o =train_test_split(inputs,target,test_size=0.2,random_state=0) #all with tmp


X_train1, X_test1, Y_train1, Y_test1 =train_test_split(inputs2,target,test_size=0.2,random_state=0) # all with out tmp


# #data preprocess with scale
# # print("- test min max scaler -")
# # print(X_train_minmax)

########## SVM method ###############
clf = svm.SVR()
clf.fit(X_train_norm, Y_train_norm)
SVR()
svmmodel = clf.predict(X_test_norm)
svmscore = clf.score(X_train_norm,Y_train_norm)
# print("SVM score: ",svmscore)

#############Gau with scale #############
model_scale= GaussianNB()
model_scale.fit(X_train_std,Y_train_std)
y_model = model_scale.predict(X_test_std)

# #############Gau scale minmax################
# model_minmax= GaussianNB()
# model_minmax.fit(X_train_mm,Y_train_mm)
# y_model_minmax = model_minmax.predict(X_test_mm)

# #############Gau Ori###########
model_ori= GaussianNB()
model_ori.fit(X_train_o,Y_train_o)
y_model_o = model_ori.predict(X_test_o)
# ###########Knn ##########
modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X_train_o,Y_train_o)
knn_score = modelknn.score(X_train_o,Y_train_o)
answer = modelknn.predict(X_test_o)
# print("Knn score : ", knn_score)
#############Gau wotmp#############
model1= GaussianNB()
model1.fit(X_train1,Y_train1)
y_model1 = model1.predict(X_test1)
#############Gau norm#############
model_norm= GaussianNB()
model_norm.fit(X_train_norm,Y_train_norm)
y_model_norm = model_norm.predict(X_test_norm)

#############Logistic Regress ############# #เป็นวิธีการที่นิยมใช้เพื่อจำแนกประเภทข้อมูลหรือสิ่งของบางอย่างออกเป็นสองกลุ่ม
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train_o,Y_train_o)
y_pred=logistic_regression.predict(X_test_o)

#############Logistic Regress with norm############# #เป็นวิธีการที่นิยมใช้เพื่อจำแนกประเภทข้อมูลหรือสิ่งของบางอย่างออกเป็นสองกลุ่ม แบบnorm
logistic_regression_norm= LogisticRegression()
logistic_regression_norm.fit(X_train_norm,Y_train_norm)
y_pred=logistic_regression_norm.predict(X_test_norm)





def get_score(model,xtrain, xtest, ytrain, ytest):  #function แสดงค่า acc ของ mddel 
    model.fit(xtrain,ytrain)
    return model.score(xtest, ytest)



# # print("Report knn :"
# #         ,classification_report(Y_test, answer))
# # print("\n")
# # print("Report naive Bayes :" ,classification_report(Y_test, y_model))

# # from sklearn.metrics import classification_report
# # print("report ", classification_report(Y_test, y_model))


# # Whisacc = accuracy_score(Y_test,y_model)

# # print(Whisacc)
# # ################################

# # normalizer = preprocessing.Normalizer().fit(inputs)

# # print("norm here", normalizer)



# # model_nb = KNeighborsClassifier(n_neighbors=1)
# # model_nb.fit(inputs, target)
# # y_model_nb = model_nb.predict(inputs)
# # acc_nb = accuracy_score(target, y_model_nb)

# # clf = svm.SVC(kernel='linear', C=1, random_state=0)
# # gau = GaussianNB()
# # scores = cross_val_score(clf, inputs, target, cv=5)
# # scores1 = cross_val_score(gau, inputs, target, cv=5)


# # print("\n"," -*****************Cross validation **** ")


# # print("----------------SVM----------")
# # print('cross of SVM :',scores)
# # print("\n/","----------------GaussianNB----------")
# # print('cross of Gau :',scores1)

# # print('')
# # cross_model_svm = scores[4]
# # cross_model = scores1[3]
# # print('')

# # print("Best acc SVM is :",cross_model_svm)
# # print("Best acc Gau is :",cross_model)

# # print("**********************************************")




# # kf = KFold(n_splits = 10, shuffle = True, random_state=200)
# # rf_reg = RandomForestRegressor()
# # rf_reg.fit(X_train,Y_train)
# # y_pred = rf_reg.predict(X_test)

# # rsqure_score = rf_reg.score(X_train,Y_train)
# # rsqure_score_test = rf_reg.score(X_test,Y_test)


# # cv_r2_scores_rf = cross_val_score(rf_reg, inputs, target, cv=5,scoring='r2')  #cross validation
# # print("---------")
# # print(rsqure_score)
# # print("---------")

# # print(cv_r2_scores_rf)

# # print("---------")


# # normalized_X = preprocessing.normalize(inputs)
# # ############################## use def get score#############################

acc_scale = get_score(model_scale, X_train_std, X_test_std, Y_train_std, Y_test_std ) #Gau
# acc_minmax = get_score(model_minmax,X_train_mm,X_test_mm,Y_train_mm,Y_test_mm) #Gau minmax

acc_ori = get_score(model_ori, X_train_o, X_test_o, Y_train_o, Y_test_o ) #Gau ori
acc_norm = get_score(model_norm, X_train_norm, X_test_norm, Y_train_norm, Y_test_norm ) #Gau ori

acc_wotemp = get_score(model1, X_train1, X_test1, Y_train1, Y_test1 ) #Gau wo tmp
acc_svm = get_score(clf,X_train_o,X_test_o,Y_train_o,Y_test_o) #use SVM
acc_svm_norm = get_score(clf,X_train_norm,X_test_norm,Y_train_norm,Y_test_norm) #use SVM
acc_knn = get_score(modelknn,X_train_o,X_test_o,Y_train_o,Y_test_o) #use KNN

acc_logistic = get_score(logistic_regression,X_train_o,X_test_o,Y_train_o,Y_test_o)
acc_logistic_norm = get_score(logistic_regression_norm,X_train_norm,X_test_norm,Y_train_norm,Y_test_norm)

# acc_notscale = get_score(model_or,X_train_o,X_test_o,Y_train_o,Y_test_o)

# print("inputs is :", inputs)
# print("Norm here :", normalized_X)

# # kf = KFold(n_splits = 10, shuffle = True, random_state=200)


# # # rf_reg1 = RandomForestRegressor()
# # # scores = []
# # # for i in range(5):
# # #     result = next(kf.split(inputs), None)
# # #     X_train_s = inputs.iloc[result[0]]
# # #     X_test_s =  inputs.iloc[result[1]]
# # #     Y_train_s = target.iloc[result[0]]
# # #     Y_test_s = target.iloc[result[1]]
# # #     model_s = rf_reg1.fit(X_train_s,Y_train_s)
# # #     predictions = rf_reg1.predict(X_test_s)
# # #     scores.append(model_s.score(X_test_s,Y_test_s))
# # # print("Score for each ", scores)
# # # print("average score : ", np.mean(scores))
# # y_model = model.predict(X_test)
# # print(y_model)
# # print('predict and x_test')

# # ############################# Print ACC ############################

print('---------------Accuracy Report------------')
print('')
# print('')
# print('ค่า Accuracy โดยที่ scale :',acc_scale)
# print('')
# print('ค่า Accuracy โดยnorm minmax :',acc_minmax)
# print('')
print('ค่า Accuracy with GaussianNB โดยที่มีค่า temp อยู่ :',acc_ori)

print('')
print('ค่า Accuracy with GaussianNB(& norm) :',acc_norm)
print('')
print('ค่า Accuracy with GaussianNB โดยที่ไม่มีปัจจัยภายนอก :',acc_wotemp)
print('')
print('ค่า Accuracy with SVM :',acc_svm)
print('')
print('ค่า Accuracy with SVM (&norm) :',acc_svm_norm)
print('')
print('ค่า Accuracy with KNN :',acc_knn)
print('')
print('ค่า Accuracy with logistic regress :',acc_logistic)
print('')
print('ค่า Accuracy with logistic regress(with norm) :',acc_logistic_norm)
print('')


# # print('ค่า Accuracy โดยที่ไม่scale :',acc_notscale)
# # print('')


# #######################################################################
# # print(model.fit(X_train,Y_train))

# # print(model1.fit(X_train1,Y_train1))

# # # print("------------------ y model ---------------")
# # print(y_model)

# # #  #######################################################


# # # model.score(X_test,Y_test)


# # # plt.scatter(df['Round'],df['Temperature'])  ##test plot Round & Temp  ### plot with dot graph

# # # ##############################################

# # #--------------show len data---------------------#
# # # print(len(X_train))
# # # print(len(X_test))

# # # # # len(inputs)
# # # # # print(len(inputs))

# # # print("---------------------------phase Data Type---------------------------")
# # # print(inputs.dtypes)

# # # is_time1 = inputs['time_only']
# # # is_temp = inputs['Temperature']

# # ######
# # ### โพยย #####
# # # start_r Round #xlabel function for  plt
# # # tmp_all  Temperature
# # # hu_all Humidity
# # # uv_all UVindex
# # # tmp_obj Object(*C)
# # # co_analog_all CO2Analog(ppm)
# # # co_pwm_all CO2PWM(ppm)

# # # plt.plot(start_r, hu_all )
# # # plt.plot(start_r, uv_all )
# # # plt.plot(start_r, tmp_obj )
# # # plt.plot(start_r, co_analog_all )
# # # plt.plot(start_r, co_pwm_all )


# # # ##-----------------------Graph plot zone -------------------------------##

# # # plt.figure(1)  #nice but not fig
# # # plt.plot(start_r, tmp_all )
# # # plt.title('กราฟแสดงอุณหภูมิ (°C)  ',fontname='Tahoma',fontsize='13') 
# # # plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
# # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # plt.savefig('./save/graph_tmp.png')


# # # plt.figure(2) #test
# # # plt.plot(start_r, uv_all )
# # # plt.title('กราฟแสดง UV  ',fontname='Tahoma',fontsize='13') 
# # # plt.ylabel('UV',fontname='Tahoma',fontsize='12')
# # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # plt.savefig('./save/graph_uv.png')


# # # plt.figure(3) #test3
# # # plt.plot(start_r, tmp_obj )
# # # plt.title('กราฟแสดง อุณหภูมิของใบ  ',fontname='Tahoma',fontsize='13') 
# # # plt.ylabel('temperature',fontname='Tahoma',fontsize='12')
# # # plt.xlabel('Queue ใน 4วัน ',fontname='Tahoma',fontsize='12')
# # # plt.savefig('./save/tem_obj.png')


# # # plt.figure(4) #test4
# # # plt.plot(start_r, co_analog_all )
# # # plt.title('กราฟแสดง ค่า CO2 แบบ Analog  ',fontname='Tahoma',fontsize='13') 
# # # plt.ylabel('carbon (ppm)',fontname='Tahoma',fontsize='12')
# # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # plt.savefig('./save/co_analog.png')

# # # plt.figure(5) #test5
# # # plt.plot(start_r, co_pwm_all )
# # # plt.title('กราฟแสดง ค่า CO2 แบบ pwm  ',fontname='Tahoma',fontsize='13') 
# # # plt.ylabel('carbon (ppm)',fontname='Tahoma',fontsize='12')
# # # plt.xlabel('Queue ใน 4 วัน ',fontname='Tahoma',fontsize='12')
# # # plt.savefig('./save/co_pwm.png')


# # # # plt.show()


# # # avg_tmp = tmp_all.mean(skipna = True)
# # # avg_hu = hu_all.mean(skipna = True)
# # # avg_uv = uv_all.mean(skipna = True)
# # avg_tmp_obj = tmp_obj.mean(skipna = True)
# # avg_co2_ppm = co_analog_all.mean(skipna = True)
# # avg_co2_pwm = co_pwm_all.mean(skipna = True)


# # print("----------------------- Summary Data 9-12 Feb 2020 ------------------------- \n" )

# # # print( "Temperature average is :" , round(float(avg_tmp),2) , "°C")
# # # print( "Humidity average is :" , round(float(avg_hu),2) , "%")
# # # print( "UV index average is :" , round(float(avg_uv),2) , " ")
# # print( "Object Temp average is :" , round(float(avg_tmp_obj),2) , "°C")
# # print( "Carbon dioxide(PPM) average is :" , round(float(avg_co2_ppm),2) , "ppm")
# # print( "Carbon dioxide(PWM) average is :" , round(float(avg_co2_pwm),2) , "ppm")

# # print()
# # print()

# # # print("---------------------------------------------------------")
# # print("------------------ Accuracy ---------------")

# # print("Accuracy Gauis :", acc_UV )

# # print()
# # print()

# # print("------------------ Accuracy Without UVindex  ---------------")

# # print("Accuracy is :", acc_wotemp )

# # print()
# # print()


# # print("------")
# # print("acc_Kneighbor : ",acc_nb)
# # print("------")




# print("\n"," -*****************End Demo **** ")

# # print(df.info())