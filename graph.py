import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from openpyxl import Workbook

# C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file
delta_good =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdatemind.xlsx', sheet_name='good')
delta_bad =pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdatemind.xlsx', sheet_name='bad')
data = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdate.xlsx', sheet_name='bad') 
data2 = pd.read_excel (r'C:\Users\OhoMindZa\Documents\jan-pro\part-ml\dlspot-code\data-file\setdate.xlsx', sheet_name='good') 


dfbad = pd.DataFrame(delta_bad, columns= ['Round','D_Co2analog(1h)'])
dfgood = pd.DataFrame(delta_good, columns= ['Round','D_Co2analog(1h)'])
dfbad = dfbad.dropna(axis=0)
dfgood = dfgood.dropna(axis=0)

# print(dfbad)
# print(dfbad.info())


# print(dfgood)
# print(dfgood.info())

# deltabadtree = dfbad.get('D_Co2analog(1h)')
# deltagoodtree = dfgood.get('D_Co2analog(1h)')
# r1 = dfbad.get('Round')
# r2 = dfgood.get('Round')


df_1 = pd.DataFrame(data, columns= ['Round','Humidity',	'Temperature','Vis','Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)','isGood' ])  #bad
df_2 = pd.DataFrame(data2, columns= ['Round','Humidity','Temperature','Vis','Object(*C)', 'CO2PWM(ppm)', 'CO2Analog(ppm)','isGood' ]) #good 



df_bad = pd.DataFrame(data, columns= ['Round','Humidity',	'Temperature','Vis','Object(*C)','Xbar_co2analog'	,'SD_co2analog'	,'Mode_co2analog'])
df_good = pd.DataFrame(data2, columns= ['Round','Humidity',	'Temperature','Vis','Object(*C)','Xbar_co2analog'	,'SD_co2analog'	,'Mode_co2analog'])

dfClean = df_bad.dropna(axis=0)  #case missing value drop ทิ้งเลย
dfClean_good = df_good.dropna(axis=0)  #case missing value drop ทิ้งเลย

# print(df_1)
# # print("----------------------------------")
# # print(df_2)
# # print(df_e.head())
# print(dfClean)
# print('')
# print(dfClean_good)

# print(data.head())



# ###########new features #################
xbarco_analog_bad = dfClean.get('Xbar_co2analog')
sdco_analog_bad = dfClean.get('SD_co2analog')
modeco_analog_bad = dfClean.get('Mode_co2analog')
xla = dfClean.get('Round')

xbarco_analog_good = dfClean_good.get('Xbar_co2analog')
sdco_analog_good = dfClean_good.get('SD_co2analog')
modeco_analog_good = dfClean_good.get('Mode_co2analog')
xla_good = dfClean_good.get('Round')

# ###########Bad#################
start_bad =df_1.get('Round')
tmp_bad =df_1.get('Temperature')
vis_bad =df_1.get('Vis')
tmp_obj_bad = df_1.get('Object(*C)')
co_analog_bad = df_1.get('CO2Analog(ppm)')
co_pwm_bad = df_1.get('CO2PWM(ppm)')
###########good#################
start_good =df_2.get('Round')
tmp_good =df_2.get('Temperature')
vis_good =df_2.get('Vis')
tmp_obj_good = df_2.get('Object(*C)')
co_analog_good = df_2.get('CO2Analog(ppm)')
co_pwm_good = df_2.get('CO2PWM(ppm)')





# plt.figure()  #good with delta
# plt.plot(r1, deltagoodtree )
# plt.xlim([ 0,4000])
# plt.ylim([-81,200])
# plt.title('กราฟแสดงdelta  (ต้นดี) ',fontname='Tahoma',fontsize='13') 
# plt.ylabel('Co2analog(ppm)',fontname='Tahoma',fontsize='12')
# plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
# plt.savefig('./graph_Delta_good.png')

# plt.figure()  #bad with delta
# plt.plot(r1, deltabadtree )
# plt.title('กราฟแสดงdelta  (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
# plt.ylabel('Co2analog(ppm)',fontname='Tahoma',fontsize='12')
# plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
# plt.savefig('./graph_Delta_bad.png')



# # # # ##-----------------------Graph plot zone -------------------------------##


plt.figure()  #good&bad with Xbar
plt.plot(xla, xbarco_analog_bad ,label='Xbar_Badtree')
plt.plot(xla_good, xbarco_analog_good ,label='Xbar_Goodtree')
plt.title('กราฟแสดงXbarค่าco2analog ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2analog(ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.legend(framealpha=1, frameon=True);
plt.savefig('./pic/bad/graph_XD_badandgood.png')



plt.figure()  #bad with Xbar
plt.plot(xla, xbarco_analog_bad )
plt.title('กราฟแสดงXbar  (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2analog(ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_XD_bad.png')

plt.figure()  #bad with SD
plt.plot(xla, sdco_analog_bad )
plt.title('กราฟแสดงSD (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('co2analog ',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_SD_bad.png')

plt.figure()  #bad with mode
plt.plot(xla, modeco_analog_bad )
plt.title('กราฟแสดงMode (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('co2analog (ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_Mode_bad.png')

plt.figure()  #good with Xbar
plt.plot(xla_good, xbarco_analog_good )
plt.title('กราฟแสดงXbar  (ต้นดี) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2analog(ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_XD_good.png')

plt.figure()  #good with SD
plt.plot(xla_good, sdco_analog_good )
plt.title('กราฟแสดงSD (ต้นดี) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('co2analog ',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_SD_good.png')

plt.figure()  #good with mode
plt.plot(xla_good, modeco_analog_good )
plt.title('กราฟแสดงMode (ต้นดี) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('co2analog (ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_Mode_good.png')

plt.figure(1)  #bad with tmp
plt.plot(start_bad, tmp_bad )
plt.title('กราฟแสดงอุณหภูมิ (°C) (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_tmp_bad.png')

plt.figure(2)  #good with tmp
plt.plot(start_good, tmp_good )
plt.title('กราฟแสดงอุณหภูมิ (°C) (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_tmp_good.png')


plt.figure(3)  #bad with Vis
plt.plot(start_bad, vis_bad )
plt.title('กราฟแสดงค่าแสง (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Vis',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_vis_bad.png')

plt.figure(4)  #good with Vis
plt.plot(start_good, vis_good )
plt.title('กราฟแสดงอุณหภูมิ (°C) (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Vis',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_vis_good.png')

plt.figure(5)  #bad with Vis
plt.plot(start_bad, vis_bad )
plt.title('กราฟแสดงค่าแสง (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Vis',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_vis_bad.png')

plt.figure(6)  #good with Vis
plt.plot(start_good, vis_good )
plt.title('กราฟแสดงค่าแสง (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Vis',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_vis_good.png')


plt.figure(7)  #bad with tmpobj
plt.plot(start_bad, tmp_obj_bad )
plt.title('กราฟแสดงอุณหภูมิของต้น (°C) (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_tmp_obj_bad.png')

plt.figure(8)  #good with tmpobj
plt.plot(start_good, tmp_obj_good )
plt.title('กราฟแสดงอุณหภูมิของต้น (°C) (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Temp (°C)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_tmp_obj_good.png')

plt.figure(9)  #bad with co_analog
plt.plot(start_bad, co_analog_bad )
plt.title('กราฟแสดง ค่า CO2 แบบ Analog (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2analog (ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_co2analog_bad.png')

plt.figure(10)  #good with co_analog
plt.plot(start_good, co_analog_good )
plt.title('กราฟแสดง ค่า CO2 แบบ Analog (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2analog (ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_co2analog_good.png')

plt.figure(11)  #bad with co_pwm
plt.plot(start_bad, co_pwm_bad )
plt.title('กราฟแสดง ค่า CO2 แบบ pwm (ต้นป่วย) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2pwm (ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/bad/graph_co2pwm_bad.png')

plt.figure(12)  #good with co_pwm
plt.plot(start_good, co_pwm_good )
plt.title('กราฟแสดง ค่า CO2 แบบ pwm (ต้นปกติ) ',fontname='Tahoma',fontsize='13') 
plt.ylabel('Co2pwm(ppm)',fontname='Tahoma',fontsize='12')
plt.xlabel('Round ',fontname='Tahoma',fontsize='12')
plt.savefig('./pic/good/graph_co2pwm_good.png')


# # plt.show()