import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# 原始数据
data = [
    "2020-01-01 729562946", "2020-02-01 279143421", "2020-03-01 435329481",
    "2020-04-01 786571801", "2020-05-01 903997110", "2020-06-01 840090138",
    "2020-07-01 816355506", "2020-08-01 774299120", "2020-09-01 1001876701",
    "2020-10-01 951567026", "2020-11-01 874078726", "2020-12-01 951266475",
    "2021-01-01 742834968", "2021-02-01 381498848", "2021-03-01 661866864",
    "2021-04-01 675509918", "2021-05-01 649804679", "2021-06-01 631873003",
    "2021-07-01 695154746", "2021-08-01 561664804", "2021-09-01 544027993",
    "2021-10-01 474355467", "2021-11-01 594087514", "2021-12-01 901256012",
    "2022-01-01 1258422202", "2022-02-01 752106702", "2022-03-01 1102262515",
    "2022-04-01 1440799940", "2022-05-01 1743854941", "2022-06-01 1648362981",
    "2022-07-01 1585041433", "2022-08-01 1336964387", "2022-09-01 1797316541",
    "2022-10-01 1803685830", "2022-11-01 1650954900", "2022-12-01 1370677617",
    "2023-01-01 845525895", "2023-02-01 773442906", "2023-03-01 1271774938",
    "2023-04-01 1085226686", "2023-05-01 1164556412", "2023-06-01 1104611864",
    "2023-07-01 1015793859", "2023-08-01 992130832", "2023-09-01 1118749018",
    "2023-10-01 981392291", "2023-11-01 1060125537", "2023-12-01 921126534",
    "2024-01-01 1170922845", "2024-02-01 1001193929", "2024-03-01 1071379123",
    "2024-04-01 1267961877", "2024-05-01 1179995939", "2024-06-01 1077022567",
    "2024-07-01 1124663223", "2024-08-01 1355701217", "2024-09-01 1671086388",
    "2024-10-01 1626066444", "2024-11-01 1744341183", "2024-12-01 1829768301"
]

# 提取日期和数值
dates = []
values = []
for item in data:
    date, value = item.split()
    dates.append(pd.to_datetime(date))
    values.append(int(value))

# 创建时间序列
ts = pd.Series(values, index=dates)

# 进行ADF检验
adf_result = adfuller(ts)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# 输出ADF检验结果
print('ADF Statistic: {}'.format(adf_result[0]))
print('ADF p-value: {}'.format(adf_result[1]))
print('ADF Critical Values:')
for key, value in adf_result[4].items():
    print('\t{}: {}'.format(key, value))

# 根据ADF p值判断平稳性
if adf_result[1] <= 0.05:
    print("ADF检验结果：数据平稳")
else:
    print("ADF检验结果：数据非平稳")

# 进行KPSS检验
kpss_result = kpss(ts)

# 输出KPSS检验结果
print('\nKPSS Statistic: {}'.format(kpss_result[0]))
print('KPSS p-value: {}'.format(kpss_result[1]))
print('KPSS Critical Values:')
for key, value in kpss_result[3].items():
    print('\t{}: {}'.format(key, value))

# 根据KPSS p值判断平稳性
if kpss_result[1] > 0.05:
    print("KPSS检验结果：数据平稳")
else:
    print("KPSS检验结果：数据非平稳")

# 绘制时间序列图
plt.figure(figsize=(12, 6))
ts.plot()
plt.title('时间序列图')
plt.xlabel('日期')
plt.ylabel('数值')
plt.grid(True)
plt.show()
