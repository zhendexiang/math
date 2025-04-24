# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# --------------- 1. 修复中文字体问题 ---------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# --------------- 2. 数据读取 ---------------
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

df = pd.DataFrame([row.split() for row in data], columns=["Date", "Value"])
df["Date"] = pd.to_datetime(df["Date"])
df["Value"] = df["Value"].astype(float)
df.set_index("Date", inplace=True)
ts = df["Value"].asfreq('MS')  # 明确设置频率为月度

# --------------- 3. 可视化原始序列 ---------------
plt.figure(figsize=(12, 6))
plt.plot(ts, label="原始序列")
plt.title("人民币月度数据趋势")
plt.xlabel("日期")
plt.ylabel("数值")
plt.legend()
plt.grid(True)
plt.show()

# --------------- 4. 二阶差分处理 ---------------
diff1 = ts.diff().dropna()
diff2 = diff1.diff().dropna()

# ADF检验
def check_stationary(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

print("二阶差分序列ADF检验:")
check_stationary(diff2)

# --------------- 5. ACF/PACF分析 ---------------
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(diff2, lags=20, ax=plt.gca(), title="ACF (二阶差分后)")
plt.subplot(122)
plot_pacf(diff2, lags=20, ax=plt.gca(), title="PACF (二阶差分后)")
plt.tight_layout()
plt.show()

# --------------- 6. 构建ARIMA模型 ---------------
# 修复参数：明确指定 order=(1,2,1)
model = ARIMA(ts, order=(1, 2, 1))  # 确保参数正确
results = model.fit()

# 输出模型摘要
print("\n模型参数和显著性检验:")
print(results.summary())

# --------------- 7. 残差诊断 ---------------
residuals = results.resid
plt.figure(figsize=(12, 4))
plot_acf(residuals, lags=20, title="残差ACF图")
plt.show()

# Ljung-Box检验
lb_test = acorr_ljungbox(residuals, lags=[10])
print(f"\nLjung-Box检验p-value: {lb_test.iloc[0, 1]}")

# --------------- 8. 预测未来12个月 ---------------
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

forecast_df = pd.DataFrame({
    "预测值": forecast_mean,
    "下限": forecast_conf_int.iloc[:, 0],
    "上限": forecast_conf_int.iloc[:, 1]
}, index=pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"))

print("\n未来12个月预测结果:")
print(forecast_df.round(2))

# --------------- 9. 可视化预测 ---------------
plt.figure(figsize=(12, 6))
plt.plot(ts, label="历史数据")
plt.plot(forecast_mean, label="预测值", color="red")
plt.fill_between(
    forecast_conf_int.index,
    forecast_conf_int.iloc[:, 0],
    forecast_conf_int.iloc[:, 1],
    color="pink",
    alpha=0.3
)
plt.title("ARIMA(1,2,1) 预测结果")
plt.xlabel("日期")
plt.ylabel("数值")
plt.legend()
plt.grid(True)
plt.show()