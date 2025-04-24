import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

# ======================
# 环境设置
# ======================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# ======================
# 数据准备
# ======================
data = {
    '数据年月': [
        '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
        '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
        '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
        '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
        '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06',
        '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
        '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
        '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',
        '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
        '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12'
    ],
    '人民币': [
        729562946, 279143421, 435329481, 786571801, 903997110, 840090138,
        816355506, 774299120, 1001876701, 951567026, 874078726, 951266475,
        742834968, 381498848, 661866864, 675509918, 649804679, 631873003,
        695154746, 561664804, 544027993, 474355467, 594087514, 901256012,
        1258422202, 752106702, 1102262515, 1440799940, 1743854941, 1648362981,
        1585041433, 1336964387, 1797316541, 1803685830, 1650954900, 1370677617,
        845525895, 773442906, 1271774938, 1085226686, 1164556412, 1104611864,
        1015793859, 992130832, 1118749018, 981392291, 1060125537, 921126534,
        1170922845, 1001193929, 1071379123, 1267961877, 1179995939, 1077022567,
        1124663223, 1355701217, 1671086388, 1626066444, 1744341183, 1829768301
    ]
}

# 创建时间序列
df = pd.DataFrame(data)
df['数据年月'] = pd.to_datetime(df['数据年月'])
df.set_index('数据年月', inplace=True)
ts = df['人民币']

# ======================
# 数据可视化
# ======================
def plot_series(series, title):
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('金额')
    plt.grid(True)
    plt.show()

plot_series(ts, '人民币时间序列')

# ======================
# 平稳性检验
# ======================
def adf_test(series):
    result = adfuller(series)
    print('ADF检验结果:')
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'P值: {result[1]:.4f}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')

adf_test(ts)

# ======================
# 模型训练
# ======================
# 自动选择ARIMA参数
auto_model = pm.auto_arima(
    ts,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    trace=True
)
best_order = (1,2,1)  # 示例参数


# 建立ARIMA模型(以自动选择结果为例)
# 获取自动选择的最优参数
arima_model = ARIMA(ts, order=best_order).fit()
 # 使用自动选择的参数
print('\n自动参数选择结果:')
print(auto_model.summary())

# 训练最终模型
final_model = ARIMA(ts, order=best_order).fit()
print('\n模型摘要:')
print(final_model.summary())

# ======================
# 模型诊断
# ======================
# 残差分析
def plot_residuals(residuals):
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    residuals.plot(title='残差序列')
    plt.subplot(222)
    residuals.plot(kind='kde', title='残差分布')
    plt.subplot(223)
    plot_acf(residuals, lags=20, ax=plt.gca())
    plt.subplot(224)
    plot_pacf(residuals, lags=20, ax=plt.gca())

    plt.tight_layout()
    plt.show()

residuals = final_model.resid
plot_residuals(residuals)

# 白噪声检验
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print('\nLjung-Box检验结果:')
print(lb_test)

# ======================
# 历史数据验证
# ======================
def evaluate_performance(actual, fitted):
    residuals = actual - fitted
    metrics = {
        'MAE': np.mean(np.abs(residuals)),
        'MSE': np.mean(residuals**2),
        'RMSE': np.sqrt(np.mean(residuals**2)),
        'MAPE': np.mean(np.abs(residuals/actual)) * 100
    }
    print("\n模型表现指标:")
    for k, v in metrics.items():
        print(f'{k}: {v:,.2f}' if k != 'MAPE' else f'{k}: {v:.2f}%')

    # 拟合效果可视化
    plt.figure(figsize=(14, 7))
    actual.plot(label='实际值', alpha=0.7)
    fitted.plot(label='拟合值', linestyle='--')
    plt.title('历史数据拟合效果')
    plt.legend()
    plt.grid()
    plt.show()

evaluate_performance(ts, final_model.fittedvalues)

# ======================
# 未来预测
# ======================
def forecast_series(model, steps=6):
    # 生成预测
    forecast = model.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    ci = forecast.conf_int()

    # 创建日期索引
    last_date = ts.index[-1]
    dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps,
        freq='MS'
    )

    # 可视化
    plt.figure(figsize=(14, 7))
    plt.plot(ts, label='历史数据')
    plt.plot(dates, mean, label='预测值', linestyle='--')
    plt.fill_between(dates, ci.iloc[:,0], ci.iloc[:,1], color='gray', alpha=0.2)
    plt.title('未来12个月预测')
    plt.legend()
    plt.grid()
    plt.show()

    # 返回预测结果
    return pd.DataFrame({
        '预测值': mean,
        '预测下限': ci.iloc[:,0],
        '预测上限': ci.iloc[:,1]
    }, index=dates)

forecast_results = forecast_series(final_model)
print('\n未来6个月预测结果:')
print(forecast_results.round(2))

# 保存结果
forecast_results.to_csv('人民币预测结果.csv')