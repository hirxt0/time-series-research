from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_min = pd.read_csv('combined_1min.csv', index_col=0, parse_dates=True)
df_min = df_min.squeeze()  # DataFrame → Series


df_min_test = df_min[df_min.index.year == 2024]
print(f'Точек: {len(df_min_test)}')

stl = STL(df_min_test.dropna(), period=1440, robust=True, seasonal=1441)
result = stl.fit()

# print('STL готов')
# print('Запускаем STL...')
# print(f'Точек: {len(df_min)}')
# print('Это займёт ~10–20 минут...')

# stl = STL(
#     df_min,
#     period=1440,    # 1440 минут = 1 сутки
#     robust=True,    # устойчив к выбросам (бурям)
#     seasonal=1441,  # нечётное >= period
# )
# result = stl.fit()

trend    = result.trend
seasonal = result.seasonal
residual = result.resid

print('STL готов')
print(f'\nТренд:      {trend.min():.0f} – {trend.max():.0f}')
print(f'Сезонность: {seasonal.min():.0f} – {seasonal.max():.0f}')
print(f'Остаток:    {residual.min():.0f} – {residual.max():.0f}')

# Визуализация
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
df_min.plot(ax=axes[0], lw=0.2, color='steelblue', title='Исходный')
pd.Series(trend,    index=df_min.index).plot(ax=axes[1], lw=0.8, color='navy',    title='Тренд')
pd.Series(seasonal, index=df_min.index).plot(ax=axes[2], lw=0.2, color='seagreen', title='Сезонность (суточная)')
pd.Series(residual, index=df_min.index).plot(ax=axes[3], lw=0.2, color='gray',     title='Остаток')
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', lw=0.3)
plt.tight_layout()
plt.show()

# Сохраняем
df_stl = pd.DataFrame({
    'value':    df_min.values,
    'trend':    trend,
    'seasonal': seasonal,
    'residual': residual,
}, index=df_min.index)

df_stl.to_parquet('combined_stl2024.parquet')
print('Сохранено: combined_stl.csv')