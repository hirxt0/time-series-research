import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_parquet('prepared_df.parquet')

FEATURE_COLS = [
    'hour',
    'day_of_week',
    'week_number',
    'month_number',
    'day_of_year',
    'sin_hour',
    'cos_hour',
    'sin_dayofyear',
    'cos_dayofyear',
    'lag_mean_30min',
    'lag_mean_1h',
    'lag_mean_3h',
    'lag_mean_6h',
    'diff_1h',
    'diff_3h',
    'slope_1h',
    'slope_3h',
    'value_ema',
    'rolling_std_24h',
]
TARGET = 'value_detrended'

feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
df_model = df[feature_cols_present + [TARGET]].dropna()

print(f'Строк для обучения: {len(df_model):,}')

X = df_model[feature_cols_present]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

print(f'Train: {len(X_train):,}  |  Test: {len(X_test):,}')

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

params = {
    'objective':        'regression',
    'metric':           ['mae', 'rmse'],
    'learning_rate':    0.05,
    'num_leaves':       127,
    'max_depth':        -1,
    'min_child_samples': 50,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       0.1,
    'verbose':          -1,
    'n_jobs':           -1,
    'seed':             42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=4000,
    valid_sets=[dtrain, dval],
    valid_names=['train', 'val'],
    callbacks=callbacks,
)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'────────────── Метрики на тесте ──────────────────')
print(f'  MAE  : {mae:,.1f}')
print(f'  RMSE : {rmse:,.1f}')
print(f'  Лучшая итерация: {model.best_iteration}')

fi = pd.DataFrame({
    'feature': feature_cols_present,
    'importance_gain': model.feature_importance(importance_type='gain'),
}).sort_values('importance_gain', ascending=False)

print('\n── Feature Importance (gain) ─────────')
print(fi.to_string(index=False))

model.save_model('lgbm_model.txt')
joblib.dump(feature_cols_present, 'lgbm_features.pkl')

print('Модель сохранена: lgbm_model.txt')
print('Список фич сохранён: lgbm_features.pkl')