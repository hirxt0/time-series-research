import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib

PARQUET_PATH   = 'prepared_df.parquet'
MODEL_PATH     = 'lgbm_model.txt'
FEATURES_PATH  = 'lgbm_features.pkl'
PTS_HOUR       = 1200

GAP_START_IDX  = None   
GAP_HOURS      = 2.0
CONTEXT_HOURS  = 4.0    

df = pd.read_parquet(PARQUET_PATH)
model = lgb.Booster(model_file=MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)

gap_start = GAP_START_IDX if GAP_START_IDX is not None else len(df) // 2
gap_len = int(GAP_HOURS * PTS_HOUR)
gap_end = min(len(df), gap_start + gap_len)

real = df['value_detrended'].values.copy()
corrupted = real.copy()
corrupted[gap_start:gap_end] = np.nan

linear = pd.Series(corrupted).interpolate(method='linear').values

X_gap = df[feature_cols].iloc[gap_start:gap_end]
lgbm_fill = corrupted.copy()
lgbm_fill[gap_start:gap_end] = model.predict(X_gap)

gt = real[gap_start:gap_end]

def mae_rmse(pred):
    p = pred[gap_start:gap_end]
    return np.mean(np.abs(p - gt)), np.sqrt(np.mean((p - gt) ** 2))

mae_lin,  rmse_lin  = mae_rmse(linear)
mae_lgbm, rmse_lgbm = mae_rmse(lgbm_fill)

print(f'{"Метод":<10} {"MAE":>10} {"RMSE":>10}')
print(f'{"Linear":<10} {mae_lin:>10.1f} {rmse_lin:>10.1f}')
print(f'{"LightGBM":<10} {mae_lgbm:>10.1f} {rmse_lgbm:>10.1f}')

ctx = int(CONTEXT_HOURS * PTS_HOUR)
vs = max(0, gap_start - ctx)
ve = min(len(df), gap_end + ctx)
x = np.arange(vs, ve)

def v(arr): 
    return arr[vs:ve]

def gap_only(arr):
    out = np.full(ve - vs, np.nan)
    ls, le = gap_start - vs, gap_end - vs
    out[ls:le] = v(arr)[ls:le]
    return out

fig, ax = plt.subplots(figsize=(15, 5))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

ax.plot(x, v(real), color='#4a9eff', lw=0.8, alpha=0.5,  label='реальный сигнал')
ax.axvspan(gap_start, gap_end, color='#ff4757', alpha=0.10)
ax.axvline(gap_start, color='#ff4757', lw=0.8, ls=':')
ax.axvline(gap_end, color='#ff4757', lw=0.8, ls=':')

ax.plot(x, gap_only(linear), color='#e67e22', lw=2.0, ls='--',
        label=f'Linear    MAE={mae_lin:,.0f}  RMSE={rmse_lin:,.0f}')
ax.plot(x, gap_only(lgbm_fill),color='#2ecc71', lw=2.2,
        label=f'LightGBM  MAE={mae_lgbm:,.0f}  RMSE={rmse_lgbm:,.0f}')

ax.set_title(f'Заполнение пропуска {GAP_HOURS:.1f}ч  (индексы {gap_start}–{gap_end})',
             color='white', fontsize=12, pad=10)
ax.set_xlabel('Индекс точки', color='#aaaaaa')
ax.set_ylabel('value_detrended (nT)', color='#aaaaaa')
ax.tick_params(colors='#aaaaaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333333')
ax.grid(True, alpha=0.15, color='#555555')
ax.legend(loc='upper left', fontsize=9, framealpha=0.3,
          labelcolor='white', facecolor='#1a1a2e')

plt.tight_layout()
plt.savefig('gap_fill_result.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print('Сохранено: gap_fill_result.png')