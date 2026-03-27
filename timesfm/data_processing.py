import pandas as pd
import numpy as np

DIFF_THRESHOLD = 150
ORIGIN = pd.Timestamp("2024-01-01 00:00:00")

STORM_WINDOW_MIN = 1440
STORM_Z_THRESHOLD = 2.5
STORM_MIN_DURATION = 30
GAP_FILL_THRESHOLD_MIN = 60

df = pd.read_csv('ARS_pos1_2024.csv')

def data_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ['seconds', 'value', 'quality', 'accuracy']

    df['diff'] = df['value'].diff().abs()

    anomaly_mask = df['diff'] > DIFF_THRESHOLD
    anomaly_mask = anomaly_mask | anomaly_mask.shift(-1).fillna(False)

    df['is_artifact'] = anomaly_mask.astype(int)

    df['value_clean'] = df['value'].astype(float)
    df.loc[df['is_artifact'] == 1, 'value_clean'] = np.nan

    df['value_clean'] = df['value_clean'].interpolate(method='linear', limit = 60)

    df['value_clean'] = df['value_clean'].ffill(limit=5)

    df['datetime'] = ORIGIN + pd.to_timedelta(df['seconds'], unit='s')
    df = df.set_index('datetime').sort_index()

    df_min = df.resample('1min').agg(
        value_mean   = ('value_clean', 'median'),
        artifact_sum = ('is_artifact', 'sum'),
    )

    df_min['mask'] = (~df_min['value_mean'].isna()).astype(int)


    df_min['sin_day'] = np.sin(2*np.pi*(df_min.index.hour + df_min.index.minute)/1440)
    df_min['cos_day'] = np.cos(2*np.pi*(df_min.index.hour + df_min.index.minute)/1440)

    mean = df_min['value_mean'].mean()
    std = df_min['value_mean'].std()
    df_min['value_norm'] = (df_min['value_mean'] - mean) / std

    df_min = detect_storms(df_min)

    gap_mask = df_min['value_mean'].isna()

    gap_lengths = gap_mask.astype(int).groupby(
        (~gap_mask).astype(int).cumsum()
    ).transform('sum')

    long_gap_mask = gap_mask & (gap_lengths >= GAP_FILL_THRESHOLD_MIN)
    df_min.loc[long_gap_mask, 'value_mean'] = 0.0
    df_min.loc[long_gap_mask, 'mask'] = 0

    return df_min


def detect_storms(df: pd.DataFrame):
    df = df.copy()
    s = df['value_mean']

    roll_mean = s.rolling(STORM_WINDOW_MIN, center=True, min_periods=10).mean()
    roll_std  = s.rolling(STORM_WINDOW_MIN, center=True, min_periods=10).std()

    z = (s - roll_mean) / roll_std.replace(0, np.nan)
    df['storm_z'] = z

    raw = (z.abs() > STORM_Z_THRESHOLD).astype(int)

    sustained = (
        raw.rolling(STORM_MIN_DURATION, center=True, min_periods=1).sum()
        >= (STORM_MIN_DURATION // 2)
    )

    df['is_storm'] = sustained.astype(int)

    med = s.median()
    mad = (s - med).abs().median()
    df['storm_level'] = ((s - med) / (mad + 1e-9)).abs()
    df['storm_level'] = np.clip(df['storm_level']/5, 0, 1)

    return df