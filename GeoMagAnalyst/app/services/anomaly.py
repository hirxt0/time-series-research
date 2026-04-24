import numpy as np
import pandas as pd
from typing import Dict, Any
 
 
def run_anomaly(df: pd.DataFrame, model_name: str = "zscore") -> Dict[str, Any]:
    """
    Детекция аномалий по rolling z-score.
    Точка аномальна если отклоняется от скользящего среднего (окно 60 мин)
    больше чем на THRESHOLD стандартных отклонений.
    """
    THRESHOLD = 3.0
    WINDOW = 60  # минут
    
    df = df.copy()
    expected = ['seconds', 'value', 'quality', 'accuracy']

    if list(df.columns[:4]) != expected:
        rename = {old: new for old, new in zip(df.columns[:4], expected)}
        df.rename(columns=rename, inplace=True)
 
    if 'seconds' not in df.columns:
        raise KeyError("DataFrame должен содержать колонку 'seconds'")
 
    data = df[['seconds', 'value']].copy()
    valid_mask = data['value'].notna()
    data_valid = data[valid_mask].copy()
 
    if len(data_valid) < WINDOW:
        return {
            "timestamps": [],
            "count": 0,
            "threshold": THRESHOLD,
            "coverage": 0.0,
            "scores": {},
            "model": model_name
        }
 
    v = data_valid['value']
 
    rolling_mean = v.rolling(window=WINDOW, min_periods=10, center=True).mean()
    rolling_std  = v.rolling(window=WINDOW, min_periods=10, center=True).std()
 
    # z-score: насколько точка далека от локального среднего
    z = ((v - rolling_mean) / (rolling_std + 1e-9)).abs()
 
    # Нормализуем скор в [0, 1] для отображения
    z_max = z.max() if z.max() > 0 else 1.0
    norm_scores = (z / z_max).fillna(0.0)
 
    anomaly_mask = z > THRESHOLD
    anomaly_timestamps = data_valid.loc[anomaly_mask, 'seconds'].tolist()
 
    # Словарь скоров для всех точек
    full_scores = pd.Series(np.nan, index=data.index)
    full_scores[valid_mask] = norm_scores.values
    scores_dict = dict(zip(data['seconds'], full_scores))
 
    coverage = len(anomaly_timestamps) / len(data_valid) if len(data_valid) > 0 else 0.0
 
    return {
        "timestamps": anomaly_timestamps,
        "count": len(anomaly_timestamps),
        "threshold": THRESHOLD,
        "coverage": round(float(coverage), 4),
        "scores": scores_dict,
        "model": model_name
    }