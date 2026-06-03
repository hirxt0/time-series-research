import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any

def run_anomaly(df: pd.DataFrame, model_name: str = "iforest") -> Dict[str, Any]:
    """
    Принимает исходный DataFrame с шагом 3 секунды. Обучает Isolation Forest на валидных (не NaN) точках,
    рассчитывает аномальные скоры, инвертирует и нормализует их в диапазон [0, 1], а затем возвращает словарь
    со списком индексов аномалий, их количеством, порогом отсечения, долей покрытия и словарем скоров, 
    где ключами являются целочисленные представления секунд (int(seconds))
    """
    expected = ['seconds', 'value', 'quality', 'accuracy']
    if list(df.columns[:4]) != expected:
        rename = {old: new for old, new in zip(df.columns[:4], expected)}
        df.rename(columns=rename, inplace=True)
        
    data = df[['seconds', 'value']].copy()
    valid_mask = data['value'].notna()
    data_valid = data[valid_mask].copy()
    
    if len(data_valid) < 100:
        return {
            "timestamps": [],
            "indices": [],
            "count": 0,
            "threshold": 0.5,
            "coverage": 0.0,
            "scores": {}
        }
        
    X = data_valid[['value']].copy()
    X['delta'] = X['value'].diff().abs().fillna(0)
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.0005,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X)
    raw_scores = model.decision_function(X)
    
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm_scores_valid = (s_max - raw_scores) / (s_max - s_min + 1e-9)
    
    anomaly_mask = predictions == -1
    
    raw_seconds = data_valid.loc[anomaly_mask, 'seconds'].astype(int).values
    minute_timestamps = [int((s // 60) * 60) for s in raw_seconds]
    unique_minute_timestamps = list(set(minute_timestamps))
    
    scores_dict = {int(k): float(v) for k, v in zip(data_valid['seconds'], norm_scores_valid)}
    
    anomaly_indices = np.where(df['seconds'].isin(raw_seconds))[0].tolist()
    coverage = len(unique_minute_timestamps) / (len(df) / 20) if len(df) > 0 else 0.0

    return {
        "timestamps": unique_minute_timestamps,
        "indices": anomaly_indices,
        "count": len(unique_minute_timestamps),
        "threshold": float(np.percentile(norm_scores_valid, 99.95)) if len(norm_scores_valid) > 0 else 0.5,
        "coverage": round(float(coverage), 4),
        "scores": scores_dict
    }