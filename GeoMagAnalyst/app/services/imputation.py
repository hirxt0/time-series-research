# app/services/imputation.py

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any
import os

# Константы
TARGET_STEP = 60
PTS_MIN = 20
PTS_HOUR = 1200
PTS_DAY = 28800
TREND_WINDOW = PTS_HOUR * 24


def run_imputation(df: pd.DataFrame, model_name: str = "lightgbm") -> Dict[str, Any]:
    """
    Заполняет пропуски с помощью обученной модели LightGBM
    """
    # Пути к файлам модели
    base_path = "models/imputation"
    model_pkl_path = f"{base_path}/lightgbm_model.pkl"
    model_txt_path = f"{base_path}/lgb_model.txt"
    features_path = f"{base_path}/feature_cols.pkl"
    
    # Загружаем список фичей
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)
        print(f"Загружены фичи из файла ({len(feature_cols)})")
    else:
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    
    # Загружаем модель
    try:
        if os.path.exists(model_pkl_path):
            with open(model_pkl_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Модель загружена из pkl")
        elif os.path.exists(model_txt_path):
            model = lgb.Booster(model_file=model_txt_path)
            print(f"Модель загружена из txt")
        else:
            raise FileNotFoundError(f"Model not found")
    except Exception as e:
        raise FileNotFoundError(f"Model load error: {e}")
    
    # Копируем данные
    df_work = df.copy()
    filled_values = df_work['value'].copy()
    
    # Проверяем наличие всех фичей
    missing_feats = [f for f in feature_cols if f not in df_work.columns]
    if missing_feats:
        print(f"Создаём недостающие фичи: {missing_feats}")
        for feat in missing_feats:
            df_work[feat] = 0
    
    # Находим пропуски
    gap_indices = [i for i, is_gap in enumerate(df_work['is_gap']) if is_gap]
    print(f"Найдено пропусков: {len(gap_indices)}")
    
    if not gap_indices:
        return {
            "filled_values": filled_values.tolist(),
            "mae": 0.0,
            "model_metrics": {model_name: 0.0}
        }
    
    # Заполняем пропуски последовательно
    for idx in gap_indices:
        # Собираем фичи для этого индекса
        features = []
        valid = True
        
        for col in feature_cols:
            if col not in df_work.columns:
                valid = False
                break
            val = df_work.loc[idx, col]
            if pd.isna(val):
                valid = False
                break
            features.append(float(val))
        
        if valid and len(features) == len(feature_cols):
            X_pred = np.array([features])
            pred = float(model.predict(X_pred)[0])
            fill_val = pred
        else:
            # Fallback: последнее известное значение
            prev_valid = filled_values[:idx].dropna()
            if len(prev_valid) > 0:
                fill_val = prev_valid.iloc[-1]
            else:
                next_valid = filled_values[idx+1:].dropna()
                fill_val = next_valid.iloc[0] if len(next_valid) > 0 else df_work['value'].mean()
        
        filled_values.iloc[idx] = fill_val
        df_work.loc[idx, 'value'] = fill_val
    
    # Считаем MAE на оригинальных данных
    mask_non_gap = ~df['is_gap'].astype(bool)
    original = df['value'][mask_non_gap]
    filled = filled_values[mask_non_gap]
    mask_valid = ~original.isna() & ~filled.isna()
    
    if mask_valid.sum() > 0:
        mae = np.mean(np.abs(original[mask_valid] - filled[mask_valid]))
    else:
        mae = 0.0
    
    return {
        "filled_values": filled_values.tolist(),
        "mae": round(float(mae), 4),
        "model_metrics": {model_name: round(float(mae), 4)}
    }
