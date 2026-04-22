import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any

def run_anomaly(df: pd.DataFrame, model_name: str = "iforest") -> Dict[str, Any]:
    """
    Выполняет поиск аномалий в геомагнитных данных с использованием алгоритма Isolation Forest.
    
    Аргументы:
        df: Подготовленный DataFrame (после EDA) с колонкой 'value'.
        model_name: Идентификатор используемой модели.
        
    Возвращает:
        Словарь с индексами аномалий, метриками и нормализованными скорами.
    """
    # 1. Формирование признакового пространства (Value + Производная)
    # Использование производной необходимо для идентификации резких техногенных скачков
    data_features = df[['value']].copy()
    data_features['delta'] = data_features['value'].diff().abs().fillna(0)
    
    # 2. Инициализация и обучение модели
    # contamination=0.02 соответствует ожидаемому уровню артефактов в 2%
    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    # 3. Детекция (1 — норма, -1 — аномалия)
    predictions = model.fit_predict(data_features)
    
    # 4. Расчет и нормализация аномальных скоров
    # decision_function возвращает значения, где меньше = аномальнее
    raw_scores = model.decision_function(data_features)
    
    # Инверсия и нормализация в диапазон [0, 1] для фронтенда
    s_min, s_max = raw_scores.min(), raw_scores.max()
    norm_scores = (s_max - raw_scores) / (s_max - s_min + 1e-9)
    
    # 5. Экстракция индексов и расчет покрытия
    anomaly_indices = np.where(predictions == -1)[0].tolist()
    coverage = len(anomaly_indices) / len(df) if len(df) > 0 else 0.0

    return {
        "indices": anomaly_indices,
        "count": len(anomaly_indices),
        "threshold": float(np.percentile(norm_scores, 98)),
        "coverage": round(float(coverage), 4),
        "scores": norm_scores.tolist()
    }
