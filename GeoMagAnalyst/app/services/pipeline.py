from app.utils.parsing import parse_csv
from app.services.eda import run_eda
from app.services.imputation import run_imputation
from app.services.anomaly import run_anomaly
import numpy as np
import math
 
 
async def run_pipeline(file, impute_model: str, anomaly_model: str):
    # 1. Парсинг CSV
    df_raw = await parse_csv(file)
 
    # 2. Обнаружение аномалий — на сырых данных, чтобы не терять пики при ресемплинге
    anomaly_result = run_anomaly(df_raw, model_name=anomaly_model)
 
    # 3. EDA: ресемплинг + feature engineering
    df_prepared, eda_meta = run_eda(df_raw)
 
    # 4. Импутация пропусков
    impute_result = run_imputation(df_prepared, model_name=impute_model)
 
    def clean_value(x):
        if x is None or isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 0.0
        return float(x)
 
    # Очищаем values
    values_clean = [clean_value(v) for v in df_prepared["value"].tolist()]
 
    # Очищаем filled_values
    filled_clean = [clean_value(v) for v in impute_result["filled_values"]]
 
    # seconds_list нужен раньше scores_clean
    seconds_list = df_prepared["seconds"].tolist()
    t_min = int(df_prepared["seconds"].min())
 
    # Конвертируем секунды аномалий (из сырых данных) в индексы минутного df
    anomaly_indices = []
    for ts in anomaly_result.get("timestamps", []):
        idx = int((int(ts) - t_min) // 60)
        if 0 <= idx < len(seconds_list) and idx not in anomaly_indices:
            anomaly_indices.append(idx)
 
    # Скоры: anomaly.py возвращает {секунда: скор} по сырым секундам,
    # выравниваем по минутному df (берём скор ближайшей сырой точки)
    scores_dict = anomaly_result.get("scores", {})
    scores_clean = []
    for s in seconds_list:
        # ищем любую сырую точку внутри этой минуты
        best = 0.0
        for offset in range(0, 60, 3):  # шаг сырых данных 3 сек
            raw_sec = int(s) + offset
            if raw_sec in scores_dict:
                best = clean_value(scores_dict[raw_sec])
                break
        scores_clean.append(best)

 
    # 5. Формирование результата
    result = {
        "rows": df_prepared.replace([np.inf, -np.inf], 0).fillna(0).to_dict(orient="records"),
        "values": values_clean,
        "filled": filled_clean,
        "gapIndices": [int(idx) for idx, is_gap in enumerate(df_prepared["is_gap"]) if is_gap],
        "anomalyIndices": anomaly_indices,
        "anomalyCount": int(anomaly_result.get("count", 0)),
        "gapCount": int(eda_meta.get("gap_count", 0)),
        "mae": float(impute_result.get("mae", 0)),
        "threshold": float(anomaly_result.get("threshold", 0.5)),
        "coverage": float(anomaly_result.get("coverage", 0.0)),
        "scores": scores_clean,
        "gapLengths": [int(l) for l in eda_meta.get("gap_lengths", [])],
        "modelMAEs": {k: float(v) for k, v in impute_result.get("model_metrics", {}).items()}
    }
 
    return result