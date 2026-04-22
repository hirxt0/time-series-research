from app.utils.parsing import parse_csv
from app.services.eda import run_eda
from app.services.imputation import run_imputation
from app.services.anomaly import run_anomaly
import numpy as np
import math


async def run_pipeline(file, impute_model: str, anomaly_model: str):
    # 1. Парсинг CSV
    df_raw = await parse_csv(file)
    
    # 2. EDA: ресемплинг + feature engineering
    df_prepared, eda_meta = run_eda(df_raw)
    
    # 3. Импутация пропусков
    impute_result = run_imputation(df_prepared, model_name=impute_model)
    
    # 4. Обнаружение аномалий
    anomaly_result = run_anomaly(df_prepared, model_name=anomaly_model)
    
    def clean_value(x):
        if x is None or isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return 0.0
        return float(x)
    
    # Очищаем values
    values_clean = [clean_value(v) for v in df_prepared["value"].tolist()]
    
    # Очищаем filled_values
    filled_clean = [clean_value(v) for v in impute_result["filled_values"]]
    
    # Очищаем scores
    scores_clean = [clean_value(s) for s in anomaly_result.get("scores", [])]
    
    # 5. Формирование результата
    result = {
        "rows": df_prepared.replace([np.inf, -np.inf], 0).fillna(0).to_dict(orient="records"),
        "values": values_clean,
        "filled": filled_clean,
        "gapIndices": [int(idx) for idx, is_gap in enumerate(df_prepared["is_gap"]) if is_gap],
        "anomalyIndices": [int(i) for i in anomaly_result.get("indices", [])],
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
