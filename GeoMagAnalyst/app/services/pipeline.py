from app.utils.parsing import parse_csv
from app.services.eda import run_eda
from app.services.imputation import run_imputation
from app.services.anomaly import run_anomaly
import numpy as np
import math


def _clean(x) -> float:
    """NaN / Inf → 0.0, всё остальное → float."""
    if x is None:
        return 0.0
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return 0.0
    return float(x)


def _clean_metrics(d: dict) -> dict:
    """Рекурсивно чистит метрики: None → null оставляем, float чистим."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = _clean_metrics(v)
        else:
            out[k] = _clean(v)
    return out


async def run_pipeline(file, impute_model: str, anomaly_model: str):
    df_raw = await parse_csv(file)

    anomaly_result = run_anomaly(df_raw, model_name=anomaly_model)

    df_prepared, eda_meta = run_eda(df_raw)

    impute_result = run_imputation(df_prepared, model_name=impute_model)

    values_clean = [_clean(v) for v in df_prepared["value"].tolist()]
    filled_clean = [_clean(v) for v in impute_result["filled_values"]]
    seconds_list = df_prepared["seconds"].tolist()
    t_min        = int(df_prepared["seconds"].min())

    anomaly_indices = []
    for ts in anomaly_result.get("timestamps", []):
        idx = int((int(ts) - t_min) // 60)
        if 0 <= idx < len(seconds_list) and idx not in anomaly_indices:
            anomaly_indices.append(idx)

    scores_dict = anomaly_result.get("scores", {})
    scores_clean = []
    for s in seconds_list:
        best = 0.0
        for offset in range(0, 60, 3):
            raw_sec = int(s) + offset
            if raw_sec in scores_dict:
                best = _clean(scores_dict[raw_sec])
                break
        scores_clean.append(best)

    metrics_by_length = _clean_metrics(impute_result.get("metrics_by_length", {}))
    metrics_summary   = _clean_metrics(impute_result.get("metrics_summary",   {}))

    result = {
        "rows":   df_prepared.replace([np.inf, -np.inf], 0).fillna(0).to_dict(orient="records"),
        "values": values_clean,
        "filled": filled_clean,

        "gapIndices":     [int(i) for i, g in enumerate(df_prepared["is_gap"]) if g],
        "anomalyIndices": anomaly_indices,

        "scores":       scores_clean,
        "anomalyCount": int(anomaly_result.get("count", 0)),
        "gapCount":     int(eda_meta.get("gap_count", 0)),
        "gapLengths":   [int(l) for l in eda_meta.get("gap_lengths", [])],

        "threshold": _clean(anomaly_result.get("threshold", 0.5)),
        "coverage":  _clean(anomaly_result.get("coverage",  0.0)),


        "mape": _clean(impute_result.get("mape", 0.0)),

        "metricsByLength": metrics_by_length,

        "metricsSummary": metrics_summary,

        "modelMAPEs": {
            k: _clean(v)
            for k, v in impute_result.get("model_metrics", {}).items()
        },
    }

    return result