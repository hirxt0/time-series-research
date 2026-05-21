import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple
import os

TARGET_STEP = 60
PTS_MIN = 20
PTS_HOUR = 1200
PTS_DAY = 28800
TREND_WINDOW = PTS_HOUR * 24

GAP_CATEGORIES = {
    "short":  (1,   30),   
    "medium": (30,  120),   
    "long":   (120, 99999), 
}


def _load_timesfm():
    import timesfm
    for backend in ["gpu", "cpu"]:
        try:
            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=1,
                    horizon_len=128,
                    context_len=512,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
                ),
            )
            return tfm
        except Exception:
            if backend == "cpu":
                raise
            continue


def _fill_gaps_timesfm(
    series: np.ndarray,
    context_len: int = 1440 * 2,
    max_gap: int = 300,
    step: int = 20,
) -> np.ndarray:

    model = _load_timesfm()
    s = series.astype(float).copy()
    i = 0

    while i < len(s):
        if not np.isnan(s[i]):
            i += 1
            continue

        gap_start = i
        while i < len(s) and np.isnan(s[i]):
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start

        if gap_len > max_gap:
            ctx = s[max(0, gap_start - context_len):gap_start]
            fallback = float(np.nanmedian(ctx)) if not np.all(np.isnan(ctx)) else 0.0
            s[gap_start:gap_end] = fallback
            continue

        pos = gap_start
        while pos < gap_end:
            chunk = min(step, gap_end - pos)
            ctx_start = max(0, pos - context_len)
            context = s[ctx_start:pos].copy()

            if len(context) < 10:
                pos += chunk
                continue

            ctx_median = np.nanmedian(context)
            context = np.where(np.isnan(context), ctx_median, context)

            point_forecast, _ = model.forecast(
                inputs=[context],
                freq=[0],
            )
            pred = np.array(point_forecast[0])
            s[pos:pos + chunk] = pred[:chunk]
            pos += chunk

    return s



def _load_model_and_features(model_name: str) -> Tuple[Any, List[str]]:
    base_path = "models/imputation"
    model_pkl = f"{base_path}/lightgbm_model.pkl"
    model_txt = f"{base_path}/lgb_model.txt"
    features_path = f"{base_path}/feature_cols.pkl"

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    with open(features_path, "rb") as f:
        feature_cols = pickle.load(f)

    if os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
    elif os.path.exists(model_txt):
        model = lgb.Booster(model_file=model_txt)
    else:
        raise FileNotFoundError("Model file not found")

    return model, feature_cols



def _fill_gaps_timesfm_df(df: pd.DataFrame) -> pd.Series:

    series = df["value"].values.copy()
    filled = _fill_gaps_timesfm(series)
    return pd.Series(filled, index=df.index, name="value")


def _fill_gaps(df: pd.DataFrame, model: Any, feature_cols: List[str]) -> pd.Series:

    df_work = df.copy()
    filled = df_work["value"].copy()
    gap_indices  = df_work.index[df_work["is_gap"].astype(bool)].tolist()

    for feat in feature_cols:
        if feat not in df_work.columns:
            df_work[feat] = 0.0

    for idx in gap_indices:
        row = df_work.loc[idx, feature_cols]
        has_nan = row.isna().any()

        if not has_nan:
            X_pred  = row.values.reshape(1, -1)
            pred = float(model.predict(X_pred)[0])
        else:
            prev = filled[:idx].dropna()
            next_ = filled[idx + 1:].dropna()
            if len(prev) > 0:
                pred = float(prev.iloc[-1])
            elif len(next_) > 0:
                pred = float(next_.iloc[0])
            else:
                pred = float(filled.mean())

        filled.loc[idx] = pred
        df_work.loc[idx, "value"] = pred  

    return filled



def _make_synthetic_gap(
    df: pd.DataFrame,
    gap_length_min: int = 120,
    seed: int = 42,
) -> Tuple[pd.DataFrame, int, int]:

    rng = np.random.default_rng(seed)
    known_idx  = df.index[~df["is_gap"].astype(bool)].tolist()

    margin = 1440
    candidates = [
        i for i in known_idx
        if i >= margin
        and i + gap_length_min < len(df) - margin
        and all((i + k) in known_idx for k in range(gap_length_min))
    ]

    if not candidates:
        raise ValueError(
            f"Не удалось найти непрерывный кусок длиной {gap_length_min} мин "
            f"среди известных данных"
        )

    start = int(rng.choice(candidates[:max(1, len(candidates) // 2)]))
    end = start + gap_length_min

    df_syn = df.copy()
    df_syn.loc[start:end - 1, "value"]  = np.nan
    df_syn.loc[start:end - 1, "is_gap"] = 1

    return df_syn, start, end


def _mae_by_category(
    ground_truth: pd.Series,
    predicted: pd.Series,
    gap_start: int,
    gap_end: int,
) -> Dict[str, float]:

    gap_len = gap_end - gap_start
    gt_gap = ground_truth.iloc[gap_start:gap_end].values
    pr_gap = predicted.iloc[gap_start:gap_end].values

    valid = ~np.isnan(pr_gap)
    if valid.sum() == 0:
        return {"overall": None, "short": None, "medium": None, "long": None}

    overall_mae = float(np.mean(np.abs(gt_gap[valid] - pr_gap[valid])))

    cat_mae: Dict[str, float] = {"overall": overall_mae}
    for cat, (lo, hi) in GAP_CATEGORIES.items():
        if lo <= gap_len < hi:
            cat_mae[cat] = overall_mae
        else:
            cat_mae[cat] = None  
    third = max(1, gap_len // 3)
    segments = {
        "start_third":  (0,           third),
        "middle_third": (third,       2 * third),
        "end_third": (2 * third,   gap_len),
    }
    for seg_name, (s, e) in segments.items():
        seg_gt = gt_gap[s:e]
        seg_pr = pr_gap[s:e]
        seg_v  = ~np.isnan(seg_pr)
        cat_mae[seg_name] = (
            float(np.mean(np.abs(seg_gt[seg_v] - seg_pr[seg_v])))
            if seg_v.sum() > 0 else None
        )

    return cat_mae


def evaluate_imputation(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    gap_lengths: List[int] = (30, 120, 300),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Для каждой длины из gap_lengths:
      1. Делаем синтетический пропуск
      2. Заполняем моделью
      3. Считаем MAE

    Возвращает словарь:
    {
      "by_length": {
          30:  {"overall": 123.4, "start_third": ..., ...},
          120: {...},
          300: {...},
      },
      "summary": {"short": 123.4, "medium": 456.7, "long": 789.0}
    }
    """
    by_length: Dict[int, Dict] = {}

    for gap_len in gap_lengths:
        try:
            df_syn, start, end = _make_synthetic_gap(df, gap_length_min=gap_len, seed=seed)
        except ValueError as e:
            by_length[gap_len] = {"error": str(e)}
            continue

        ground_truth = df["value"].copy()   
        filled       = _fill_gaps(df_syn, model, feature_cols)

        metrics = _mae_by_category(ground_truth, filled, start, end)
        by_length[gap_len] = metrics

    summary: Dict[str, float] = {}
    for cat, (lo, hi) in GAP_CATEGORIES.items():
        vals = [
            m["overall"]
            for gl, m in by_length.items()
            if "overall" in m and m["overall"] is not None and lo <= gl < hi
        ]
        summary[cat] = float(np.mean(vals)) if vals else None

    return {"by_length": by_length, "summary": summary}



def run_imputation(df: pd.DataFrame, model_name: str = "lightgbm") -> Dict[str, Any]:

    use_timesfm = model_name.lower() == "timesfm"

    if use_timesfm:
        filled_values = _fill_gaps_timesfm_df(df)
        metrics = _evaluate_timesfm(df, gap_lengths=[30, 120, 300])
    else:
        model, feature_cols = _load_model_and_features(model_name)
        filled_values = _fill_gaps(df, model, feature_cols)
        metrics = evaluate_imputation(df, model, feature_cols, gap_lengths=[30, 120, 300])

    overall_maes = [
        m["overall"]
        for m in metrics["by_length"].values()
        if isinstance(m, dict) and m.get("overall") is not None
    ]
    mae_overall = float(np.mean(overall_maes)) if overall_maes else 0.0

    return {
        "filled_values":  filled_values.tolist(),
        "mae":            mae_overall,
        "metrics_by_length": {
            str(k): v for k, v in metrics["by_length"].items()
        },
        "metrics_summary": metrics["summary"],
        "model_metrics": {
            model_name: mae_overall,
        },
    }


def _evaluate_timesfm(
    df: pd.DataFrame,
    gap_lengths: List[int] = (30, 120, 300),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Оценка качества TimesFM на синтетических пропусках.
    Та же логика что evaluate_imputation, но использует _fill_gaps_timesfm вместо LightGBM.
    """
    by_length: Dict[int, Dict] = {}

    for gap_len in gap_lengths:
        try:
            df_syn, start, end = _make_synthetic_gap(df, gap_length_min=gap_len, seed=seed)
        except ValueError as e:
            by_length[gap_len] = {"error": str(e)}
            continue

        ground_truth = df["value"].copy()
        filled = _fill_gaps_timesfm_df(df_syn)

        metrics = _mae_by_category(ground_truth, filled, start, end)
        by_length[gap_len] = metrics

    summary: Dict[str, float] = {}
    for cat, (lo, hi) in GAP_CATEGORIES.items():
        vals = [
            m["overall"]
            for gl, m in by_length.items()
            if "overall" in m and m["overall"] is not None and lo <= gl < hi
        ]
        summary[cat] = float(np.mean(vals)) if vals else None

    return {"by_length": by_length, "summary": summary}