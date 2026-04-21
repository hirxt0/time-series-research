from app.utils.parsing import parse_csv
from app.services.eda import run_eda
from app.services.imputation import run_imputation
from app.services.anomaly import run_anomaly


async def run_pipeline(file, impute_model: str, anomaly_model: str):
    # 1. parse
    df = await parse_csv(file)

    # 2. EDA
    eda_result = run_eda(df)

    # 3. imputation
    impute_result = run_imputation(df, model_name=impute_model)

    # 4. anomaly detection
    anomaly_result = run_anomaly(df, model_name=anomaly_model)

    # 5. aggregation (формат строго под фронт)
    result = {
        "rows": df.to_dict(orient="records"),
        "values": df["value"].tolist(),

        "filled": impute_result["filled_values"],

        "gapIndices": eda_result["gap_indices"],
        "anomalyIndices": anomaly_result["indices"],

        "anomalyCount": anomaly_result["count"],
        "gapCount": eda_result["gap_count"],

        "mae": impute_result["mae"],
        "threshold": anomaly_result["threshold"],
        "coverage": anomaly_result["coverage"],

        "scores": anomaly_result["scores"],
        "gapLengths": eda_result["gap_lengths"],

        "modelMAEs": impute_result["model_metrics"]
    }

    return result