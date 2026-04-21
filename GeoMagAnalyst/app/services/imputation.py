def run_imputation(df, model_name: str):
    """
    TODO:
    - загрузка модели из /models/imputation/
    - генерация лагов
    - предсказание пропусков
    - расчет MAE
    """

    return {
        "filled_values": df["value"].tolist(),  # same length
        "mae": 0.0,
        "model_metrics": {
            "LightGBM": 0
        }
    }