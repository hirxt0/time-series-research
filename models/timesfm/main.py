import pandas as pd
import matplotlib.pyplot as plt
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig


from data_processing import data_aggregation
from evaluation import evaluate_imputation
from visual import plot_gap_fill


df = pd.read_csv('ARS_pos1_2024.csv')
df_clean = data_aggregation(df)

model = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
)

forecast_config = ForecastConfig(
    max_horizon=60 * 5,     
    max_context=1440 * 2,    
    per_core_batch_size=1,
)

model.compile(forecast_config=forecast_config)

evaluate_imputation(model, df_clean)

plot_gap_fill(
    model,
    df_clean,
    gap_start_dt='2024-06-03 12:04:00',
    gap_hours=2.28,
    context_hours=12,
)


plot_gap_fill(
    model,
    df_clean,
    gap_start_dt='2024-07-31 15:00',
    gap_hours=2.0,
    context_hours=12,
)