import numpy as np
import pandas as pd

from inputation import fill_gaps_timesfm

CONTEXT = 1440
HORIZON = 15


def inject_gaps(series, storm_flag, max_gap=120):
    x = series.copy()

    if np.random.rand() < 0.5:
        idx = np.where(storm_flag == 0)[0]
    else:
        idx = np.where(storm_flag == 1)[0]

    if len(idx) == 0:
        return x, None, None

    start = np.random.choice(idx)
    gap = np.random.randint(10, max_gap)

    end = min(len(x), start + gap)
    x[start:end] = np.nan

    return x, start, end - start


def make_dataset(df: pd.DataFrame):
    X, y = [], []

    target = df['value_mean'].values
    storm  = df['is_storm'].values
    level  = df['storm_level'].values
    sin_d  = df['sin_day'].values
    cos_d  = df['cos_day'].values

    for i in range(CONTEXT, len(df) - HORIZON):

        window = target[i-CONTEXT:i]

        if np.isnan(window).mean() > 0.2:
            continue

        corrupted, start, gap = inject_gaps(window, storm[i-CONTEXT:i])

        if start is None or gap < 5:
            continue

        context = corrupted

        features = np.stack([
            context,
            storm[i-CONTEXT:i],
            level[i-CONTEXT:i],
            sin_d[i-CONTEXT:i],
            cos_d[i-CONTEXT:i],
        ], axis=-1)

        target_y = target[i:i+HORIZON]

        if np.isnan(target_y).any():
            continue

        X.append(features)
        y.append(target_y)

    return np.array(X), np.array(y)


def evaluate_imputation(model, df_val: pd.DataFrame, n_samples: int = 20):
    series = df_val['value_norm'].values
    clean_mask = ~np.isnan(series)

    maes, rmses = [], []

    for trial in range(n_samples):
        clean_idx = np.where(clean_mask)[0]

        valid_starts = [
            idx for idx in clean_idx
            if idx >= 200         
            and idx + 120 < len(series)
            and clean_mask[idx-200:idx].all() 
        ]

        if len(valid_starts) == 0:
            print("нет валидных стартов")
            break

        start = valid_starts[np.random.randint(len(valid_starts))]
        gap_len = np.random.randint(10, 60)
        end = start + gap_len

        ground_truth = series[start:end].copy()

        if np.isnan(ground_truth).any():
            continue

        corrupted = series.copy()
        corrupted[start:end] = np.nan

        filled = fill_gaps_timesfm(model, corrupted, context_len=1440, max_gap=120)

        pred = filled[start:end]

        print(f"trial {trial}: start={start}, gap={gap_len}")
        print(f"pred NaN: {np.isnan(pred).sum()}, gt NaN: {np.isnan(ground_truth).sum()}")
        print(f"pred[:3]: {pred[:3]}, gt[:3]: {ground_truth[:3]}")

        if np.isnan(pred).any():
            continue

        maes.append(np.mean(np.abs(pred - ground_truth)))
        rmses.append(np.sqrt(np.mean((pred - ground_truth) ** 2)))

    if len(maes) == 0:
        print("Ни один trial не прошёл — все pred содержат NaN")
        return [], []

    print(f"\nImputation MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"Imputation RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    return maes, rmses