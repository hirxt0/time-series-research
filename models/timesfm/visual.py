import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from inputation import fill_gaps_timesfm, bidirectional_fill


def plot_gap_fill(model, df: pd.DataFrame, gap_start_dt, gap_hours=5.0, context_hours=12):
    series = df['value_mean'].copy()

    gap_start = df.index.get_loc(gap_start_dt)
    gap_len = int(gap_hours * 60)
    gap_end = min(len(series), gap_start + gap_len)

    ground_truth = series.values[gap_start:gap_end].copy()
    corrupted = series.values.copy()
    corrupted[gap_start:gap_end] = np.nan

    filled_linear = pd.Series(corrupted).interpolate(method='linear').values
    filled_ffill = pd.Series(corrupted).ffill().values
    filled_tfm = fill_gaps_timesfm(model, corrupted)
    filled_bidir  = bidirectional_fill(model, corrupted)

    gt_clean = ~np.isnan(ground_truth)

    def metrics(pred):
        p = pred[gap_start:gap_end][gt_clean]
        g = ground_truth[gt_clean]
        return np.mean(np.abs(p - g)), np.sqrt(np.mean((p - g) ** 2))

    results = {
        'ffill': (filled_ffill,  *metrics(filled_ffill)),
        'Linear': (filled_linear, *metrics(filled_linear)),
        'TimesFM ': (filled_tfm,    *metrics(filled_tfm)),
        'TimesFM (bidirectional_fill)':     (filled_bidir,  *metrics(filled_bidir)),
    }

    view_start = max(0, gap_start - 130)
    view_end = min(len(series), gap_end + 130)
    time_idx = df.index[view_start:view_end]

    def view(arr):
        return arr[view_start:view_end]

    def gap_only(arr):
        v = np.full(len(time_idx), np.nan)
        ls = gap_start - view_start
        le = gap_end   - view_start
        v[ls:le] = view(arr)[ls:le]
        return v

    orig_view = series.values[view_start:view_end]
    gap_span  = (df.index[gap_start], df.index[gap_end - 1])

    colors = {
        'ffill': ('purple', '--'),
        'Linear': ('orange', '-'),
        'TimesFM' : ('tomato', '-'),
        'TimesFM (bidirectional_fill)': ('green', '-'),
    }

    fig, (ax_main, ax_table) = plt.subplots(
        2, 1,
        figsize=(15, 9),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f"Заполнение пропуска ~{gap_hours}ч", fontsize=13, fontweight='bold')

    ax_main.plot(time_idx, orig_view, color='steelblue', linewidth=0.8,
                 alpha=0.35, label='оригинал')

    ax_main.axvspan(gap_span[0], gap_span[1], color='salmon', alpha=0.12, label='пропуск')

    for name, (filled, mae, rmse) in results.items():
        color, ls = colors[name]
        ax_main.plot(
            time_idx, gap_only(filled),
            color=color, linewidth=2, linestyle=ls,
            label=f"{name}  MAE={mae:.0f}",
            zorder=5,
        )

    ax_main.set_ylabel("nT")
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=15)

    ax_table.axis('off')

    best_mae = min(results.items(), key=lambda x: x[1][1])[0]

    table_data = []
    for name, (_, mae, rmse) in results.items():
        marker = ' ✓' if name == best_mae else ''
        table_data.append([f"{name}{marker}", f"{mae:.1f}", f"{rmse:.1f}"])

    table = ax_table.table(
        cellText=table_data,
        colLabels=['Метод', 'MAE', 'RMSE'],
        cellLoc='center',
    loc='center',
        bbox=[0.2, 0.0, 0.6, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, (name, _) in enumerate(results.items()):
        if name == best_mae:
            for j in range(3):
                table[i+1, j].set_facecolor('#c8f0c8')  # светло-зелёный

    for j in range(3):
        table[0, j].set_facecolor('#d0d8e8')
        table[0, j].set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"{'Метод':<14} {'MAE':>8} {'RMSE':>8}")
    for name, (_, mae, rmse) in results.items():
        marker = ' ←' if name == best_mae else ''
        print(f"{name:<14} {mae:>8.1f} {rmse:>8.1f}{marker}")
