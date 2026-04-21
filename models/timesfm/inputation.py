import numpy as np


def fill_gaps_timesfm(
    model,
    series: np.ndarray,
    context_len: int = 1440 * 2,
    max_gap: int = 240,
    step: int = 20,
) -> np.ndarray:
    """
    Rolling forecast: заполняем пропуск кусками по step минут.
    После каждого шага заполненные точки добавляются в контекст.
    """
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
            print(f"Пропуск [{gap_start}:{gap_end}] ({gap_len} мин) - слишком длинный, пропускаем")
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
                inputs=np.array([context]),
                horizon=chunk,
            )

            pred = np.array(point_forecast[0])
            
            s[pos:pos + chunk] = pred[:chunk]

            pos += chunk
    
    return s


def bidirectional_fill(model, series, max_context=2880, max_gap=240):
    original = series.astype(float).copy()

    fwd = fill_gaps_timesfm(model, original, max_context, max_gap)
    bwd = fill_gaps_timesfm(
        model, original[::-1].copy(), max_context, max_gap
    )[::-1].copy()

    gap_mask = np.isnan(original)
    result = original.copy()

    gap_indices = np.where(gap_mask)[0]
    if len(gap_indices) == 0:
        return result

    breaks = np.where(np.diff(gap_indices) > 1)[0] + 1
    segments = np.split(gap_indices, breaks)

    for seg in segments:
        n = len(seg)
        fwd_vals = fwd[seg]
        bwd_vals = bwd[seg]

        use_fwd_only = np.isnan(bwd_vals) & ~np.isnan(fwd_vals)
        use_bwd_only = np.isnan(fwd_vals) & ~np.isnan(bwd_vals)
        use_both     = ~np.isnan(fwd_vals) & ~np.isnan(bwd_vals)

        t = np.linspace(0.0, 1.0, n)
        w_fwd = (1.0 - t) ** 2
        w_bwd = t ** 2
        total = w_fwd + w_bwd
        w_fwd /= total
        w_bwd /= total

        filled_seg = np.full(n, np.nan)
        filled_seg[use_both]     = (w_fwd[use_both] * fwd_vals[use_both] +
                                    w_bwd[use_both] * bwd_vals[use_both])
        filled_seg[use_fwd_only] = fwd_vals[use_fwd_only]
        filled_seg[use_bwd_only] = bwd_vals[use_bwd_only]

        result[seg] = filled_seg

    return result
