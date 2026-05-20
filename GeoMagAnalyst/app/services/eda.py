"""
app/services/eda.py

Принимает сырой DataFrame (seconds, value, quality, accuracy),
возвращает полностью подготовленный DataFrame с:
  - ресемплингом до 1-минутных точек
  - разметкой пропусков (NaN) там где шаг нарушен
  - feature engineering (временные + лаговые признаки)

Используется как первый шаг pipeline перед моделью.
"""

import pandas as pd
import numpy as np
from typing import Tuple

STEP_SEC = 3       
PTS_MIN = 20     
PTS_HOUR = 1200    
PTS_DAY = 28800   
TARGET_STEP = 60      
DIFF_THRESH = 50_000   
INTERP_LIMIT = 20      
TREND_WINDOW = PTS_HOUR * 24  



def run_eda(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Parameters
    ----------
    df_raw : pd.DataFrame
        Сырые данные с колонками: seconds, value, quality, accuracy.
        Колонки могут называться по-другому — первые 4 по порядку.

    Returns
    -------
    df_prepared : pd.DataFrame
        Подготовленный датафрейм с временными признаками и NaN в пропусках.
    meta : dict
        Статистика для отображения в UI.
    """
    df = _normalize_columns(df_raw)
    df = _validate_input(df)

    df_min, gap_info = _resample_to_minutes(df)

    df_min = _feature_engineering(df_min)

    df_min = _lag_features(df_min)

    meta = _build_meta(df_raw, df_min, gap_info)

    return df_min, meta



def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает первые 4 колонки в стандартные имена."""
    df = df.copy()
    expected = ['seconds', 'value', 'quality', 'accuracy']

    if list(df.columns[:4]) != expected:
        rename = {old: new for old, new in zip(df.columns[:4], expected)}
        df.rename(columns=rename, inplace=True)
    return df


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Базовые проверки и приведение типов."""
    df = df.copy()
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['value']   = pd.to_numeric(df['value'],   errors='coerce')
    df = df.dropna(subset=['seconds']).sort_values('seconds').reset_index(drop=True)
    return df



def _resample_to_minutes(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    1. Создаём регулярную сетку с шагом TARGET_STEP (60 сек).
    2. Усредняем исходные 3-секундные точки в каждую минуту.
    3. Минуты без данных помечаются NaN — это и есть пропуски.

    Возвращает датафрейм с индексом по секундам и словарь с
    информацией о пропусках (список (start_sec, end_sec, length_min)).
    """
    t_min = int(df['seconds'].min())
    t_max = int(df['seconds'].max())

    grid = np.arange(t_min, t_max + TARGET_STEP, TARGET_STEP)

    df = df.copy()
    df['minute_bucket'] = (
        ((df['seconds'] - t_min) // TARGET_STEP).astype(int) * TARGET_STEP + t_min
    )

    agg = df.groupby('minute_bucket').agg(
        value    =('value',    'mean'),
        quality  =('quality',  'mean'),
        accuracy =('accuracy', 'mean'),
        n_pts    =('value',    'count'),   
    )

    df_min = pd.DataFrame({'seconds': grid}).set_index('seconds')
    df_min = df_min.join(agg, how='left')

    gap_mask = df_min['n_pts'].isna() | (df_min['n_pts'] == 0)
    df_min.loc[gap_mask, 'value'] = np.nan

    gap_info = _find_gap_blocks(gap_mask.values, grid)

    df_min.reset_index(inplace=True)
    df_min['is_gap'] = gap_mask.values.astype(int)

    return df_min, gap_info

def _find_gap_blocks(gap_mask: np.ndarray, seconds: np.ndarray) -> dict:
    """
    Находит непрерывные блоки пропусков.
    Возвращает список словарей {start, end, length_min}.
    """
    blocks = []
    in_gap = False
    start_i = 0

    for i, is_gap in enumerate(gap_mask):
        if is_gap and not in_gap:
            in_gap = True
            start_i = i
        elif not is_gap and in_gap:
            in_gap = False
            length = i - start_i
            blocks.append({
                'start_sec': int(seconds[start_i]),
                'end_sec':   int(seconds[i - 1]),
                'length_min': length,
            })

    if in_gap:  
        length = len(gap_mask) - start_i
        blocks.append({
            'start_sec': int(seconds[start_i]),
            'end_sec':   int(seconds[-1]),
            'length_min': length,
        })

    lengths = [b['length_min'] for b in blocks]
    return {
        'blocks': blocks,
        'total_gap_min': sum(lengths),
        'count': len(blocks),
        'max_gap_min': max(lengths) if lengths else 0,
        'length_distribution': pd.Series(lengths).value_counts().sort_index().to_dict(),
    }




def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Воспроизводит feature engineering из ноутбука, но теперь
    через реальные минутные индексы (0, 1, 2, ...) а не секунды.
    """
    df = df.copy()
    pts = df.index 
    df['minute'] = pts % 60
    df['hour'] = (pts // 60) % 24
    df['day'] = pts // 1440
    df['day_of_week'] = (df['day'] + 2) % 7
    df['week_number'] = (df['day'] // 7) + 1
    df['day_of_year'] = df['day'] % 365
    df['month_number'] = _days_to_month(df['day'].values)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df


def _days_to_month(days: np.ndarray) -> np.ndarray:
    """Переводит номер дня в месяц (1–12), учитывая длины месяцев."""
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    cumulative = np.cumsum([0] + days_in_month[:-1])
    return np.searchsorted(cumulative, days % 365, side='right').clip(1, 12)



def _lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Лаговые и скользящие признаки
    """
    df = df.copy()
    v = df['value']
    
    v_temp = v.ffill().bfill()
    
    df['lag_mean_30min'] = v_temp.rolling(30, min_periods=1).mean().shift(1)
    df['lag_mean_1h'] = v_temp.rolling(60, min_periods=1).mean().shift(1)
    df['lag_mean_3h'] = v_temp.rolling(180, min_periods=1).mean().shift(1)
    df['lag_mean_6h'] = v_temp.rolling(360, min_periods=1).mean().shift(1)
    
    df['diff_1h'] = v_temp.diff(60)
    df['diff_3h'] = v_temp.diff(180)
    
    df['slope_1h'] = df['lag_mean_1h'] - df['lag_mean_1h'].shift(60)
    df['slope_3h'] = df['lag_mean_3h'] - df['lag_mean_3h'].shift(180)
    
    df['value_ema'] = v_temp.ewm(span=60, adjust=False).mean()
    df['rolling_std_24h'] = v_temp.rolling(1440, min_periods=1).std()
    
    df['trend'] = v_temp.rolling(1440, min_periods=1).mean()
    df['value_detrended'] = v_temp - df['trend']
    
    return df


def _build_meta(
    df_raw: pd.DataFrame,
    df_min: pd.DataFrame,
    gap_info: dict,
) -> dict:
    total_min = len(df_min)
    gap_min   = gap_info['total_gap_min']

    return {
        'raw_rows': len(df_raw),
        'total_minutes': total_min,
        'gap_count': gap_info['count'],
        'gap_total_min': gap_min,
        'gap_pct': round(100 * gap_min / total_min, 4) if total_min else 0,
        'gap_max_min': gap_info['max_gap_min'],
        'gap_distribution': gap_info['length_distribution'],
        'gap_blocks': gap_info['blocks'],
        'gap_lengths': [b['length_min'] for b in gap_info['blocks']],  # ← ДОБАВИТЬ
        'artifact_count': 0,
        'artifact_pct': 0,
        'value_mean': round(float(df_min['value'].mean(skipna=True)), 2),
        'value_std': round(float(df_min['value'].std(skipna=True)), 2),
        'value_min': round(float(df_min['value'].min(skipna=True)), 2),
        'value_max': round(float(df_min['value'].max(skipna=True)), 2),
        'step_sec': TARGET_STEP,
    }