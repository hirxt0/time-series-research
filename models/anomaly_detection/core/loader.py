import pandas as pd

def get_data(path='ARS_pos1_2024.csv'):
    df = pd.read_csv(path, header=None, names=['seconds', 'value', 'q', 'a'])
    df['dt'] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df['seconds'], unit='s')
    return df.set_index('dt')