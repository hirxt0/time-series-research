from sklearn.ensemble import IsolationForest

def run_iforest(df, contamination=0.02):
    feat = df[['value']].copy()
    feat['delta'] = feat['value'].diff().abs().fillna(0)
    model = IsolationForest(contamination=contamination, n_jobs=-1)
    df['anomaly'] = (model.fit_predict(feat) == -1).astype(int)
    df['delta'] = feat['delta']
    return df