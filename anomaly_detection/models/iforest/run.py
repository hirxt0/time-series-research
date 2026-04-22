import sys
import pandas as pd
from sklearn.ensemble import IsolationForest
sys.path.append('.')
from core.loader import get_data
from core.evaluator import calculate_metrics, plot_feature_space

def run_iforest(df):
    res = df.copy()
    res['delta'] = res['value'].diff().abs().fillna(0)
    res['v_long'] = res['value'].rolling(1200, center=True).std().fillna(0)
    
    model = IsolationForest(contamination=0.02, n_jobs=-1)
    res['anomaly'] = (model.fit_predict(res[['value', 'delta', 'v_long']]) == -1).astype(int)
    return res

if __name__ == "__main__":
    df = get_data()
    df = run_iforest(df)
    y_true = (df['delta'] > 150).astype(int)
    calculate_metrics(y_true, df['anomaly'], "Isolation Forest")
    plot_feature_space(df, 'anomaly', "IForest")