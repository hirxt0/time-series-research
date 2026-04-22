import sys
import torch
import numpy as np
sys.path.append('.')
from core.loader import get_data
from core.evaluator import calculate_metrics, plot_feature_space
from models.dcdetector.model import train_dc

if __name__ == "__main__":
    df = get_data()
    df = train_dc(df)
    
    df['delta'] = df['value'].diff().abs().fillna(0)
    df['v_long'] = df['value'].rolling(1200, center=True).std().fillna(0)
    
    y_true = (df['delta'] > 150).astype(int)
    calculate_metrics(y_true, df['anomaly'], "DC-Detector")
    plot_feature_space(df, 'anomaly', "DC-Detector")