import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def calculate_metrics(y_true, y_pred, model_name):
    print(f"\n=== METRICS: {model_name} ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

def plot_feature_space(df, anomaly_col, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    subset = df.iloc[::100]
    ax1.plot(subset.index, subset['value'], color='lightgray', alpha=0.5)
    
    anoms = subset[subset[anomaly_col] == 1]
    norms = subset[subset[anomaly_col] == 0]
    
    ax1.scatter(norms.index, norms['value'], color='orange', s=10, alpha=0.6, label='Natural')
    ax1.scatter(anoms.index, anoms['value'], color='red', s=10, label='Artifact')
    ax1.set_title(f"{title}: Classification")
    ax1.legend()

    ax2.scatter(norms['v_long'], norms['delta'], c='orange', s=15, alpha=0.4)
    ax2.scatter(anoms['v_long'], anoms['delta'], c='red', s=15, alpha=0.6)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Volatility (v_long)')
    ax2.set_ylabel('Jump (delta)')
    plt.tight_layout()
    plt.show()