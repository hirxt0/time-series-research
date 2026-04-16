import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

class GeomagneticLabDetector:
    
    def __init__(self, contamination: float = 0.02, threshold_sigma: float = 3.0):
        self.contamination = contamination
        self.threshold_sigma = threshold_sigma
        self.model = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=42, n_jobs=-1)

    def _feature_engineering(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['delta_1'] = df[target].diff().abs().fillna(0)
        features['volatility_short'] = df[target].rolling(window=20, center=True).std().fillna(0)
        features['volatility_long'] = df[target].rolling(window=1200, center=True).std().fillna(0)
        return features

    def process(self, df: pd.DataFrame, target: str = 'value'):
        X = self._feature_engineering(df, target)
    
        preds = self.model.fit_predict(X)
        
        df_res = df.copy()
        df_res['is_anomaly'] = preds
        df_res['delta'] = X['delta_1']
        df_res['v_long'] = X['volatility_long']
        
        df_res['class'] = 'Normal'
        anomaly_mask = df_res['is_anomaly'] == -1
        
    
        tech_limit = df_res['delta'].mean() + self.threshold_sigma * df_res['delta'].std()
        
        is_tech = anomaly_mask & (df_res['delta'] > tech_limit)
        is_storm = anomaly_mask & (df_res['delta'] <= tech_limit)
        
        df_res.loc[is_storm, 'class'] = 'Natural Storm'
        df_res.loc[is_tech, 'class'] = 'Technogenic Artifact'
        
        return df_res

    def visualize(self, df: pd.DataFrame, target: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        ax1.plot(df.index, df[target], color='lightgray', label='Signal', lw=0.5)
        colors = {'Natural Storm': 'orange', 'Technogenic Artifact': 'red'}
        for label, color in colors.items():
            sub = df[df['class'] == label]
            ax1.scatter(sub.index, sub[target], c=color, s=15, label=label, zorder=3)
        ax1.set_title("Classification of Geomagnetic Events")
        ax1.legend()

        sns.scatterplot(data=df[df['class'] != 'Normal'], 
                        x='v_long', y='delta', hue='class', 
                        palette=colors, ax=ax2, alpha=0.6)
        ax2.set_yscale('log')
        ax2.set_title("Feature Space: Long-term Volatility vs Instantaneous Jump")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    column_names = ['seconds', 'value', 'quality', 'accuracy']

    df = pd.read_csv('ARS_pos1_2023.csv', header=None, names=column_names)

    if 'seconds' in df.columns:
        origin = pd.Timestamp("2024-01-01")
        df['dt'] = origin + pd.to_timedelta(df['seconds'], unit='s')
        df = df.set_index('dt').sort_index()

    target_col = 'value'
    
    detector = GeomagneticLabDetector(contamination=0.01)
    results = detector.process(df, target=target_col)
    
    print(f"Detected events:\n{results['class'].value_counts()}")
    detector.visualize(results, target=target_col)