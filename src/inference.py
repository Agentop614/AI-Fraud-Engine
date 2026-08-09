import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

MODEL_PATH = "models/xgboost_model.json"

class FraudInferenceEngine:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at '{MODEL_PATH}'. Run 'python src/train.py' first.")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)
        self.explainer = shap.TreeExplainer(self.model)

    def predict_transaction(self, transaction_df: pd.DataFrame):
        prediction = self.model.predict(transaction_df)[0]
        risk_score = float(self.model.predict_proba(transaction_df)[0][1])
        return prediction, risk_score

    def get_shap_explanation(self, transaction_df: pd.DataFrame):
        shap_vals = self.explainer.shap_values(transaction_df)
        feature_names = transaction_df.columns
        
        # Top 3 anomalous features
        top_indices = np.argsort(shap_vals[0])[-3:]
        top_features = [
            f"{feature_names[idx]} (SHAP impact: +{shap_vals[0][idx]:.2f})"
            for idx in top_indices
        ]
        return top_features