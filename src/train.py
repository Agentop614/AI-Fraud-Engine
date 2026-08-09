import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from imblearn.over_sampling import SMOTE
import mlflow

DATA_PATH = "data/creditcard.csv"
MODEL_SAVE_PATH = "models/xgboost_model.json"

def load_data(sample_size=50000):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Place 'creditcard.csv' in the data/ directory.")
    df = pd.read_csv(DATA_PATH).sample(n=sample_size, random_state=42)
    return df

def train_and_save_pipeline():
    print("Loading data...")
    df = load_data()
    
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Training XGBoost model...")
    mlflow.set_experiment("Fraud_Detection_XGBoost")
    
    with mlflow.start_run():
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train_res, y_train_res)
        
        # Evaluate
        preds = model.predict(X_test)
        recall = recall_score(y_test, preds)
        precision = precision_score(y_test, preds)
        
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        
        # Ensure models directory exists and save artifact
        os.makedirs("models", exist_ok=True)
        model.save_model(MODEL_SAVE_PATH)
        print(f"✅ Model saved successfully to {MODEL_SAVE_PATH}")
        print(f"Metrics - Recall: {recall:.4f} | Precision: {precision:.4f}")

if __name__ == "__main__":
    train_and_save_pipeline()