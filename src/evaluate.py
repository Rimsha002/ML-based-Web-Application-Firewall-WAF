# src/evaluate.py
import pandas as pd, joblib, os
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "../data/sample_requests.csv"
MODEL_PATH = "../models/waf_pipeline.joblib"

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=['request','label'])
    X = df['request'].astype(str).values
    y = (df['label'].astype(str) == 'malicious').astype(int)
    pipeline = joblib.load(MODEL_PATH)
    vec = pipeline['vectorizer']
    model = pipeline['model']
    Xv = vec.transform(X)
    preds = model.predict(Xv)
    print(classification_report(y, preds, target_names=['benign','malicious']))
    print(confusion_matrix(y, preds))

if __name__ == "__main__":
    main()
