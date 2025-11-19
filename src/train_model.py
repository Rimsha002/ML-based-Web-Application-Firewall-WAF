# src/train_model.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from features import RequestVectorizer

DATA_PATH = "../data/sample_requests.csv"   # adjust if using other dataset
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "waf_pipeline.joblib")

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['request','label'])
    X = df['request'].astype(str).values
    y = (df['label'].astype(str) == 'malicious').astype(int)  # 1 = malicious, 0 = benign
    return X, y

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vec = RequestVectorizer()
    vec.fit(X_train)

    # transform training data
    X_train_vec = vec.transform(X_train)
    X_test_vec = vec.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    preds = clf.predict(X_test_vec)
    print("Classification report on test set:")
    print(classification_report(y_test, preds, target_names=['benign','malicious']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    # Save both vectorizer and model together as a dict
    pipeline = {"vectorizer": vec, "model": clf}
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()

