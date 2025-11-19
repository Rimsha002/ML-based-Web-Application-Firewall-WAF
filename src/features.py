# src/features.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import joblib

SUSPICIOUS_TOKENS = ["'", "--", "<script", "%3C", "union", "select", "drop", "or", "and", "1=1"]

def basic_stats(request):
    return {
        "length": len(request),
        "num_params": request.count("&") + request.count("?"),
        "num_special": sum(request.count(ch) for ch in ["'", "\"", "<", ">", ";", "--", "%"]),
        "suspicious_token_count": sum(request.lower().count(tok) for tok in SUSPICIOUS_TOKENS)
    }

class RequestVectorizer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), max_features=2000)

    def fit(self, requests):
        self.tfidf.fit(requests)
        return self

    def transform(self, requests):
        tf = self.tfidf.transform(requests).toarray()
        stats = np.array([[basic_stats(r)[k] for k in ("length","num_params","num_special","suspicious_token_count")] for r in requests])
        return np.hstack([tf, stats])

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
