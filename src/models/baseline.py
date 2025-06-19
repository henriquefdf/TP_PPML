# %%  
# src/models/baseline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# 1) Load split data
train_df = pd.read_parquet("../../data/processed/train.parquet")
test_df  = pd.read_parquet("../../data/processed/test.parquet")

X_train, y_train = train_df["message"], train_df["label"]
X_test,  y_test  = test_df["message"],  test_df["label"]

# 2) Define the pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=5,        # drop very rare tokens
        max_df=0.8,      # drop overly common tokens
    )),
    ("clf", LogisticRegression(
        solver="liblinear",       # good for small datasets
        class_weight="balanced",  # compensate for imbalance
        random_state=42,
    )),
])

# 3) Train
pipeline.fit(X_train, y_train)

# 4) Evaluate
y_pred = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# Save the entire TF-IDF + LR pipeline
#joblib.dump(pipeline, "common/tfidf_logreg_pipeline.pkl")

# Or just the vectorizer if you prefer to reinstantiate the model on each client
joblib.dump(pipeline.named_steps["tfidf"], "../common/tfidf_vectorizer.pkl")

