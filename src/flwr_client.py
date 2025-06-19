
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import flwr as fl

class SMSClient(fl.client.NumPyClient):
    def __init__(self, shard_path: str):
        # Load local data
        df = pd.read_parquet(shard_path)
        self.X = joblib.load("./src/common/tfidf_vectorizer.pkl").transform(df["message"])
        # Encode labels: ham=0, spam=1
        self.y = df["label"].map({"ham": 0, "spam": 1}).values
        # Initialize fresh model with same hyperparams
        self.model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )

    def get_parameters(self):
        # Return as list of NumPy arrays
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        coef, intercept = parameters
        self.model.coef_ = np.array(coef)
        self.model.intercept_ = np.array(intercept)

    def fit(self, parameters, config):
        # Receive global parameters → set them
        self.set_parameters(parameters)
        # Local training (one epoch)
        self.model.fit(self.X, self.y)
        # Return updated parameters + number of samples
        return self.get_parameters(), len(self.X), {}

    def evaluate(self, parameters, config):
        # Use global params for local evaluation
        self.set_parameters(parameters)
        preds_proba = self.model.predict_proba(self.X)
        loss = log_loss(self.y, preds_proba)
        acc  = accuracy_score(self.y, self.model.predict(self.X))
        return float(loss), len(self.X), {"accuracy": float(acc)}

if __name__ == "__main__":
    import sys
    shard = sys.argv[1]  # e.g. "data/partitions/client_01.parquet"
    client = SMSClient(shard)
    fl.client.start_numpy_client(
        server_address="0.0.0.0:9092",
        client=client,
    )
