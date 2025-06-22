# src/federated_learning/federated_simulation_phase1.py

import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Researcher's Justification ---
# In a real FL system, each client would have its own local data. For this simulation,
# we centrally load the preprocessed data and then partition it among virtual clients.
# This allows us to control the simulation environment and reproduce experiments.
# We also pre-fit a global TF-IDF vectorizer. This is a necessary simplification for this phase.
# In a true production system where client vocabularies are unknown, more advanced
# techniques like using pre-trained word embeddings (our Phase 2) or federated
# vocabulary generation would be required.
# ------------------------------------

# 1. Centralized Data Loading and Pre-processing
print("Loading and preparing data for federation...")
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Handle potential NaNs
train_df['cleaned_message'] = train_df['cleaned_message'].fillna('')
test_df['cleaned_message'] = test_df['cleaned_message'].fillna('')

# Fit a global TF-IDF vectorizer on the entire training data
vectorizer = TfidfVectorizer(max_features=3000)
X_train_full_tfidf = vectorizer.fit_transform(train_df['cleaned_message'])
y_train_full = train_df['label'].values

# Transform the test set using the same vectorizer for consistent evaluation
X_test_tfidf = vectorizer.transform(test_df['cleaned_message'])
y_test = test_df['label'].values

# Partition the training data among clients
NUM_CLIENTS = 10
train_indices = list(range(len(train_df)))
# We shuffle the indices to simulate a random distribution of data
np.random.shuffle(train_indices)
partition_size = len(train_indices) // NUM_CLIENTS
client_data_indices = [
    train_indices[i * partition_size: (i + 1) * partition_size]
    for i in range(NUM_CLIENTS)
]
print(f"Data partitioned for {NUM_CLIENTS} clients.")

# 2. Define the Flower Client
# --- Researcher's Justification ---
# The Flower client (`NumPyClient`) is the core component that lives on the user's device.
# It needs to define three key methods:
# - get_parameters: Returns the local model's weights to the server.
# - set_parameters: Updates the local model with weights received from the server.
# - fit: Trains the local model on its local data and returns the updated weights.
# We use scikit-learn's LogisticRegression. Since it doesn't support incremental updates
# like a neural network, our `fit` method will re-train the model on the local data
# for a single epoch in each round, starting with the parameters from the server.
# ------------------------------------
class SmsSpamClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train_tfidf, y_train, data_indices):
        self.client_id = client_id
        self.X_train_tfidf = X_train_tfidf
        self.y_train = y_train
        self.data_indices = data_indices
        self.model = LogisticRegression(
            class_weight='balanced', max_iter=100, warm_start=True, random_state=42
        )
        # Set initial parameters to zeros
        self.model.coef_ = np.zeros((1, self.X_train_tfidf.shape[1]))
        self.model.intercept_ = np.zeros(1)
        self.model.classes_ = np.array([0, 1])

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy ndarrays."""
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays."""
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        """Train model on the locally held dataset."""
        self.set_parameters(parameters)
        
        # Get this client's partition of data
        client_X = self.X_train_tfidf[self.data_indices]
        client_y = self.y_train[self.data_indices]
        
        # Train the model for one local epoch
        self.model.fit(client_X, client_y)
        
        # Return the updated parameters and the number of examples used for training
        return self.get_parameters(config={}), client_X.shape[0], {}

    def evaluate(self, parameters, config):
        """Evaluate the provided parameters on the local validation set."""
        # Note: In this simulation, we do a centralized evaluation on the server.
        # This method is included for completeness but won't be the primary
        # source of our final metrics.
        self.set_parameters(parameters)
        client_X = self.X_train_tfidf[self.data_indices]
        client_y = self.y_train[self.data_indices]
        
        loss = log_loss(client_y, self.model.predict_proba(client_X))
        accuracy = self.model.score(client_X, client_y)
        
        return loss, client_X.shape[0], {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client for a given client ID."""
    return SmsSpamClient(cid, X_train_full_tfidf, y_train_full, client_data_indices[int(cid)])

# 3. Define the Server-Side Evaluation Function
# --- Researcher's Justification ---
# Centralized evaluation provides a consistent and unbiased measure of the global model's
# performance throughout the training process. We use our held-out test set for this.
# This function will be called by the Strategy on the server after each round of training.
# This is how we will track if our federated model is learning effectively.
# ------------------------------------
def get_evaluate_fn():
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = LogisticRegression(
            class_weight='balanced', max_iter=100, warm_start=True, random_state=42
        )
        model.classes_ = np.array([0, 1])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

        # Predict on the centralized test set
        y_pred = model.predict(X_test_tfidf)

        # Calculate all relevant metrics
        loss = log_loss(y_test, model.predict_proba(X_test_tfidf))
        accuracy = model.score(X_test_tfidf, y_test)
        # Use pos_label=1 to calculate metrics for the 'spam' class
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        print(
            f"Server-side evaluation round {server_round} - "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return evaluate


# 4. Define the Federated Averaging (FedAvg) Strategy
# --- Researcher's Justification ---
# FedAvg is the canonical federated learning algorithm. It averages the model parameters
# contributed by the selected clients in each round.
# - min_fit_clients: Number of clients to participate in training each round.
# - min_available_clients: Minimum number of clients that must be connected to start a round.
# - evaluate_fn: The server-side evaluation function we defined above.
# We sample 100% of clients each round (fraction_fit=1.0) since we only have 10.
# ------------------------------------
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)

# 5. Start the Simulation
# --- Researcher's Justification ---
# `fl.simulation` is Flower's tool for running simulations on a single machine.
# It orchestrates the entire process: spinning up a "virtual" server, creating clients
# via `client_fn`, running the specified number of rounds, and collecting results.
# This is the ideal way to test and debug an FL system before deploying to real devices.
# ------------------------------------
print("Starting Federated Learning simulation...")
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5), # Let's run for 5 rounds
    strategy=strategy,
)

print("--- Federated Learning Simulation Complete ---")
print("History (loss, metrics):", history.losses_centralized, history.metrics_centralized)

# To get the final accuracy, we can look at the last entry in the metrics history
final_accuracy = history.metrics_centralized['accuracy'][-1][1]
print(f"\nFinal centralized accuracy after 5 rounds: {final_accuracy:.4f}")
print("Compare this to the centralized baseline accuracy of 0.9812.")