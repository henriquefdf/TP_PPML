# src/federated_learning/federated_simulation_phase1_dp.py

import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score
# NENHUMA importação de DP do Flower é necessária

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Carregamento e Preparação dos Dados (sem alterações)
print("Loading and preparing data for federation...")
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

train_df['cleaned_message'] = train_df['cleaned_message'].fillna('')
test_df['cleaned_message'] = test_df['cleaned_message'].fillna('')

vectorizer = TfidfVectorizer(max_features=3000)
X_train_full_tfidf = vectorizer.fit_transform(train_df['cleaned_message'])
y_train_full = train_df['label'].values

X_test_tfidf = vectorizer.transform(test_df['cleaned_message'])
y_test = test_df['label'].values

NUM_CLIENTS = 10
train_indices = list(range(len(train_df)))
np.random.shuffle(train_indices)
partition_size = len(train_indices) // NUM_CLIENTS
client_data_indices = [
    train_indices[i * partition_size: (i + 1) * partition_size]
    for i in range(NUM_CLIENTS)
]
print(f"Data partitioned for {NUM_CLIENTS} clients.")

# PARÂMETROS DE PRIVACIDADE DIFERENCIAL
L2_NORM_CLIP = 1.0 
NOISE_MULTIPLIER = 0.5 
NOISE_STD_DEV = L2_NORM_CLIP * NOISE_MULTIPLIER

# 2. Cliente Flower com Lógica de DP Manual
class SmsSpamClient(fl.client.NumPyClient):
    def __init__(self, client_id, X_train_tfidf, y_train, data_indices):
        self.client_id = client_id
        self.X_train_tfidf = X_train_tfidf
        self.y_train = y_train
        self.data_indices = data_indices
        self.model = LogisticRegression(
            class_weight='balanced', max_iter=100, warm_start=True, random_state=42
        )
        self.model.coef_ = np.zeros((1, self.X_train_tfidf.shape[1]))
        self.model.intercept_ = np.zeros(1)
        self.model.classes_ = np.array([0, 1])

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        # Armazena os parâmetros antigos (antes do treino)
        old_parameters = self.get_parameters(config)

        # Treina o modelo nos dados locais
        self.set_parameters(parameters)
        client_X = self.X_train_tfidf[self.data_indices]
        client_y = self.y_train[self.data_indices]
        self.model.fit(client_X, client_y)
        
        # Calcula a atualização (delta) dos parâmetros
        new_parameters = self.get_parameters(config)
        param_update = [new - old for new, old in zip(new_parameters, old_parameters)]

        # --- LÓGICA DE DP MANUAL ---
        # 1. Clipping: Limita a norma L2 da atualização
        # Achatamos os parâmetros em um único vetor para calcular a norma
        flat_update = np.concatenate([arr.flatten() for arr in param_update])
        l2_norm = np.linalg.norm(flat_update)
        clip_factor = min(1.0, L2_NORM_CLIP / (l2_norm + 1e-6))
        
        clipped_update = [arr * clip_factor for arr in param_update]

        # 2. Noising: Adiciona ruído gaussiano à atualização recortada
        noisy_update = []
        for arr in clipped_update:
            noise = np.random.normal(0, NOISE_STD_DEV, arr.shape)
            noisy_update.append(arr + noise)
        
        # Reconstrói os parâmetros finais somando a atualização com ruído aos parâmetros antigos
        final_parameters = [old + noisy for old, noisy in zip(old_parameters, noisy_update)]
        
        # Define os novos parâmetros no modelo
        self.set_parameters(final_parameters)
        
        # Retorna os parâmetros com ruído
        return self.get_parameters(config={}), client_X.shape[0], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        client_X = self.X_train_tfidf[self.data_indices]
        client_y = self.y_train[self.data_indices]
        loss = log_loss(client_y, self.model.predict_proba(client_X))
        accuracy = self.model.score(client_X, client_y)
        return loss, client_X.shape[0], {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    """Cria um cliente Flower para um determinado ID."""
    return SmsSpamClient(cid, X_train_full_tfidf, y_train_full, client_data_indices[int(cid)])

# 3. Função de Avaliação do Lado do Servidor
def get_evaluate_fn():
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        model = LogisticRegression(class_weight='balanced', max_iter=100, warm_start=True, random_state=42)
        model.classes_ = np.array([0, 1])
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]
        y_pred = model.predict(X_test_tfidf)
        loss = log_loss(y_test, model.predict_proba(X_test_tfidf))
        accuracy = model.score(X_test_tfidf, y_test)
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

# 4. Estratégia (FedAvg simples é suficiente agora)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(),
)

# 5. Iniciar Simulação (usando a API de simulação legada, mais simples e robusta)
print("Starting Manually-Implemented Differentially-Private Federated Learning simulation...")
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

print("--- Federated Learning Simulation Complete ---")
print("History (loss, metrics):", history.losses_centralized, history.metrics_centralized)

if history.metrics_centralized and 'accuracy' in history.metrics_centralized and history.metrics_centralized['accuracy']:
    final_accuracy = history.metrics_centralized['accuracy'][-1][1]
    print(f"\nFinal centralized accuracy after 5 rounds with DP: {final_accuracy:.4f}")
else:
    print("\nCould not determine final accuracy.")
print("Compare this to the centralized baseline accuracy of 0.9812 and non-private FL.")