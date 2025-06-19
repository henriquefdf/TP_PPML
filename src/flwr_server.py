# src/flwr_server.py

import flwr as fl

def main():
    # Configure FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,          # 50% of clients per round
        fraction_evaluate=0.5,     # 50% of clients for evaluation
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=None,  # default
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:9092",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
