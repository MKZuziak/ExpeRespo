import sys
import os

import warnings
import flwr as fl
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils
import custom_classes

if __name__ == "__main__":
    print("""Starting the simulation, package information:
    Python: {},
    Flower: {},
    Numpy: {},
    Sklearn: {}.""".format(sys.version, fl.__version__, np.__version__, sklearn.__version__))
    
    
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    train_set = [utils.partition(X_train, y_train, 100)[partition_id] for partition_id in range(100)]
    test_set = [utils.partition(X_test, y_test, 100)[partition_id] for partition_id in range(100)]
    
    NUM_CLIENTS = 100

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, cid, model, train_data, test_data):
            self.model = model
            self.train_data = train_data
            self.test_data = test_data
            self.cid = cid

        def get_parameters(self, config): # type: ignore
            #print(f"[Client {self.cid}] get_parameters")
            return utils.get_model_parameters(self.model)

        def fit(self, parameters, config): # type: ignore
            # Read values from config
            server_round = config["server_round"]

            # Use values provided by the config
            #print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
            
            utils.set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(self.train_data[0], self.train_data[1])
            return utils.get_model_parameters(self.model), len(self.test_data), {}

        def evaluate(self, parameters, config): # type: ignore
            #print(f"[Client {self.cid}] evaluate, config: {config}")
            utils.set_model_params(self.model, parameters)
            loss = log_loss(self.test_data[1], self.model.predict_proba(self.test_data[0]))
            accuracy = self.model.score(self.test_data[0], self.test_data[1])
            return loss, len(self.test_data[0]), {"accuracy": accuracy, "cid":self.cid}

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = train_set[int(cid)]
        valloader = test_set[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(cid, model, trainloader, valloader)

    def weighted_average(metrics):
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

    # The `evaluate` function will be by Flower called after every round
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        evaluation_set = test_set[0]
        utils.set_model_params(model, parameters) # Update model with the latest parameters
        
        loss = log_loss(evaluation_set[1], model.predict_proba(evaluation_set[0]))
        accuracy = model.score(evaluation_set[0], evaluation_set[1])
        print(f"Server-side evaluation: accuracy {accuracy}")

        file_name = os.path.join("Metrics", "Server_Metrics.csv")
        with open(file_name, 'a') as server_f:
            server_f.write("{}, {} \n".format(server_round, accuracy))

        return loss, {"accuracy": accuracy}

    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,  # The current round of federated learning
        }
        return config

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
        )

    utils.set_initial_params(model)

    # Create FedAvg strategy
    strategy = custom_classes.AggregateCustomMetricStrategy(
            fraction_fit=0.1,  # Sample 100% of available clients for training
            fraction_evaluate=0.05,  # Sample 50% of available clients for evaluation
            min_fit_clients=10,  # Never sample less than 10 clients for training
            min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
            min_available_clients=10,  # Wait until all 10 clients are available
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(utils.get_model_parameters(model)), # Passing the initial set of parameters to the strategy.
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,  # Pass the fit_config function
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )