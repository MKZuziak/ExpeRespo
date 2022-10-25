# First simulation from 18.10.2022
# 261 participants (some of the data points may be doubled)


from cgi import test
from tabnanny import verbose
import warnings
import flwr as fl
import numpy as np
from Vast_Flower.FinStat_Flower.utils import get_model_parameters

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils

if __name__ == "__main__":

    load_path = r'.\array_limited_100_v2.npy'
    # For now, the test set is a ragged sequence of a shape (261, 2). Each tuple (X, y) nested in the list contains a three dimensional list of observations (so the shape is [n, 3]
    # where n stands for the number of observations in that tuple) and corresponding list of dependant variables (y).
    # Each tuple represents a set of observations that belongs to certain participant. So the train_set[0] will be a set of observations (X, y) that belongs to the user 0. 
    
    data_set = np.load(load_path, allow_pickle=True)

    train_set = data_set[0]
    test_set = data_set[1]

    

    model = LogisticRegression(
    penalty="l2",
    max_iter=100,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
    )

    NUM_CLIENTS = 100
    utils.set_initial_params(model)

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, model, train_data, test_data):
            self.model = model
            self.train_data = train_data
            self.test_data = test_data

            self.X_train = train_data[0]
            self.y_train = train_data[1]

            self.X_test = test_data[0]
            self.y_test = test_data[1]

            #print("Model initialized with training dataset: {} and {}, testing dataset: {} and {}".format(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape))

        def get_parameters(self, config): # type: ignore
            return utils.get_model_parameters(self.model)

        def fit(self, parameters, config): # type: ignore
            utils.set_model_params(self.model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            return utils.get_model_parameters(self.model), len(self.y_train), {}

        def evaluate(self, parameters, config): # type: ignore
            utils.set_model_params(self.model, parameters)
            loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
            accuracy = self.model.score(self.X_test, self.y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}

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

# Create FedAvg strategy
params = get_model_parameters(model)
strategy = fl.server.strategy.FedAdagrad(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.2,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 1 client for training
        min_evaluate_clients=1,  # Never sample less than 1 client for evaluation
        min_available_clients=1,  # Wait until all 1 client are available
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)