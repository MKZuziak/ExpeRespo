import warnings
from xml.dom.minidom import Element
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        acc = [r.metrics["accuracy"]for _, r in results]
        acc_str = ','.join([str(example) for example in acc])
        ex_str = ','.join([str(example) for example in examples])

        # Aggregate and print custom metric
        with open('Aggregated_Metrics.csv', "a") as metric_f:
            metric_f.write("{},{},{}\n".format(server_round, acc_str, ex_str))
        
        with open("Aggregation_Log.txt", 'a') as agre_f:
            agre_f.write("Server round:{} results:{}\n".format(server_round, results))

        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}