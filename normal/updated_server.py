import argparse
from collections import OrderedDict
import psutil
import logging
import numpy as np
import torch
import torchvision
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.client_manager import SimpleClientManager
import utils
from typing import List, Tuple, Dict, Callable, Optional, Union
from flwr.server.client_proxy import ClientProxy



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--server_address", type=str, required=True, help="gRPC server address")
parser.add_argument("--rounds", type=int, default=1, help="Number of rounds of federated learning (default: 1)")
parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of available clients used for fit/evaluate (default: 1.0)")
parser.add_argument("--min_sample_size", type=int, default=8, help="Minimum number of clients used for fit/evaluate (default: 2)")
parser.add_argument("--min_num_clients", type=int, default=2, help="Minimum number of available clients required for sampling (default: 2)")
parser.add_argument("--log_host", type=str, help="Logserver address (no default)")
parser.add_argument("--model", type=str, default="ResNet18", choices=["Net", "ResNet18", "ResNet50", "DenseNet121", "MobileNetV2", "EfficientNetB0"], help="Model to train")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataset reading")
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

class MinimalCustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _log_memory_usage(self):
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024 ** 3)  # Convert bytes to gigabytes
        total_gb = memory.total / (1024 ** 3)  # Convert bytes to gigabytes
        print("=========================================================")
        print(f"Memory Usage: {memory.percent}% used of {total_gb:.2f}GB (Used: {used_gb:.2f} GB)")
        print("=========================================================")


    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, fl.common.FitRes]], failures):
        print(f"Memory before round {rnd}:")
        self._log_memory_usage()
        
        aggregated_weights = None
        total_data_points = 0
        all_weights = []
        
        for client_proxy, fit_res in results:
            # print(f"Memory before processing client:")
            # self._log_memory_usage()
            weights = parameters_to_weights(fit_res.parameters)
            # print(f"Memory after processing client:")
            # self._log_memory_usage()
            weighted_weights = [np.array(w) * fit_res.num_examples for w in weights]
            all_weights.append(weighted_weights)
            total_data_points += fit_res.num_examples

        if all_weights:
            num_layers = len(all_weights[0])
            aggregated_weights = [sum([weights[layer] for weights in all_weights]) / total_data_points for layer in range(num_layers)]
            aggregated_parameters = weights_to_parameters(aggregated_weights)

        print(f"Memory after round {rnd}:")
        self._log_memory_usage()

        return aggregated_parameters if aggregated_weights else None, {}

def main():
    fl.common.logger.configure("server", host=args.log_host)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logging.getLogger('').addHandler(console)

    _, testset = utils.load_cifar()
    client_manager = SimpleClientManager()
    strategy = MinimalCustomFedAvg(fraction_fit=args.sample_fraction, min_fit_clients=args.min_sample_size, min_available_clients=args.min_num_clients, eval_fn=get_eval_fn(testset), on_fit_config_fn=fit_config)
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    fl.server.start_server(server_address=args.server_address, server=server, config={"num_rounds": args.rounds})

def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    # print(f"Memory before setting config for round {server_round}:")
    # custom_fed_avg_instance = MinimalCustomFedAvg()                                                                                                                              
    # custom_fed_avg_instance._log_memory_usage()
    config = {"epoch_global": str(server_round), "epochs": str(3), "batch_size": str(args.batch_size), "num_workers": str(args.num_workers), "pin_memory": str(args.pin_memory), "total_rounds": str(args.rounds)}
    # print(f"Memory after setting config for round {server_round}:")
    # custom_fed_avg_instance._log_memory_usage()
    return config

def get_eval_fn(testset: torchvision.datasets.CIFAR10) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=DEVICE)
        return loss, {"accuracy": accuracy}
    return evaluate

def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    state_dict = OrderedDict({k: torch.tensor(np.atleast_1d(v)) for k, v in zip(model.state_dict().keys(), weights)})
    model.load_state_dict(state_dict)

if __name__ == "__main__":
    main()