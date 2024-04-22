import argparse
import timeit
from collections import OrderedDict
from importlib import import_module
import time
import flwr as fl
import numpy as np
import torch
import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from tqdm import tqdm
import utils
from utils import get_weights, set_weights
from torch.utils.data import DataLoader, random_split


# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import socket
import fcntl
import struct

def get_ip_address(ifname='enp24s0f1'):
    """Get the local IP address of the client from a specific interface."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Use the SIOCGIFADDR ioctl to get the IP address for the specified interface
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', bytes(ifname[:15], 'utf-8'))
        )[20:24])
    except Exception as e:
        print(f"Exception encountered: {e}")
        return '127.0.0.1'
    finally:
        s.close()

# Example usage
interface_name = 'enp24s0f1'
ip_address = get_ip_address(interface_name)
print(f"IP address of {interface_name}: {ip_address}")


class CifarClient(fl.client.Client):

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        trainset: torchvision.datasets.CIFAR10,
        testset: torchvision.datasets.CIFAR10,
        val_ratio: float = 0.1,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.previous_loss = float('inf')
        self.force_update_next_round = False
        self.current_round = 0
        self.total_rounds = 0
        self.client_update_server = 0
        self.no_update_this_round = 0

        val_size = int(len(trainset) * val_ratio)
        train_size = len(trainset) - val_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def _instantiate_model(self, model_str: str):

        # will load utils.model_str
        m = getattr(import_module("utils"), model_str)
        # instantiate model
        self.model = m()

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")
        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()
        client_ip = get_ip_address()
        # Initialize epochs from the server's configuration or use a default value
        self.epochs = int(config.get("epochs", 2))
        self.total_rounds = int(config.get ("total_rounds"))

        # Set model parameters
        set_weights(self.model, weights)

        # Prepare DataLoader kwargs
        kwargs = {"num_workers": int(config.get("num_workers", 0)), "pin_memory": bool(config.get("pin_memory", True)), "drop_last": True} if torch.cuda.is_available() else {"drop_last": True}
        train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy = self.train_and_evaluate(self.train_loader, self.val_loader, self.test_loader)
        ##########################################################################
        improvement = self.previous_loss - val_loss
        self.previous_loss = val_loss
        self.current_round += 1
        improvement = 1
        if self.force_update_next_round:
            # If the previous round sent an empty list, send actual weights this time
            print(f"Client {self.cid}: Force updating server with actual weights this round.")
            weights_prime = get_weights(self.model)
            self.client_update_server +=1
            self.force_update_next_round = False  # Reset the flag since we're fulfilling the forced update
        elif improvement <= 0.1:
            # Check if improvement is insufficient and we haven't just sent an empty list
            print(f"Client {self.cid}: Insufficient improvement, sending empty weights list.")
            weights_prime = []
            self.no_update_this_round +=1
            # Send an empty list of weights for this round
            self.force_update_next_round = True  # Ensure we send actual weights in the next round
        else:
            # Sufficient improvement, or it's a normal round without forced conditions
            print(f"Client {self.cid}: Updating server with actual weights.")
            weights_prime = get_weights(self.model)
            self.client_update_server +=1
        ############################################################################

        params_prime = fl.common.weights_to_parameters(weights_prime)

        num_examples_train = len(self.trainset)

        metrics = {
            "duration": timeit.default_timer() - fit_begin, 
            "train_loss": train_loss, 
            "train_accuracy": train_accuracy, 
            "test_loss": test_loss, 
            "test_accuracy": test_accuracy,
            "client_ip": client_ip,
        }
        if self.current_round >= self.total_rounds:
            print(f"#########_client update server ={self.client_update_server}")
            print(f"#########_client did not send = {self.no_update_this_round}")
        return FitRes(parameters=params_prime, num_examples=num_examples_train, metrics=metrics)


    def evaluate_dataset(self, loader, criterion):
        """Evaluate the model on a given dataset."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = total_correct / len(loader.dataset)
        return avg_loss, accuracy

    def train_and_evaluate(self, train_loader, val_loader, test_loader):
        """Train the model and evaluate it to return the training and test accuracy."""
        self.model.train()  # Set model to training mode
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.epochs):
            for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True):
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        train_loss, train_accuracy = self.evaluate_dataset(train_loader, criterion)
        val_loss, val_accuracy = self.evaluate_dataset(val_loader, criterion)
        test_loss, test_accuracy = self.evaluate_dataset(test_loader, criterion)
        print(f"Epoch {epoch+1}/{self.epochs}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        client_ip = get_ip_address()
        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = utils.test(self.model, testloader, device=DEVICE)
        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {
        "accuracy": float(accuracy),
        "client_ip": client_ip,  # Add the client IP address here
         }
        return EvaluateRes(loss=float(loss), num_examples=len(self.testset), metrics=metrics)


def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the dataset lives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Net",
        choices=["Net", "ResNet18","ResNet50","DenseNet121","MobileNetV2","EfficientNetB0"],
        help="model to train",
    )
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # model
    model = utils.load_model(args.model)
    model.to(DEVICE)
    # load (local, on-device) dataset
    trainset, testset = utils.load_cifar()

    # Start client
    client = CifarClient(args.cid, model, trainset, testset)
    fl.client.start_client(args.server_address, client)
    #print_test_accuracy()


if __name__ == "__main__":
    main()
