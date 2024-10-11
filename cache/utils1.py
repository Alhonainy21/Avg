from collections import OrderedDict
from pathlib import Path
from typing import Tuple
import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import densenet121, mobilenet_v2
from efficientnet_pytorch import EfficientNet

DATA_ROOT = Path("/lung_3_sub")

class Net(nn.Module):
    """Your custom Net class remains unchanged."""
    # Your Net class definition here...


def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    
def ResNet18():
    model = resnet18(num_classes=3)

    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    model.maxpool = torch.nn.Identity()

    return model

def ResNet50():
    """Returns a ResNet50 model from TorchVision adapted for CIFAR-10."""

    model = resnet50(num_classes=3)

    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    model.maxpool = torch.nn.Identity()

    return model

def DenseNet121():
    """Returns a DenseNet121 model from TorchVision adapted for CIFAR-10."""
    model = densenet121(num_classes=3)

    # replace w/ smaller input layer
    model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features.conv0.weight, mode="fan_out", nonlinearity="relu")

    return model


def MobileNetV2():
    """Returns a MobileNetV2 model from TorchVision adapted for CIFAR-10."""
    model = mobilenet_v2(num_classes=3)

    # replace w/ smaller input layer
    model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.features[0][0].weight, mode="fan_out", nonlinearity="relu")

    return model

def EfficientNetB0():
    """Returns an EfficientNetB0 model from efficientnet_pytorch adapted for CIFAR-10."""
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

    # replace w/ smaller input layer
    model._conv_stem = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model._conv_stem.weight, mode="fan_out", nonlinearity="relu")

    return model

def load_model(model_name: str) -> nn.Module:
    if model_name == "Net":
        return Net()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "DenseNet121":
        return DenseNet121()
    elif model_name == "MobileNetV2":
        return MobileNetV2()
    elif model_name == "EfficientNetB0":
        return EfficientNetB0()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

def load_cifar(download=True) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    training_set = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testing_set = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    classes = torch.tensor([0, 1, 2, 3, 4])
    
    indices_t = (torch.tensor(training_set.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    indices_s = (torch.tensor(testing_set.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    
    trainset = torch.utils.data.Subset(training_set, indices_t)
    testset = torch.utils.data.Subset(testing_set, indices_s)

    return trainset, testset

def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, ascii=True), 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print(f"Epoch took: {time() - t:.2f} seconds")
    test_loss, test_accuracy = test(net, testloader, device)
    print(" ")
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print(" ")

def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
