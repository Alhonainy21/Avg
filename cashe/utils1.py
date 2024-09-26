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
