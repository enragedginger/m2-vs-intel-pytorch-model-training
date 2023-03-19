import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist

def train(rank, world_size):
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size = 64
    lr = 0.01
    momentum = 0.5
    epochs = 10

    # Initialize DDP
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )

    # Define model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)

    # Wrap model in DDP
    model = DDP(model)

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Rank {rank}: Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    # Evaluate model
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Rank {rank}: Test set accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
