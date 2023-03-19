import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train(rank, world_size):
    # Initialize the distributed environment.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Set up the model.
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )

    # Wrap the model with DDP.
    ddp_model = DDP(model)

    # Set up the optimizer.
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Set up the data.
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        sampler=train_sampler
    )

    # Train the model.
    for epoch in range(10):
        for batch, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = ddp_model(data.view(-1, 784))
            loss = nn.functional.binary_cross_entropy(output, target.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            # Print out the training progress.
            if batch % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch}, Loss {loss.item()}")

if __name__ == "__main__":
    # Set up the distributed environment.
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
