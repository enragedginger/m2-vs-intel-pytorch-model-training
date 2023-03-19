# Intel vs M2 Macbook Pytorch Model Training Benchmarks #

Apple advertises that the M2 GPU cores that you can use for ML.
I wanted some basic validation of this claim, but struggled to find any real explorations on this.

Pytorch supports training on M1 / M2 GPU cores through MPS.
I really wanted to kick in the tires on this to see if it's that much faster.
The goal of this project is to run a few basic Pytorch model training + evaluation samples to compare performance between an early-2023 Macbook with an M2 Max processor and a late-2019 Macbook with an Intel i9 processor.
Both machines were almost "maxed out" when purchased from Apple.

All code was provided by Bing AI Search because I didn't care enough to write any of this myself.
The code it produced after the first prompt never worked on the first try.
But after feeding error messages back to it 2-3 times, it got to the code you see in the repo.

This is about on par with my skills, but Bing AI was much faster.

These examples suggest a 1.7x - 3.5x speed up on the M2 Max over the Intel i9 (2.4 GHz, 8-core).

# Env Setup #
Uses pytorch 2!
* Install Poetry (I used v1.4.0)
* Install Python 3.11
* Run `poetry install`

# Examples #
This is pretty crude, but for timing, we're just using the "time" command.
Each example potentially downloads a dataset. I recommend running the example once to get the
dataset, and then running it a second time to get the approximate timing values.

## Example 1 ##
CNN with SGD. Uses the MNIST dataset.
* `time poetry run python model_training_test/train.py`
* M2: 124.96s
* Intel: 208.54s
* Speedup factor: 1.7x

## Example 2 ##
A different CNN with SGD, but this one doesn't try to manage moving data between devices
and uses the CIFAR10 dataset.
* `time poetry run python model_training_test/train2.py`
* M2: 40.73s
* Intel: 96.92s
* Speedup factor: 2.4x

## Example 3 ##
Run distributed training for MNIST using "Distributed Data Parallel" (DPP) and two workers.
Note: The model itself has a negative loss value and is hot-garbage.
* `time MASTER_ADDR=localhost MASTER_PORT=8083 poetry run python model_training_test/train3.py`
* M2: 28.13s
* Intel: 98.73s
* Speedup factor: 3.5x

## Example 4 ##
Another DPP example that uses four workers and an even simpler sequential NN on MNIST.
* `time MASTER_ADDR=localhost MASTER_PORT=8083 poetry run python model_training_test/train4.py`
* M2: 30.47s
* Intel: 108.34s
* Speedup factor: 3.5x
